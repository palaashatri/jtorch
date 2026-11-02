package org.jtorch.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ShapeTest {
    @Test
    void scalarShapeHasOneElement() {
        assertEquals(0, Shape.scalar().rank());
        assertEquals(1, Shape.scalar().numel());
    }

    @Test
    void numelMultipliesDimensions() {
        assertEquals(6, Shape.of(2, 3).numel());
        assertEquals(0, Shape.of(2, 0, 3).numel());
        assertEquals(1, Shape.of(1, 1).numel());
    }

    @Test
    void numelOverflowIsRejected() {
        ShapeException error = assertThrows(ShapeException.class, () -> Shape.of(Long.MAX_VALUE, 2));
        assertTrue(error.getMessage().contains("overflows"), error.getMessage());
    }

    @Test
    void negativeDimensionsAreRejected() {
        assertThrows(ShapeException.class, () -> Shape.of(2, -1));
    }

    @Test
    void constructionIsDefensive() {
        long[] source = {2, 3};
        Shape shape = Shape.of(source);
        source[0] = 99;
        assertEquals(2, shape.dim(0));
        long[] copy = shape.toArray();
        copy[1] = 42;
        assertEquals(3, shape.dim(1));
    }

    @Test
    void valueSemantics() {
        assertEquals(Shape.of(2, 3), Shape.of(2, 3));
        assertEquals(Shape.of(2, 3).hashCode(), Shape.of(2, 3).hashCode());
        assertNotEquals(Shape.of(2, 3), Shape.of(3, 2));
    }

    @Test
    void negativeDimensionNormalisationMatchesPyTorch() {
        Shape shape = Shape.of(2, 3, 4);
        assertEquals(2, shape.normalizeDim(-1));
        assertEquals(0, shape.normalizeDim(-3));
        assertThrows(IndexOutOfBoundsException.class, () -> shape.normalizeDim(3));
        assertThrows(IndexOutOfBoundsException.class, () -> shape.normalizeDim(-4));
    }

    @Test
    void broadcastingFollowsRightAlignedRules() {
        assertEquals(Shape.of(2, 3), Shape.broadcast(Shape.of(2, 3), Shape.of(3)));
        assertEquals(Shape.of(2, 3), Shape.broadcast(Shape.of(2, 1), Shape.of(1, 3)));
        assertEquals(Shape.of(5), Shape.broadcast(Shape.scalar(), Shape.of(5)));
        assertThrows(ShapeException.class, () -> Shape.broadcast(Shape.of(2, 3), Shape.of(2)));
        assertThrows(ShapeException.class, () -> Shape.broadcast(Shape.of(2, 3), Shape.of(3, 2)));
    }

    @Test
    void dimensionEditing() {
        assertEquals(Shape.of(3, 4), Shape.of(2, 3, 4).withoutDim(0));
        assertEquals(Shape.of(2, 1, 3, 4), Shape.of(2, 3, 4).withDim(1, 1));
        assertEquals(Shape.of(2, 5, 4), Shape.of(2, 3, 4).withSize(1, 5));
    }

    @Test
    void toStringUsesTorchSizeSyntax() {
        assertEquals("torch.Size([2, 3])", Shape.of(2, 3).toString());
    }
}
