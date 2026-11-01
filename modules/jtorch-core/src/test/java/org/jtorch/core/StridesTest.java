package org.jtorch.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class StridesTest {
    @Test
    void contiguousStridesAreRowMajor() {
        assertArrayEquals(new long[] {3, 1}, Strides.contiguous(Shape.of(2, 3)).toArray());
        assertArrayEquals(new long[] {24, 8, 1}, Strides.contiguous(Shape.of(2, 3, 8)).toArray());
        assertArrayEquals(new long[0], Strides.contiguous(Shape.scalar()).toArray());
    }

    @Test
    void emptyTensorsAreContiguous() {
        assertTrue(Strides.of(0, 0).isContiguous(Shape.of(0, 3)));
        assertTrue(Strides.of(1, 1).isContiguous(Shape.of(2, 0)));
    }

    @Test
    void sizeOneDimensionsDoNotAffectContiguity() {
        assertTrue(Strides.of(3, 999, 1).isContiguous(Shape.of(2, 1, 3)));
        assertFalse(Strides.of(6, 3, 1).isContiguous(Shape.of(2, 1, 3)));
    }

    @Test
    void nonContiguousStridesAreDetected() {
        assertFalse(Strides.of(1, 2).isContiguous(Shape.of(2, 3)));
        assertFalse(Strides.of(6, 1).isContiguous(Shape.of(2, 3)));
        assertFalse(Strides.of(1, 2).isContiguous(Shape.of(3, 2)));
    }

    @Test
    void negativeStridesAreRejected() {
        assertThrows(ShapeException.class, () -> Strides.of(-1, 1));
    }

    @Test
    void constructionIsDefensive() {
        long[] source = {3, 1};
        Strides strides = Strides.of(source);
        source[0] = 100;
        assertEquals(3, strides.stride(0));
        long[] copy = strides.toArray();
        copy[0] = 7;
        assertEquals(3, strides.stride(0));
    }
}
