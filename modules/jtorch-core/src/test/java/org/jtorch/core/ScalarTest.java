package org.jtorch.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ScalarTest {
    @Test
    void integerAndFloatKindsArePreserved() {
        assertTrue(Scalar.of(2).isIntegral());
        assertTrue(Scalar.of(2.0).isFloatingPoint());
        assertTrue(Scalar.of(true).isBoolean());
        assertEquals(2.5, Scalar.of(2.5).toDouble());
        assertEquals(2, Scalar.of(2.9).toLong());
    }

    @Test
    void booleanConversionsUseNonZeroSemantics() {
        assertTrue(Scalar.of(2).toBoolean());
        assertFalse(Scalar.of(0).toBoolean());
        assertTrue(Scalar.of(0.5).toBoolean());
    }

    @Test
    void castingToDtypeUsesTruncation() {
        assertEquals(2.0, Scalar.of(2.9).toDouble(DType.INT64));
        assertEquals(1.0, Scalar.of(2).toDouble(DType.BOOL));
        assertEquals(2.9, Scalar.of(2.9).toDouble(DType.FLOAT64));
    }

    @Test
    void parseDistinguishesIntegersFromFloats() {
        assertTrue(Scalar.parse("3").isIntegral());
        assertTrue(Scalar.parse("3.5").isFloatingPoint());
        assertTrue(Scalar.parse("true").isBoolean());
        assertThrows(DTypeException.class, () -> Scalar.parse("nope"));
    }

    @Test
    void valueEqualityUsesKind() {
        assertEquals(Scalar.of(1), Scalar.of(1));
        assertNotEquals(Scalar.of(1), Scalar.of(1.0));
    }
}
