package org.jtorch.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DTypeTest {
    @Test
    void dtypeProperties() {
        assertEquals(4, DType.FLOAT32.sizeBytes());
        assertEquals(1, DType.BOOL.sizeBytes());
        assertTrue(DType.FLOAT64.isFloatingPoint());
        assertTrue(DType.INT32.isIntegral());
        assertTrue(DType.BOOL.isBoolean());
        assertFalse(DType.INT64.isFloatingPoint());
    }

    @Test
    void parseAcceptsPyTorchSpellings() {
        assertEquals(DType.FLOAT32, DType.parse("float32"));
        assertEquals(DType.FLOAT32, DType.parse("torch.float"));
        assertEquals(DType.FLOAT64, DType.parse("double"));
        assertEquals(DType.INT64, DType.parse("long"));
        assertEquals(DType.BOOL, DType.parse("bool"));
        assertThrows(DTypeException.class, () -> DType.parse("float16"));
    }

    @Test
    void tensorPromotionLattice() {
        assertEquals(DType.INT32, DType.resultType(DType.BOOL, DType.INT32));
        assertEquals(DType.INT64, DType.resultType(DType.INT32, DType.INT64));
        assertEquals(DType.FLOAT32, DType.resultType(DType.INT64, DType.FLOAT32));
        assertEquals(DType.FLOAT64, DType.resultType(DType.FLOAT32, DType.FLOAT64));
        assertEquals(DType.FLOAT32, DType.resultType(DType.FLOAT32, DType.FLOAT32));
    }

    @Test
    void weakScalarPromotionMatchesPyTorch() {
        // Integer scalars never widen a floating tensor.
        assertEquals(DType.FLOAT32, DType.resultTypeWithScalar(DType.FLOAT32, Scalar.of(2)));
        // Float scalars lift integer tensors to the default float dtype.
        assertEquals(DType.FLOAT32, DType.resultTypeWithScalar(DType.INT32, Scalar.of(0.5)));
        assertEquals(DType.FLOAT32, DType.resultTypeWithScalar(DType.INT64, Scalar.of(0.5)));
        // Integer scalars leave integer tensors unchanged.
        assertEquals(DType.INT32, DType.resultTypeWithScalar(DType.INT32, Scalar.of(2)));
        // Integer scalars lift bool tensors to int64.
        assertEquals(DType.INT64, DType.resultTypeWithScalar(DType.BOOL, Scalar.of(1)));
        // Float scalars keep float64 precision.
        assertEquals(DType.FLOAT64, DType.resultTypeWithScalar(DType.FLOAT64, Scalar.of(0.5)));
    }
}
