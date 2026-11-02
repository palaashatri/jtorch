package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class DTypeOpsTest {
    private static TensorOptions f32() {
        return TensorOptions.defaults();
    }

    @Test
    void int64ArithmeticIsExactBeyondDoublePrecision() {
        long big = (1L << 60) + 1;
        Tensor a = Torch.tensor(new long[] {big, 3}, Shape.of(2));
        Tensor b = Torch.tensor(new long[] {1, 4}, Shape.of(2));
        assertArrayEquals(new long[] {big + 1, 7}, a.add(b).toLongArray());
        assertEquals(DType.INT64, a.add(b).dtype());
    }

    @Test
    void promotionFollowsTensorLattice() {
        Tensor int32 = Torch.tensor(new int[] {1, 2}, Shape.of(2));
        Tensor int64 = Torch.tensor(new long[] {3, 4}, Shape.of(2));
        Tensor float32 = Torch.tensor(new float[] {0.5f, 1.5f}, Shape.of(2));
        Tensor float64 = Torch.tensor(new double[] {0.25, 0.75}, Shape.of(2));
        assertEquals(DType.INT64, int32.add(int64).dtype());
        assertEquals(DType.FLOAT32, int64.add(float32).dtype());
        assertEquals(DType.FLOAT64, float32.add(float64).dtype());
        assertEquals(DType.BOOL, int32.eq(int64).dtype());
    }

    @Test
    void trueDivisionPromotesIntegersToDefaultFloat() {
        Tensor a = Torch.tensor(new long[] {1, 3}, Shape.of(2));
        Tensor b = Torch.tensor(new long[] {2, 2}, Shape.of(2));
        Tensor quotient = a.div(b);
        assertEquals(DType.FLOAT32, quotient.dtype());
        assertArrayEquals(new double[] {0.5, 1.5}, quotient.toDoubleArray(), 1e-7);
    }

    @Test
    void booleanArithmeticUsesLogicalSemantics() {
        Tensor yes = Torch.tensor(new boolean[] {true, true, false}, Shape.of(3));
        Tensor no = Torch.tensor(new boolean[] {true, false, false}, Shape.of(3));
        assertArrayEquals(new boolean[] {true, true, false}, yes.add(no).toBooleanArray(), "bool add");
        assertArrayEquals(new boolean[] {true, false, false}, yes.mul(no).toBooleanArray(), "bool mul");
        assertArrayEquals(new boolean[] {false, false, true}, yes.logicalNot().toBooleanArray(), "bool not");
        assertEquals(DType.BOOL, yes.add(no).dtype());
    }

    @Test
    void reductionsFollowPyTorchAccumulationDtypes() {
        Tensor bools = Torch.tensor(new boolean[] {true, true, false}, Shape.of(3));
        Tensor sum = bools.sum();
        assertEquals(DType.INT64, sum.dtype());
        assertEquals(2, sum.itemLong());

        Tensor ints = Torch.tensor(new int[] {1, 2, 3}, Shape.of(3));
        assertEquals(DType.INT64, ints.sum().dtype());
        assertThrows(DTypeException.class, ints::mean);
    }

    @Test
    void emptyReductions() {
        Tensor empty = Torch.zeros(Shape.of(0));
        assertEquals(0, empty.sum().item(), 0);
        assertTrue(Double.isNaN(empty.mean().item()));
        assertThrows(ShapeException.class, empty::max);
        Tensor matrix = Torch.zeros(Shape.of(0, 3));
        assertEquals(Shape.of(1, 3), matrix.sum(0, true).shape());
        assertEquals(Shape.of(3), matrix.sum(0, false).shape());
    }

    @Test
    void maxAndMinPropagateNaN() {
        Tensor values = Torch.tensor(new float[] {1, Float.NaN, 3}, Shape.of(3));
        assertTrue(Float.isNaN(values.max().itemFloat()));
        assertTrue(Float.isNaN(values.min().itemFloat()));
        Tensor matrix = Torch.tensor(new float[] {1, Float.NaN, 3, 4}, Shape.of(2, 2));
        assertTrue(Float.isNaN(matrix.maxWithIndices(1, false).values().toFloatArray()[0]));
    }

    @Test
    void argmaxUsesFirstOccurrenceAndFlattenedIndexing() {
        Tensor values = Torch.tensor(new float[] {1, 3, 3, 2}, Shape.of(2, 2));
        assertEquals(1, values.argmax().itemLong());
        Tensor.TensorTuple result = values.maxWithIndices(1, false);
        assertArrayEquals(new double[] {3, 3}, result.values().toDoubleArray(), 0);
        assertArrayEquals(new long[] {1, 0}, result.indices().toLongArray());
    }

    @Test
    void int64ExtremaAreExact() {
        long base = 1L << 60;
        Tensor values = Torch.tensor(new long[] {base, base + 1, base - 1}, Shape.of(3));
        assertEquals(base + 1, values.max().itemLong());
        assertEquals(base - 1, values.min().itemLong());
        Tensor negatives = Torch.tensor(new long[] {-5, -2, -7}, Shape.of(3));
        assertEquals(-2, negatives.max().itemLong());
        assertEquals(-7, negatives.min().itemLong());
        assertEquals(1, values.argmax().itemLong());
        assertEquals(2, values.argmin().itemLong());
        Tensor.TensorTuple result = values.reshape(1, 3).maxWithIndices(1, false);
        assertEquals(base + 1, result.values().itemLong());
        assertEquals(1, result.indices().itemLong());
    }

    @Test
    void invalidArgumentsAreRejectedWithCategories() {
        Tensor a = Torch.zeros(Shape.of(2, 3));
        Tensor b = Torch.zeros(Shape.of(2));
        assertThrows(ShapeException.class, () -> a.add(b));
        assertThrows(ShapeException.class, () -> a.matmul(Torch.zeros(Shape.of(2, 2))));
        assertThrows(ShapeException.class, () -> a.reshape(4));
        assertThrows(IllegalArgumentException.class, () -> a.clamp(1, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> a.get(2, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> a.set(1, 0, 3));
        assertThrows(IndexOutOfBoundsException.class, () -> Tensor.cat(
                java.util.List.of(a, Torch.zeros(Shape.of(2, 3))), 2));
        assertThrows(ShapeException.class, () ->
                Tensor.stack(java.util.List.of(a, Torch.zeros(Shape.of(2, 2))), 0));
    }

    @Test
    void integerInPlaceDivisionIsRejectedLikePyTorch() {
        Tensor ints = Torch.tensor(new long[] {3, 4}, Shape.of(2));
        Tensor divisor = Torch.tensor(new long[] {2, 2}, Shape.of(2));
        DTypeException error = assertThrows(DTypeException.class, () -> ints.div_(divisor));
        assertTrue(error.getMessage().contains("result type"), error.getMessage());
    }

    @Test
    void castSemantics() {
        Tensor values = Torch.tensor(new float[] {1.9f, -1.9f, 0f}, Shape.of(3));
        assertArrayEquals(new long[] {1, -1, 0}, values.toLong().toLongArray());
        assertArrayEquals(new boolean[] {true, true, false}, values.toBool().toBooleanArray());
        assertEquals(DType.INT32, values.to(org.jtorch.core.DType.INT32).dtype());
        assertThrows(DTypeException.class, () -> values.toLong().requiresGrad(true));
    }

    @Test
    void scalarAlphaScalesUnlikeTensors() {
        Tensor ints = Torch.tensor(new long[] {1, 2}, Shape.of(2));
        Tensor other = Torch.tensor(new long[] {2, 4}, Shape.of(2));
        assertArrayEquals(new long[] {5, 10}, ints.add(other, 2).toLongArray());
        assertArrayEquals(new long[] {-3, -6}, ints.sub(other, 2).toLongArray());
        assertFalse(ints.add(other, 2).requiresGrad());
    }
}
