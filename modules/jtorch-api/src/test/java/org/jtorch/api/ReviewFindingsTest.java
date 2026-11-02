package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
import org.jtorch.core.Device;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

/** Regression coverage for PR review findings. */
class ReviewFindingsTest {
    @Test
    void mixedDtypeArithmeticRoundsOperandsToThePromotedDtype() {
        Tensor integers = Torch.tensor(new long[] {16777217}, Shape.of(1));
        Tensor floats = Torch.tensor(new float[] {-16777216f}, Shape.of(1));
        Tensor sum = integers.add(floats);
        assertEquals(DType.FLOAT32, sum.dtype());
        assertEquals(0.0, sum.item(), 0);
        assertEquals(false, integers.eq(floats).itemBoolean());
    }

    @Test
    void integralInPlaceOperationsRejectFloatingAlpha() {
        Tensor left = Torch.tensor(new long[] {1}, Shape.of(1));
        Tensor right = Torch.tensor(new long[] {1}, Shape.of(1));
        assertThrows(DTypeException.class, () -> Torch.noGrad(() -> left.add_(right, 0.5)));
        assertThrows(DTypeException.class, () -> Torch.noGrad(() -> left.sub_(right, 0.5)));
    }

    @Test
    void metaEmptyReductionsFailLikeCpu() {
        Tensor emptyMeta = Torch.zeros(Shape.of(0),
                TensorOptions.defaults().device(Device.meta()));
        assertThrows(ShapeException.class, emptyMeta::argmax);
        assertThrows(ShapeException.class, emptyMeta::max);
        assertThrows(ShapeException.class, emptyMeta::min);
        Tensor matrixMeta = Torch.zeros(Shape.of(0, 3),
                TensorOptions.defaults().device(Device.meta()));
        assertThrows(ShapeException.class, () -> matrixMeta.argmax(0, false));
    }

    @Test
    void itemRejectsUnrepresentableLongMaxValue() {
        Tensor maximum = Torch.tensor(new long[] {Long.MAX_VALUE}, Shape.of(1));
        assertThrows(JTorchException.class, maximum::item);
        assertEquals(Long.MAX_VALUE, maximum.itemLong());
        Tensor exact = Torch.tensor(new long[] {42}, Shape.of(1));
        assertEquals(42.0, exact.item(), 0);
    }

    @Test
    void copyIntoEmptyTargetStillValidatesBroadcastability() {
        Tensor target = Torch.zeros(Shape.of(0));
        Tensor source = Torch.ones(Shape.of(2));
        assertThrows(ShapeException.class, () -> Torch.noGrad(() -> target.copy_(source)));
    }

    @Test
    void comparisonUsesPromotedOperands() {
        Tensor integers = Torch.tensor(new long[] {16777217, 1}, Shape.of(2));
        Tensor floats = Torch.tensor(new float[] {16777216f, 2f}, Shape.of(2));
        assertArrayEquals(new boolean[] {true, false}, integers.eq(floats).toBooleanArray());
    }
}
