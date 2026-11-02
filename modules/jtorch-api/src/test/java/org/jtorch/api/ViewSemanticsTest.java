package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class ViewSemanticsTest {
    private static Tensor matrix() {
        return Torch.tensor(new float[] {1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
    }

    @Test
    void reshapeSharesStorageAndVersion() {
        Tensor base = matrix();
        Tensor reshaped = base.reshape(3, 2);
        assertTrue(reshaped.sharesStorageWith(base));
        assertTrue(reshaped.isView());
        assertSame(base.versionCounter(), reshaped.versionCounter());
        reshaped.set(99, 0, 0);
        assertArrayEquals(new double[] {99, 2, 3, 4, 5, 6}, base.toDoubleArray(), 0);
    }

    @Test
    void transposePermuteAndNarrowProduceExpectedGeometry() {
        Tensor base = matrix();
        Tensor transposed = base.transpose(0, 1);
        assertArrayEquals(new long[] {3, 2}, transposed.shape().toArray());
        assertArrayEquals(new long[] {1, 3}, transposed.strides().toArray());
        assertArrayEquals(new double[] {1, 4, 2, 5, 3, 6}, transposed.toDoubleArray(), 0);

        Tensor permuted = base.permute(1, 0);
        assertArrayEquals(transposed.strides().toArray(), permuted.strides().toArray());

        Tensor narrowed = base.narrow(1, 1, 2);
        assertArrayEquals(new long[] {2, 2}, narrowed.shape().toArray());
        assertArrayEquals(new long[] {3, 1}, narrowed.strides().toArray());
        assertArrayEquals(new double[] {2, 3, 5, 6}, narrowed.toDoubleArray(), 0);
        assertTrue(narrowed.storageOffset() > 0);

        Tensor selected = base.select(0, 1);
        assertArrayEquals(new long[] {3}, selected.shape().toArray());
        assertArrayEquals(new double[] {4, 5, 6}, selected.toDoubleArray(), 0);
    }

    @Test
    void squeezeUnsqueezeFlattenAndT() {
        Tensor base = Torch.tensor(new float[] {1, 2}, Shape.of(1, 2, 1));
        assertArrayEquals(new long[] {2}, base.squeeze().shape().toArray());
        assertArrayEquals(new long[] {2, 1}, base.squeeze(0).shape().toArray());
        assertArrayEquals(new long[] {1, 2, 1, 1}, base.unsqueeze(3).shape().toArray());
        assertArrayEquals(new long[] {6}, matrix().flatten().shape().toArray());
        assertArrayEquals(new long[] {2, 3}, matrix().t().t().shape().toArray());
        assertArrayEquals(new long[] {1, 2, 1}, base.unsqueeze(0).squeeze(0).shape().toArray());
    }

    @Test
    void expandUsesZeroStridesAndSharesData() {
        Tensor base = Torch.tensor(new float[] {1, 2}, Shape.of(2, 1));
        Tensor expanded = base.expand(2, 3);
        assertArrayEquals(new long[] {2, 3}, expanded.shape().toArray());
        assertArrayEquals(new long[] {1, 0}, expanded.strides().toArray());
        assertArrayEquals(new double[] {1, 1, 1, 2, 2, 2}, expanded.toDoubleArray(), 0);
        Torch.noGrad(() -> base.set(7, 0, 0));
        assertArrayEquals(new double[] {7, 7, 7, 2, 2, 2}, expanded.toDoubleArray(), 0);
    }

    @Test
    void viewRejectsIncompatibleShapesButReshapeCopies() {
        Tensor transposed = matrix().t();
        assertFalse(transposed.isContiguous());
        assertThrows(ShapeException.class, () -> transposed.view(6));
        Tensor reshaped = transposed.reshape(6);
        assertArrayEquals(new double[] {1, 4, 2, 5, 3, 6}, reshaped.toDoubleArray(), 0);
        assertFalse(reshaped.sharesStorageWith(transposed));
    }

    @Test
    void contiguousIsIdentityOrCopy() {
        Tensor base = matrix();
        assertSame(base, base.contiguous());
        Tensor transposed = base.t();
        Tensor contiguous = transposed.contiguous();
        assertFalse(contiguous.sharesStorageWith(transposed));
        assertTrue(contiguous.isContiguous());
        assertArrayEquals(new double[] {1, 4, 2, 5, 3, 6}, contiguous.toDoubleArray(), 0);
    }

    @Test
    void cloneDetachesStorageAndDetachKeepsStorage() {
        Tensor base = matrix().requiresGrad(true);
        Tensor clone = base.clone();
        assertFalse(clone.sharesStorageWith(base));
        assertTrue(clone.requiresGrad());
        Tensor detached = base.detach();
        assertTrue(detached.sharesStorageWith(base));
        assertFalse(detached.requiresGrad());
        assertTrue(detached.isLeaf());
    }

    @Test
    void versionCountersAreSharedAcrossViewsAndBumpedByInPlaceOps() {
        Tensor base = matrix();
        Tensor view = base.transpose(0, 1);
        long before = base.versionCounter().current();
        Torch.noGrad(() -> view.fill_(0));
        assertTrue(base.versionCounter().current() > before);
        assertArrayEquals(new double[] {0, 0, 0, 0, 0, 0}, base.toDoubleArray(), 0);
    }

    @Test
    void invalidViewArgumentsAreRejected() {
        Tensor base = matrix();
        assertThrows(IndexOutOfBoundsException.class, () -> base.narrow(0, 1, 5));
        assertThrows(IndexOutOfBoundsException.class, () -> base.select(0, 5));
        assertThrows(ShapeException.class, () -> base.expand(2, 4));
        assertThrows(IndexOutOfBoundsException.class, () -> base.transpose(0, 5));
        assertThrows(IndexOutOfBoundsException.class, () -> base.unsqueeze(4));
        assertThrows(ShapeException.class, () -> Torch.zeros(Shape.of(2, 2)).reshape(5));
    }

    @Test
    void copyBroadcastsIntoInPlaceTarget() {
        Tensor target = Torch.zeros(Shape.of(2, 3));
        Tensor source = Torch.tensor(new float[] {1, 2, 3}, Shape.of(1, 3));
        Torch.noGrad(() -> target.copy_(source));
        assertArrayEquals(new double[] {1, 2, 3, 1, 2, 3}, target.toDoubleArray(), 0);
    }

    @Test
    void toSameDtypeReturnsSelfAndCastPreservesStrides() {
        Tensor base = matrix();
        assertSame(base, base.to(org.jtorch.core.DType.FLOAT32));
        Tensor doubles = base.to(org.jtorch.core.DType.FLOAT64);
        assertFalse(doubles.sharesStorageWith(base));
        assertArrayEquals(base.strides().toArray(), doubles.strides().toArray());
        assertArrayEquals(new double[] {1, 2, 3, 4, 5, 6}, doubles.toDoubleArray(), 0);
    }

    @Test
    void nonContiguousAccessorsRespectStridesAndOffsets() {
        Tensor base = Torch.tensor(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, Shape.of(2, 4));
        Tensor step = base.narrow(0, 0, 1).select(0, 0);
        assertArrayEquals(new double[] {1, 2, 3, 4}, step.toDoubleArray(), 0);
        Tensor transposed = base.t();
        assertArrayEquals(new double[] {1, 2, 3, 4}, transposed.narrow(1, 0, 1).squeeze(1).toDoubleArray(), 0);
    }
}
