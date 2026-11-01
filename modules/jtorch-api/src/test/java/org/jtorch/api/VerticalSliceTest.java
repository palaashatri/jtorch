package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.jtorch.core.AutogradException;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.GradMode;
import org.jtorch.core.JTorch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class VerticalSliceTest {
    private static final double TOLERANCE = 1e-6;

    private static TensorOptions f32() {
        return TensorOptions.defaults();
    }

    @Test
    void runtimeBootstrapsBackendAndAutograd() {
        Torch.tensor(new float[] {1}, Shape.of(1));
        assertTrue(JTorch.isInitialized());
    }

    @Test
    void verticalSliceForwardAndBackward() {
        Tensor x = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
        Tensor w = Torch.tensor(new float[] {0.5f, 0.25f}, Shape.of(2, 1)).requiresGrad(true);

        Tensor loss = x.matmul(w).square().mean();
        assertEquals(3.625, loss.item(), TOLERANCE);
        loss.backward();

        assertArrayEquals(new double[] {0.5, 0.25, 1.25, 0.625}, x.grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(new double[] {8.5, 12.0}, w.grad().toDoubleArray(), TOLERANCE);
        assertEquals(Shape.of(2, 2), x.grad().shape());
    }

    @Test
    void metaExecutionMatchesCpuMetadata() {
        Tensor meta = Torch.zeros(
                Shape.of(2, 3), TensorOptions.defaults().device(Device.meta()));
        Tensor metaWeight = Torch.zeros(
                Shape.of(3, 4), TensorOptions.defaults().device(Device.meta()));
        Tensor product = meta.matmul(metaWeight);
        assertEquals(Shape.of(2, 4), product.shape());
        assertEquals(Device.meta(), product.device());
        assertEquals(DType.FLOAT32, product.dtype());

        Tensor cpu = Torch.zeros(Shape.of(2, 3));
        Tensor cpuWeight = Torch.zeros(Shape.of(3, 4));
        assertEquals(cpu.matmul(cpuWeight).shape(), product.shape());
        assertEquals(cpu.matmul(cpuWeight).dtype(), product.dtype());
    }

    @Test
    void mutationAfterSaveIsDetected() {
        Tensor x = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
        Tensor y = x.mul(x);
        Tensor loss = y.sum();
        GradMode.noGrad(() -> x.add_(Torch.ones(Shape.of(2, 2))));
        AutogradException error = assertThrows(AutogradException.class, loss::backward);
        assertTrue(error.getMessage().contains("modified by an inplace operation"), error.getMessage());
    }

    @Test
    void noGradPreventsGraphConstruction() {
        Tensor x = Torch.tensor(new float[] {1, 2}, Shape.of(2)).requiresGrad(true);
        Tensor y = Torch.noGrad(() -> x.mul(x));
        assertFalse(y.requiresGrad());
        assertTrue(y.isLeaf());
    }

    @Test
    void inPlaceOnTensorRequiringGradFailsLoudly() {
        Tensor x = Torch.tensor(new float[] {1, 2}, Shape.of(2)).requiresGrad(true);
        assertThrows(AutogradException.class, () -> x.add_(Torch.ones(Shape.of(2))));
    }

    @Test
    void broadcastGradientIsReducedToInputShape() {
        Tensor a = Torch.tensor(new float[] {1, 2, 3}, Shape.of(3)).requiresGrad(true);
        Tensor b = Torch.tensor(new float[] {10, 20}, Shape.of(2, 1)).requiresGrad(true);
        Tensor out = a.add(b);
        out.sum().backward();
        assertArrayEquals(new double[] {2, 2, 2}, a.grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(new double[] {3, 3}, b.grad().toDoubleArray(), TOLERANCE);
        assertEquals(Shape.of(2, 1), b.grad().shape());
    }

    @Test
    void dispatchKeySelectionIsVisible() {
        Tensor x = Torch.tensor(new float[] {1, 2}, Shape.of(2));
        Tensor y = Torch.zeros(Shape.of(2), TensorOptions.defaults().device(Device.meta()));
        assertThrows(org.jtorch.core.DeviceException.class, () -> x.add(y));
    }

    @Test
    void graphIsFreedAfterBackwardUnlessRetained() {
        Tensor x = Torch.tensor(new float[] {1, 2}, Shape.of(2)).requiresGrad(true);
        Tensor loss = x.mul(x).sum();
        loss.backward(/* retainGraph= */ true);
        loss.backward();
        Tensor fresh = Torch.tensor(new float[] {1, 2}, Shape.of(2)).requiresGrad(true);
        Tensor freshLoss = fresh.mul(fresh).sum();
        freshLoss.backward();
        AutogradException error = assertThrows(AutogradException.class, freshLoss::backward);
        assertTrue(error.getMessage().contains("second time"), error.getMessage());
    }
}
