package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class ConvolutionTest {
    @Test
    void conv2dForwardMatchesHandComputation() {
        Tensor input = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(1, 1, 2, 2));
        Tensor weight = Torch.tensor(new double[] {1, 0, 0, -1}, Shape.of(1, 1, 2, 2));
        Tensor bias = Torch.tensor(new double[] {1}, Shape.of(1));
        Tensor output = input.conv2d(weight, bias, new int[] {1, 1}, new int[] {0, 0});
        assertEquals(Shape.of(1, 1, 1, 1), output.shape());
        assertEquals(-2.0, output.item(), 1e-12);
    }

    @Test
    void conv2dSupportsStridePaddingAndNoBias() {
        Tensor input = Torch.ones(Shape.of(1, 2, 5, 5));
        Tensor weight = Torch.ones(Shape.of(3, 2, 3, 3));
        Tensor output = input.conv2d(weight, null, new int[] {2, 2}, new int[] {1, 1});
        assertEquals(Shape.of(1, 3, 3, 3), output.shape());
        assertEquals(8.0, output.get(0, 0, 0, 0), 1e-12);
    }

    @Test
    void conv2dBackwardPropagatesToInputWeightAndBias() {
        Tensor input = Torch.ones(Shape.of(1, 1, 2, 2)).requiresGrad(true);
        Tensor weight = Torch.ones(Shape.of(1, 1, 2, 2)).requiresGrad(true);
        Tensor bias = Torch.ones(Shape.of(1)).requiresGrad(true);
        input.conv2d(weight, bias, new int[] {1, 1}, new int[] {0, 0}).sum().backward();
        assertArrayEquals(new double[] {1, 1, 1, 1}, input.grad().toDoubleArray(), 1e-12);
        assertArrayEquals(new double[] {1, 1, 1, 1}, weight.grad().toDoubleArray(), 1e-12);
        assertArrayEquals(new double[] {1}, bias.grad().toDoubleArray(), 1e-12);
    }

    @Test
    void conv2dRejectsUnsupportedOrInvalidConfigurations() {
        Tensor input = Torch.ones(Shape.of(1, 2, 4, 4));
        Tensor weight = Torch.ones(Shape.of(3, 2, 3, 3));
        assertThrows(ShapeException.class,
                () -> input.reshape(2, 4, 4).conv2d(weight, null, new int[] {1, 1}, new int[] {0, 0}));
        assertThrows(ShapeException.class,
                () -> input.conv2d(Torch.ones(Shape.of(3, 1, 3, 3)), null, new int[] {1, 1}, new int[] {0, 0}));
        assertThrows(org.jtorch.core.JTorchException.class,
                () -> input.conv2d(weight, null, new int[] {1, 1}, new int[] {0, 0}, new int[] {1, 1}, 2));
        Tensor bools = Torch.ones(Shape.of(1, 2, 4, 4), TensorOptions.defaults().dtype(DType.BOOL));
        assertThrows(DTypeException.class,
                () -> bools.conv2d(weight, null, new int[] {1, 1}, new int[] {0, 0}));
    }

    @Test
    void maxPoolSelectsTheMaximumAndRoutesGradientBack() {
        Tensor input = Torch.tensor(new double[] {1, 3, 2, 4}, Shape.of(1, 1, 2, 2)).requiresGrad(true);
        Tensor pooled = input.maxPool2d(2, 2, 0);
        assertEquals(Shape.of(1, 1, 1, 1), pooled.shape());
        assertEquals(4.0, pooled.item(), 0);
        pooled.backward();
        assertArrayEquals(new double[] {0, 0, 0, 1}, input.grad().toDoubleArray(), 0);
    }

    @Test
    void maxPoolValidatesConfiguration() {
        Tensor input = Torch.ones(Shape.of(1, 1, 4, 4));
        assertThrows(org.jtorch.core.JTorchException.class, () -> input.maxPool2d(
                new int[] {2, 2}, new int[] {2, 2}, new int[] {0, 0}, new int[] {1, 1}, true));
        assertThrows(ShapeException.class, () -> input.maxPool2d(
                new int[] {2, 2}, new int[] {2, 2}, new int[] {2, 2}, new int[] {1, 1}, false));
        assertThrows(DTypeException.class,
                () -> Torch.ones(Shape.of(1, 1, 4, 4), TensorOptions.defaults().dtype(DType.INT64))
                        .maxPool2d(2, 2, 0));
    }

    @Test
    void metaConvAndPoolAgreeWithCpuMetadata() {
        Tensor cpuInput = Torch.ones(Shape.of(1, 2, 5, 5));
        Tensor cpuWeight = Torch.ones(Shape.of(4, 2, 3, 3));
        Tensor metaInput = Torch.ones(Shape.of(1, 2, 5, 5), TensorOptions.defaults()
                .device(org.jtorch.core.Device.meta()));
        Tensor metaWeight = Torch.ones(Shape.of(4, 2, 3, 3), TensorOptions.defaults()
                .device(org.jtorch.core.Device.meta()));
        Tensor cpu = cpuInput.conv2d(cpuWeight, null, new int[] {2, 2}, new int[] {1, 1});
        Tensor meta = metaInput.conv2d(metaWeight, null, new int[] {2, 2}, new int[] {1, 1});
        assertEquals(cpu.shape(), meta.shape());
        assertEquals(cpu.strides(), meta.strides());
        assertEquals(cpuInput.maxPool2d(2, 2, 0).shape(), metaInput.maxPool2d(2, 2, 0).shape());
    }
}
