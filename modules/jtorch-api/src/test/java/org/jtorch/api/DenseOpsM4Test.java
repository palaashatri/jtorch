package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
import org.jtorch.core.NotSupportedException;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class DenseOpsM4Test {
    @Test
    void adaptivePoolingHandlesBatchedAndChannelOnlyInputs() {
        Tensor batched = Torch.arange(0, 16).to(DType.FLOAT64).reshape(1, 1, 4, 4);
        Tensor pooled = batched.adaptiveAvgPool2d(new int[] {2, 2});
        assertEquals(Shape.of(1, 1, 2, 2), pooled.shape());
        assertArrayEquals(new double[] {2.5, 4.5, 10.5, 12.5}, pooled.toDoubleArray(), 1e-12);

        Tensor channels = Torch.arange(0, 32).to(DType.FLOAT64).reshape(2, 4, 4);
        assertEquals(Shape.of(2, 2, 2), channels.adaptiveAvgPool2d(new int[] {2, 2}).shape());
        assertEquals(Shape.of(2, 3, 3), channels.adaptiveAvgPool2d(new int[] {3, 3}).shape());
    }

    @Test
    void adaptivePoolingBackwardPartitionsGradientAcrossRegions() {
        Tensor input = Torch.ones(Shape.of(1, 1, 4, 4)).requiresGrad(true);
        input.adaptiveAvgPool2d(new int[] {2, 2}).sum().backward();
        assertArrayEquals(new double[] {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
                input.grad().toDoubleArray(), 1e-12);
        assertEquals(Shape.of(1, 1, 0, 2), input.adaptiveAvgPool2d(new int[] {0, 2}).shape());
        assertThrows(DTypeException.class,
                () -> Torch.arange(0, 4).reshape(1, 2, 2).to(DType.INT64).adaptiveAvgPool2d(new int[] {1, 1}));
    }

    @Test
    void interpolateSupportsNearestAndBilinearModes() {
        Tensor input = Torch.tensor(new double[] {0, 1, 2, 3}, Shape.of(1, 1, 2, 2)).requiresGrad(true);
        Tensor nearest = input.interpolate(new int[] {3, 3}, "nearest", false);
        assertArrayEquals(new double[] {0, 0, 1, 0, 0, 1, 2, 2, 3}, nearest.toDoubleArray(), 1e-12);
        Tensor bilinear = input.interpolate(new int[] {3, 3}, "bilinear", false);
        assertArrayEquals(new double[] {0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3}, bilinear.toDoubleArray(), 1e-12);
        bilinear.sum().backward();
        assertArrayEquals(new double[] {2.25, 2.25, 2.25, 2.25}, input.grad().toDoubleArray(), 1e-12);
        assertThrows(ShapeException.class, () -> input.interpolate(new int[] {3, 3}, "bicubic", false));
        assertThrows(ShapeException.class, () -> input.interpolate(new int[] {3, 3}, "nearest", true));
        assertThrows(ShapeException.class, () -> input.interpolate(new int[] {0, 3}, "nearest", false));
    }

    @Test
    void geluMatchesBothDocumentedForms() {
        Tensor input = Torch.tensor(new double[] {-1, 0, 1}, Shape.of(3)).requiresGrad(true);
        Tensor exact = input.gelu();
        assertArrayEquals(new double[] {-0.15865525393145707, 0, 0.8413447460685429},
                exact.toDoubleArray(), 1e-12);
        Tensor tanh = input.gelu("tanh");
        assertArrayEquals(new double[] {-0.15880800939172324, 0, 0.8411919906082768},
                tanh.toDoubleArray(), 1e-12);
        exact.sum().backward();
        assertEquals(3, input.grad().numel());
        assertThrows(ShapeException.class, () -> input.gelu("polynomial"));
        assertThrows(DTypeException.class,
                () -> Torch.tensor(new long[] {0, 1}, Shape.of(2)).gelu());
    }

    @Test
    void embeddingSupportsPaddingFrequencyScalingAndRejectsSparse() {
        Tensor weight = Torch.tensor(new double[] {0, 1, 2, 3, 4, 5}, Shape.of(3, 2))
                .requiresGrad(true);
        Tensor indices = Torch.tensor(new long[] {2, 0, 2},
                Shape.of(3), TensorOptions.defaults().dtype(DType.INT64));
        Tensor selected = weight.embedding(indices, null, true, false);
        assertEquals(Shape.of(3, 2), selected.shape());
        assertArrayEquals(new double[] {4, 5, 0, 1, 4, 5}, selected.toDoubleArray(), 0);
        selected.sum().backward();
        assertArrayEquals(new double[] {1, 1, 0, 0, 1, 1}, weight.grad().toDoubleArray(), 0);

        Tensor paddedWeight = Torch.tensor(new double[] {0, 1, 2, 3, 4, 5}, Shape.of(3, 2))
                .requiresGrad(true);
        Tensor padded = paddedWeight.embedding(indices, 2, false, false);
        assertArrayEquals(new double[] {4, 5, 0, 1, 4, 5}, padded.toDoubleArray(), 0);
        padded.sum().backward();
        assertArrayEquals(new double[] {1, 1, 0, 0, 0, 0}, paddedWeight.grad().toDoubleArray(), 0);

        assertThrows(IndexOutOfBoundsException.class,
                () -> weight.embedding(Torch.tensor(new long[] {3}, Shape.of(1),
                        TensorOptions.defaults().dtype(DType.INT64)), null, false, false));
        assertThrows(NotSupportedException.class,
                () -> weight.embedding(indices, null, false, true));
    }
}
