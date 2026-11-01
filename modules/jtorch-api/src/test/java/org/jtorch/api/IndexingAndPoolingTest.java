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

class IndexingAndPoolingTest {
    @Test
    void flipReversesRequestedDimensions() {
        Tensor input = Torch.tensor(new double[] {1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
        assertArrayEquals(new double[] {6, 5, 4, 3, 2, 1}, input.flip(0, 1).toDoubleArray(), 0);
        assertArrayEquals(new double[] {3, 2, 1, 6, 5, 4}, input.flip(1).toDoubleArray(), 0);
        assertThrows(ShapeException.class, () -> input.flip(0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> input.flip(2));
    }

    @Test
    void constantPaddingAddsTheRequestedBorders() {
        Tensor input = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(2, 2));
        Tensor padded = input.constantPadNd(new int[] {1, 1, 0, 0}, 5.0);
        assertEquals(Shape.of(2, 4), padded.shape());
        assertArrayEquals(new double[] {5, 1, 2, 5, 5, 3, 4, 5}, padded.toDoubleArray(), 0);
        Tensor both = input.constantPadNd(new int[] {0, 1, 1, 0}, -1.0);
        assertEquals(Shape.of(3, 3), both.shape());
        assertArrayEquals(new double[] {-1, -1, -1, 1, 2, -1, 3, 4, -1}, both.toDoubleArray(), 0);
        assertThrows(ShapeException.class, () -> input.constantPadNd(new int[] {-1, 0, 0, 0}, 0));
    }

    @Test
    void gatherSelectsAlongOneDimensionAndBackpropagates() {
        Tensor input = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
        Tensor index = Torch.tensor(new long[] {1, 0, 0, 1}, Shape.of(2, 2),
                TensorOptions.defaults().dtype(DType.INT64));
        Tensor gathered = input.gather(1, index);
        assertArrayEquals(new double[] {2, 1, 3, 4}, gathered.toDoubleArray(), 0);
        gathered.sum().backward();
        assertArrayEquals(new double[] {1, 1, 1, 1}, input.grad().toDoubleArray(), 0);
        Tensor wrongDtype = Torch.tensor(new float[] {1, 0, 0, 1}, Shape.of(2, 2));
        assertThrows(DTypeException.class, () -> input.gather(1, wrongDtype));
        Tensor outOfRange = Torch.tensor(new long[] {0, 2, 0, 1}, Shape.of(2, 2),
                TensorOptions.defaults().dtype(DType.INT64));
        assertThrows(IndexOutOfBoundsException.class, () -> input.gather(1, outOfRange));
    }

    @Test
    void indexSelectDuplicatesRowsAndAccumulatesGradient() {
        Tensor input = Torch.tensor(new double[] {1, 2, 3, 4, 5, 6}, Shape.of(3, 2)).requiresGrad(true);
        Tensor index = Torch.tensor(new long[] {2, 0, 2}, Shape.of(3),
                TensorOptions.defaults().dtype(DType.INT64));
        Tensor selected = input.indexSelect(0, index);
        assertArrayEquals(new double[] {5, 6, 1, 2, 5, 6}, selected.toDoubleArray(), 0);
        selected.sum().backward();
        assertEquals(Shape.of(3, 2), input.grad().shape());
        assertArrayEquals(new double[] {1, 1, 0, 0, 2, 2}, input.grad().toDoubleArray(), 0);
    }

    @Test
    void sliceReturnsAStridedViewAndBackpropagates() {
        Tensor input = Torch.tensor(new double[] {0, 1, 2, 3, 4, 5, 6, 7}, Shape.of(2, 4))
                .requiresGrad(true);
        Tensor sliced = input.slice(1, 1, 4, 2);
        assertEquals(Shape.of(2, 2), sliced.shape());
        assertArrayEquals(new double[] {1, 3, 5, 7}, sliced.toDoubleArray(), 0);
        assertArrayEquals(new long[] {4, 2}, sliced.strides().toArray());
        sliced.sum().backward();
        assertArrayEquals(new double[] {0, 1, 0, 1, 0, 1, 0, 1}, input.grad().toDoubleArray(), 0);
        assertEquals(Shape.of(2, 0), input.slice(1, 3, 1, 1).shape());
        assertThrows(ShapeException.class, () -> input.slice(1, 0, 4, 0));
    }

    @Test
    void scatterWritesSelectedPositionsAndBackpropagates() {
        Tensor self = Torch.tensor(new double[] {1, 2, 3, 4, 5, 6}, Shape.of(2, 3)).requiresGrad(true);
        Tensor index = Torch.tensor(new long[] {0, 2, 1, 0}, Shape.of(2, 2),
                TensorOptions.defaults().dtype(DType.INT64));
        Tensor src = Torch.tensor(new double[] {10, 11, 12, 13}, Shape.of(2, 2)).requiresGrad(true);
        Tensor scattered = self.scatter(1, index, src);
        assertArrayEquals(new double[] {10, 2, 11, 13, 12, 6}, scattered.toDoubleArray(), 0);
        scattered.sum().backward();
        assertArrayEquals(new double[] {0, 1, 0, 0, 0, 1}, self.grad().toDoubleArray(), 0);
        assertArrayEquals(new double[] {1, 1, 1, 1}, src.grad().toDoubleArray(), 0);
        Tensor badIndex = Torch.tensor(new long[] {0, 3, 0, 0}, Shape.of(2, 2),
                TensorOptions.defaults().dtype(DType.INT64));
        assertThrows(IndexOutOfBoundsException.class, () -> self.scatter(1, badIndex, src));
    }

    @Test
    void scatterUsesSrcCoordinatesWhenSrcIsLargerThanIndex() {
        Tensor self = Torch.zeros(Shape.of(2, 3)).requiresGrad(true);
        Tensor index = Torch.tensor(new long[] {0, 2, 1, 0}, Shape.of(2, 2),
                TensorOptions.defaults().dtype(DType.INT64));
        Tensor src = Torch.tensor(new float[] {10, 11, 12, 13, 14, 15}, Shape.of(2, 3))
                .requiresGrad(true);
        Tensor scattered = self.scatter(1, index, src);
        assertArrayEquals(new double[] {10, 0, 11, 14, 13, 0}, scattered.toDoubleArray(), 0);
        scattered.sum().backward();
        assertArrayEquals(new double[] {1, 1, 0, 1, 1, 0}, src.grad().toDoubleArray(), 0);
        assertArrayEquals(new double[] {0, 1, 0, 0, 0, 1}, self.grad().toDoubleArray(), 0);
    }

    @Test
    void averagePoolingUsesTheDocumentedDivisor() {
        Tensor input = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(1, 1, 2, 2));
        Tensor pooled = input.avgPool2d(new int[] {2, 2}, new int[] {2, 2}, new int[] {0, 0}, false, true);
        assertEquals(2.5, pooled.item(), 1e-12);
        Tensor paddedInclude = input.avgPool2d(
                new int[] {2, 2}, new int[] {1, 1}, new int[] {1, 1}, false, true);
        Tensor paddedExclude = input.avgPool2d(
                new int[] {2, 2}, new int[] {1, 1}, new int[] {1, 1}, false, false);
        assertEquals(Shape.of(1, 1, 3, 3), paddedInclude.shape());
        assertEquals(0.25, paddedInclude.get(0, 0, 0, 0), 1e-12);
        assertEquals(1.0, paddedExclude.get(0, 0, 0, 0), 1e-12);
        assertThrows(org.jtorch.core.JTorchException.class, () -> input.avgPool2d(
                new int[] {2, 2}, new int[] {2, 2}, new int[] {0, 0}, true, true));
    }
}
