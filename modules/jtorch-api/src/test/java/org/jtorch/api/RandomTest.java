package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class RandomTest {
    @Test
    void manualSeedMakesStreamsReproducible() {
        Torch.manualSeed(42);
        Tensor first = Torch.randn(Shape.of(64));
        Torch.manualSeed(42);
        Tensor second = Torch.randn(Shape.of(64));
        assertArrayEquals(first.toFloatArray(), second.toFloatArray(), 0f);

        Torch.manualSeed(42);
        Tensor uniform = Torch.rand(Shape.of(32));
        Torch.manualSeed(42);
        assertArrayEquals(uniform.toFloatArray(), Torch.rand(Shape.of(32)).toFloatArray(), 0f);

        Torch.manualSeed(43);
        assertNotEquals(first.toFloatArray()[0], Torch.randn(Shape.of(64)).toFloatArray()[0]);
    }

    @Test
    void uniformSamplesStayInTheUnitInterval() {
        Torch.manualSeed(7);
        Tensor values = Torch.rand(Shape.of(4096), TensorOptions.defaults().dtype(DType.FLOAT64));
        for (double value : values.toDoubleArray()) {
            assertTrue(value >= 0 && value < 1, "uniform sample out of range: " + value);
        }
    }

    @Test
    void randintRespectsBoundsAndDtype() {
        Torch.manualSeed(11);
        Tensor values = Torch.randint(3, 9, Shape.of(2048));
        assertEquals(DType.INT64, values.dtype());
        for (long value : values.toLongArray()) {
            assertTrue(value >= 3 && value < 9, "randint sample out of range: " + value);
        }
    }

    @Test
    void normalSamplesHaveExpectedMoments() {
        Torch.manualSeed(123);
        Tensor values = Torch.randn(Shape.of(20_000), TensorOptions.defaults().dtype(DType.FLOAT64));
        double[] data = values.toDoubleArray();
        double mean = 0;
        for (double value : data) {
            mean += value;
        }
        mean /= data.length;
        double variance = 0;
        for (double value : data) {
            variance += (value - mean) * (value - mean);
        }
        variance /= data.length - 1;
        assertTrue(Math.abs(mean) < 0.03, "normal mean drifted: " + mean);
        assertTrue(Math.abs(Math.sqrt(variance) - 1.0) < 0.03, "normal std drifted: " + Math.sqrt(variance));
    }

    @Test
    void uniformSamplesHaveExpectedMean() {
        Torch.manualSeed(321);
        Tensor values = Torch.rand(Shape.of(20_000), TensorOptions.defaults().dtype(DType.FLOAT64));
        double mean = 0;
        for (double value : values.toDoubleArray()) {
            mean += value;
        }
        mean /= values.numel();
        assertTrue(Math.abs(mean - 0.5) < 0.01, "uniform mean drifted: " + mean);
    }

    @Test
    void factoriesProduceRequestedShapesAndDtypes() {
        Torch.manualSeed(1);
        assertEquals(Shape.of(3, 4), Torch.randn(Shape.of(3, 4)).shape());
        assertEquals(DType.FLOAT64,
                Torch.rand(Shape.of(2), TensorOptions.defaults().dtype(DType.FLOAT64)).dtype());
        assertEquals(Shape.of(4, 4), Torch.eye(4).shape());
    }
}
