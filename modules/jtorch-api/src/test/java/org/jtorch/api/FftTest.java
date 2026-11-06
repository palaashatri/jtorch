package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class FftTest {
    private static Tensor real(double[] values, long[] shape) {
        return Torch.tensor(values, Shape.of(shape));
    }

    private static Tensor complex(double[] real, double[] imag) {
        Tensor tensor = Torch.empty(Shape.of(real.length), TensorOptions.defaults().dtype(DType.COMPLEX128));
        for (int i = 0; i < real.length; i++) {
            tensor.setComplex(real[i], imag[i], i);
        }
        return tensor;
    }

    private static void assertComplex(Tensor actual, double[] real, double[] imag) {
        assertEquals(DType.COMPLEX128, actual.dtype());
        assertEquals(real.length, actual.numel());
        for (int i = 0; i < real.length; i++) {
            assertEquals(real[i], actual.getComplexReal(i), 1e-12, "real " + i);
            assertEquals(imag[i], actual.getComplexImaginary(i), 1e-12, "imag " + i);
        }
    }

    @Test
    void fftAndRfftMatchOracle() {
        Tensor output = real(new double[] {0, 1, 2, 3}, new long[] {4}).fft(null, -1, null);
        assertComplex(output, new double[] {6, -2, -2, -2}, new double[] {0, 2, 0, -2});
        assertEquals(Shape.of(4), output.shape());
        Tensor truncated = real(new double[] {0, 1, 2, 3}, new long[] {4}).rfft(null, -1, null);
        assertComplex(truncated, new double[] {6, -2, -2}, new double[] {0, 2, 0});
        assertEquals(Shape.of(3), truncated.shape());
    }

    @Test
    void fftReturnsContiguousOutput() {
        Tensor output = Torch.tensor(new double[] {1, 2, 3, 4, 5, 6}, Shape.of(2, 3)).fft(null, 0, "ortho");
        assertEquals(Shape.of(2, 3), output.shape());
        assertArrayEquals(new long[] {3, 1}, output.strides().toArray());
        Tensor last = Torch.tensor(new double[] {1, 2, 3, 4, 5, 6}, Shape.of(2, 3)).fft(null, -1, null);
        assertArrayEquals(new long[] {3, 1}, last.strides().toArray());
    }

    @Test
    void complexFftAndIfftRoundTrip() {
        Tensor input = complex(new double[] {1, 2, 0.5, -1}, new double[] {0.5, -1, 0.25, 2});
        Tensor transformed = input.fft(null, -1, null);
        Tensor roundTrip = transformed.ifft(null, -1, null);
        assertEquals(DType.COMPLEX128, roundTrip.dtype());
        for (int i = 0; i < 4; i++) {
            assertEquals(input.getComplexReal(i), roundTrip.getComplexReal(i), 1e-12);
            assertEquals(input.getComplexImaginary(i), roundTrip.getComplexImaginary(i), 1e-12);
        }
    }

    @Test
    void rfftBackwardMatchesOracle() {
        Tensor input = Torch.tensor(new double[] {0, 1, 2, 3}, Shape.of(4)).requiresGrad(true);
        Tensor output = input.rfft(null, -1, null);
        output.to(DType.FLOAT64).sum().backward();
        // d sum Re(rfft(x)) / dx = [3, 0, 1, 0] for the default (backward) norm.
        assertArrayEquals(new double[] {3, 0, 1, 0}, input.grad().toDoubleArray(), 1e-12);

        Tensor seeded = Torch.tensor(new double[] {0, 1, 2, 3}, Shape.of(4)).requiresGrad(true);
        Tensor seededOutput = seeded.rfft(null, -1, null);
        Tensor grad = complex(new double[] {1, 1, 1}, new double[] {0, 1, 0});
        seededOutput.backward(grad);
        // Matches PyTorch real+imag sum gradient [3, -1, 1, 1].
        assertArrayEquals(new double[] {3, -1, 1, 1}, seeded.grad().toDoubleArray(), 1e-12);
    }

    @Test
    void fftIfIrBackwardSeededGradients() {
        Tensor input = complex(new double[] {1, 2, 0.5, -1}, new double[] {0.5, -1, 0.25, 2})
                .requiresGrad(true);
        Tensor output = input.fft(null, -1, null);
        Tensor ones = complex(new double[] {1, 1, 1, 1}, new double[] {2, 2, 2, 2});
        output.backward(ones);
        // PyTorch: grad = [4+8j, 0, 0, 0] for this seed.
        assertEquals(4.0, input.grad().getComplexReal(0), 1e-12);
        assertEquals(8.0, input.grad().getComplexImaginary(0), 1e-12);
        for (int i = 1; i < 4; i++) {
            assertEquals(0.0, input.grad().getComplexReal(i), 1e-12);
            assertEquals(0.0, input.grad().getComplexImaginary(i), 1e-12);
        }

        Tensor rInput = Torch.tensor(new double[] {0, 1, 2, 3}, Shape.of(4)).requiresGrad(true);
        Tensor padded = rInput.fft(5, -1, null);
        assertEquals(Shape.of(5), padded.shape());
        Tensor paddedGrad = complex(new double[] {1, 1, 1, 1, 1}, new double[] {1, 1, 1, 1, 1});
        padded.backward(paddedGrad);
        // DC component: sum of (1+1j) over 5 bins, real input keeps the real part [5, ~0, ~0, ~0].
        assertEquals(5.0, rInput.grad().get(0), 1e-12);
        assertEquals(0.0, rInput.grad().get(1), 1e-12);

        Tensor cInput = complex(new double[] {1, -2, -2}, new double[] {0, 2, 0}).requiresGrad(true);
        Tensor realOut = cInput.irfft(4, -1, null);
        Tensor weighted = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(4));
        realOut.mul(weighted).sum().backward();
        // PyTorch grad: [2.5+0j, -1+1j, -0.5+0j]
        assertEquals(2.5, cInput.grad().getComplexReal(0), 1e-12);
        assertEquals(0.0, cInput.grad().getComplexImaginary(0), 1e-12);
        assertEquals(-1.0, cInput.grad().getComplexReal(1), 1e-12);
        assertEquals(1.0, cInput.grad().getComplexImaginary(1), 1e-12);
        assertEquals(-0.5, cInput.grad().getComplexReal(2), 1e-12);
    }

    @Test
    void irfftInvertsRfft() {
        Tensor input = Torch.tensor(new double[] {0, 1, 2, 3}, Shape.of(4));
        Tensor roundTrip = input.rfft(null, -1, null).irfft(null, -1, null);
        assertEquals(DType.FLOAT64, roundTrip.dtype());
        assertArrayEquals(new double[] {0, 1, 2, 3}, roundTrip.toDoubleArray(), 1e-12);
    }

    @Test
    void metaExecutionMatchesCpuMetadata() {
        Tensor cpu = real(new double[] {0, 1, 2, 3}, new long[] {4}).fft(null, -1, null);
        Tensor meta = Torch.zeros(Shape.of(4), TensorOptions.defaults()
                        .dtype(DType.FLOAT64).device(org.jtorch.core.Device.meta()))
                .fft(null, -1, null);
        assertEquals(cpu.shape(), meta.shape());
        assertEquals(cpu.dtype(), meta.dtype());
        assertEquals(cpu.strides(), meta.strides());
    }

    @Test
    void fftRejectsBadInputs() {
        Tensor real = real(new double[] {1, 2}, new long[] {2});
        assertThrows(IllegalArgumentException.class, () -> real.fft(0, -1, null));
        assertThrows(IllegalArgumentException.class, () -> real.fft(null, -1, "banana"));
        Tensor complex = complex(new double[] {1, 2}, new double[] {0, 0});
        assertThrows(org.jtorch.core.DTypeException.class, () -> complex.rfft(null, -1, null));
        assertThrows(org.jtorch.core.DTypeException.class, () -> real.irfft(null, -1, null));
    }
}
