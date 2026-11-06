package org.jtorch.cpu;

import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Strides;
import org.jtorch.core.Tensor;
import org.jtorch.core.value.Value;

/** Shared FFT shape/dtype/norm rules for meta and CPU kernels. */
final class Fft {
    private Fft() {}

    enum Kind {
        FFT,
        IFFT,
        RFFT,
        IRFFT
    }

    static Kind kindFor(String opName) {
        return switch (opName) {
            case "aten::fft_fft.default" -> Kind.FFT;
            case "aten::fft_ifft.default" -> Kind.IFFT;
            case "aten::fft_rfft.default" -> Kind.RFFT;
            case "aten::fft_irfft.default" -> Kind.IRFFT;
            default -> throw new IllegalStateException("Not an FFT op: " + opName);
        };
    }

    static String parseNorm(Value normArg) {
        if (normArg.isNone()) {
            return "backward";
        }
        String norm = normArg.asString();
        if (!norm.equals("backward") && !norm.equals("forward") && !norm.equals("ortho")) {
            throw new IllegalArgumentException(
                    "FFT norm must be \"backward\", \"forward\" or \"ortho\", got \"" + norm + "\"");
        }
        return norm;
    }

    static double forwardScale(String norm, long n) {
        return switch (norm) {
            case "backward" -> 1.0;
            case "forward" -> 1.0 / n;
            case "ortho" -> 1.0 / Math.sqrt(n);
            default -> throw new IllegalArgumentException("Unknown FFT norm " + norm);
        };
    }

    static double inverseScale(String norm, long n) {
        return switch (norm) {
            case "backward" -> 1.0 / n;
            case "forward" -> 1.0;
            case "ortho" -> 1.0 / Math.sqrt(n);
            default -> throw new IllegalArgumentException("Unknown FFT norm " + norm);
        };
    }

    static int resolveDim(Tensor input, int dim) {
        int normalized = input.shape().normalizeDim(dim);
        if (normalized < 0 || normalized >= input.dim()) {
            throw new IndexOutOfBoundsException(
                    "FFT dimension out of range for rank " + input.dim() + ": " + dim);
        }
        return normalized;
    }

    /** Resolves transform length N and validates dtypes. Returns N. */
    static long resolveN(Kind kind, Tensor input, int dim, Integer n) {
        long inputSize = input.size(dim);
        long resolved;
        switch (kind) {
            case FFT, IFFT -> {
                if (n == null) {
                    resolved = inputSize;
                } else {
                    if (n <= 0) {
                        throw new IllegalArgumentException("FFT n must be positive, got " + n);
                    }
                    resolved = n;
                }
            }
            case RFFT -> {
                if (n == null) {
                    resolved = inputSize;
                } else {
                    if (n <= 0) {
                        throw new IllegalArgumentException("FFT n must be positive, got " + n);
                    }
                    resolved = n;
                }
            }
            case IRFFT -> {
                if (n == null) {
                    if (inputSize == 0) {
                        throw new ShapeException("irfft input must be non-empty");
                    }
                    resolved = 2 * (inputSize - 1);
                    if (resolved <= 0) {
                        throw new IllegalArgumentException("irfft n must be positive, got " + resolved);
                    }
                } else {
                    if (n <= 0) {
                        throw new IllegalArgumentException("FFT n must be positive, got " + n);
                    }
                    resolved = n;
                }
            }
            default -> throw new IllegalStateException("kind");
        }
        if (resolved <= 0 || resolved > Integer.MAX_VALUE) {
            throw new ShapeException("FFT length is too large for the reference backend: " + resolved);
        }
        switch (kind) {
            case RFFT -> {
                if (input.dtype().isComplex()) {
                    throw new DTypeException("rfft expects a real input tensor, but got " + input.dtype());
                }
                if (input.dtype().isBoolean() || input.dtype().isIntegral() || input.dtype().isFloatingPoint()) {
                    // accepted
                } else {
                    throw new DTypeException("rfft expects a real input tensor, but got " + input.dtype());
                }
            }
            case IRFFT -> {
                if (!input.dtype().isComplex()) {
                    throw new DTypeException("irfft expects a complex input tensor, but got " + input.dtype());
                }
            }
            case FFT, IFFT -> {
                // real or complex accepted
            }
            default -> throw new IllegalStateException("kind");
        }
        return resolved;
    }

    static DType outputDtype(Kind kind, Tensor input) {
        switch (kind) {
            case FFT, IFFT, RFFT -> {
                if (input.dtype().isComplex()) {
                    return input.dtype();
                }
                return input.dtype() == DType.FLOAT64 ? DType.COMPLEX128 : DType.COMPLEX64;
            }
            case IRFFT -> {
                return input.dtype().componentDType();
            }
            default -> throw new IllegalStateException("kind");
        }
    }

    static Shape outputShape(Kind kind, Tensor input, int dim, long n) {
        long[] sizes = input.shape().toArray();
        switch (kind) {
            case FFT, IFFT -> sizes[dim] = n;
            case RFFT -> sizes[dim] = n / 2 + 1;
            case IRFFT -> sizes[dim] = n;
            default -> throw new IllegalStateException("kind");
        }
        return Shape.of(sizes);
    }

    /**
     * Strides for the FFT output. The pinned oracle allocates a fresh contiguous output for these
     * out-of-place ops and the checked-in fixtures are generated by the canonical CI runtime, so the
     * reference backend returns plain contiguous strides. {@code dim} is retained so call sites read
     * uniformly across kinds.
     */
    static Strides outputStrides(Shape shape, int dim) {
        return Strides.contiguous(shape);
    }

    static void validateForBackward(Tensor input, Kind kind) {
        if (kind == Kind.RFFT && input.dtype().isComplex()) {
            throw new DTypeException("rfft expects a real input tensor, but got " + input.dtype());
        }
        if (kind == Kind.IRFFT && !input.dtype().isComplex()) {
            throw new DTypeException("irfft expects a complex input tensor, but got " + input.dtype());
        }
    }
}
