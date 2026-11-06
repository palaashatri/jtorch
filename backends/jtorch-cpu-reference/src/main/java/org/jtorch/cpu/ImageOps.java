package org.jtorch.cpu;

import java.util.Locale;
import org.jtorch.core.DTypeException;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;

/** Shared validation and shape rules for adaptive pooling and image resizing. */
final class ImageOps {
    private ImageOps() {}

    static int[] adaptiveOutputSize(long[] raw, long height, long width) {
        if (raw.length < 1 || raw.length > 2) {
            throw new ShapeException("adaptive_avg_pool2d output_size must have one or two entries");
        }
        long first = raw[0];
        long second = raw.length == 1 ? first : raw[1];
        if (first < 0 || second < 0) {
            throw new ShapeException("adaptive_avg_pool2d output_size entries must be non-negative");
        }
        return new int[] {Math.toIntExact(first), Math.toIntExact(second)};
    }

    static Shape adaptiveShape(Tensor input, int[] outputSize) {
        validatePoolInput(input, "adaptive_avg_pool2d");
        if (input.dim() == 3) {
            return Shape.of(input.size(0), outputSize[0], outputSize[1]);
        }
        return Shape.of(input.size(0), input.size(1), outputSize[0], outputSize[1]);
    }

    static int[] interpolateSize(long[] raw, Tensor input) {
        if (raw.length != 2) {
            throw new ShapeException("interpolate size must have two entries (height, width)");
        }
        if (raw[0] <= 0 || raw[1] <= 0) {
            throw new ShapeException("interpolate size entries must be positive");
        }
        return new int[] {Math.toIntExact(raw[0]), Math.toIntExact(raw[1])};
    }

    static Shape interpolateShape(Tensor input, int[] size) {
        validateImageInput(input);
        return Shape.of(input.size(0), input.size(1), size[0], size[1]);
    }

    static void validateMode(String mode) {
        String normalized = mode.toLowerCase(Locale.ROOT);
        if (!normalized.equals("nearest") && !normalized.equals("bilinear")) {
            throw new ShapeException("interpolate supports mode 'nearest' or 'bilinear', got '" + mode + "'");
        }
    }

    static void validateMode(String mode, boolean alignCorners) {
        validateMode(mode);
        if (alignCorners && mode.equalsIgnoreCase("nearest")) {
            throw new ShapeException(
                    "interpolate align_corners is only supported for bilinear mode, not nearest");
        }
    }

    static void validatePoolInput(Tensor input, String op) {
        if (input.dim() != 3 && input.dim() != 4) {
            throw new ShapeException(op + " expects a 3-D CHW or 4-D NCHW tensor, got " + input.shape());
        }
        if (!input.dtype().isFloatingPoint()) {
            throw new DTypeException(op + " requires a floating point tensor, got " + input.dtype());
        }
    }

    static void validateImageInput(Tensor input) {
        if (input.dim() != 4) {
            throw new ShapeException("interpolate expects a 4-D NCHW tensor, got " + input.shape());
        }
        if (!input.dtype().isFloatingPoint()) {
            throw new DTypeException("interpolate requires a floating point tensor, got " + input.dtype());
        }
    }

    static boolean tanhApproximation(String approximate) {
        String normalized = approximate.toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "none" -> false;
            case "tanh" -> true;
            default -> throw new ShapeException(
                    "gelu approximate must be 'none' or 'tanh', got '" + approximate + "'");
        };
    }

    static double sourceCoordinate(long destination, long input, long output, boolean alignCorners) {
        if (alignCorners) {
            return output <= 1 ? 0.0 : destination * (double) (input - 1) / (output - 1);
        }
        return (destination + 0.5) * ((double) input / output) - 0.5;
    }
}
