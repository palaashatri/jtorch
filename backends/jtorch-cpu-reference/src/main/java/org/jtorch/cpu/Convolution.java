package org.jtorch.cpu;

import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;

/** Validation and shape rules shared by conv2d/max_pool2d forward, meta and backward helpers. */
final class Convolution {
    private Convolution() {}

    record ConvConfig(int[] stride, int[] padding, int[] dilation, int groups) {}

    record PoolConfig(int[] kernel, int[] stride, int[] padding, int[] dilation, boolean ceilMode) {}

    static ConvConfig convConfig(long[] stride, long[] padding, long[] dilation, int groups) {
        if (groups != 1) {
            throw new JTorchException("conv2d in this build supports groups=1 only, got " + groups);
        }
        return new ConvConfig(pair(stride, "stride"), pair(padding, "padding"), pair(dilation, "dilation"), groups);
    }

    static PoolConfig poolConfig(long[] kernel, long[] stride, long[] padding, long[] dilation, boolean ceilMode) {
        if (ceilMode) {
            throw new JTorchException("max_pool2d in this build does not support ceil_mode=true");
        }
        return new PoolConfig(pair(kernel, "kernel_size"), pair(stride, "stride"), pair(padding, "padding"),
                pair(dilation, "dilation"), false);
    }

    static int[] pair(long[] values, String name) {
        if (values.length == 1) {
            return new int[] {Math.toIntExact(values[0]), Math.toIntExact(values[0])};
        }
        if (values.length == 2) {
            return new int[] {Math.toIntExact(values[0]), Math.toIntExact(values[1])};
        }
        throw new ShapeException(name + " must have one or two entries, got " + values.length);
    }

    static void validateConv(Tensor input, Tensor weight, Tensor bias, ConvConfig config) {
        if (input.dim() != 4) {
            throw new ShapeException("conv2d expects a 4-D NCHW input, got " + input.shape());
        }
        if (weight.dim() != 4) {
            throw new ShapeException("conv2d expects a 4-D OIHW weight, got " + weight.shape());
        }
        if (!input.dtype().isFloatingPoint() || !weight.dtype().isFloatingPoint()) {
            throw new DTypeException("conv2d requires floating point tensors");
        }
        if (input.dtype() != weight.dtype()) {
            throw new DTypeException("conv2d input and weight dtypes differ: " + input.dtype() + " vs "
                    + weight.dtype());
        }
        if (bias != null) {
            if (bias.dim() != 1 || bias.size(0) != weight.size(0)) {
                throw new ShapeException("conv2d bias must have shape [" + weight.size(0) + "], got " + bias.shape());
            }
            if (bias.dtype() != input.dtype()) {
                throw new DTypeException("conv2d bias dtype must match input dtype");
            }
        }
        if (input.size(1) != weight.size(1)) {
            throw new ShapeException("conv2d input channels " + input.size(1) + " do not match weight channels "
                    + weight.size(1));
        }
        for (int i = 0; i < 2; i++) {
            if (config.stride()[i] <= 0 || config.dilation()[i] <= 0 || config.padding()[i] < 0) {
                throw new ShapeException("conv2d stride and dilation must be positive and padding non-negative");
            }
        }
    }

    static Shape convOutputShape(Tensor input, Tensor weight, ConvConfig config) {
        long height = outputSize(input.size(2), config.padding()[0], config.dilation()[0],
                (int) weight.size(2), config.stride()[0]);
        long width = outputSize(input.size(3), config.padding()[1], config.dilation()[1],
                (int) weight.size(3), config.stride()[1]);
        return Shape.of(input.size(0), weight.size(0), height, width);
    }

    static void validatePool(Tensor input, PoolConfig config) {
        if (input.dim() != 4) {
            throw new ShapeException("max_pool2d expects a 4-D NCHW input, got " + input.shape());
        }
        if (!input.dtype().isFloatingPoint()) {
            throw new DTypeException("max_pool2d requires a floating point tensor");
        }
        for (int i = 0; i < 2; i++) {
            if (config.kernel()[i] <= 0 || config.stride()[i] <= 0 || config.dilation()[i] <= 0
                    || config.padding()[i] < 0) {
                throw new ShapeException("max_pool2d kernel, stride and dilation must be positive");
            }
            if (config.padding()[i] > config.kernel()[i] / 2) {
                throw new ShapeException("max_pool2d padding must not exceed half the kernel size");
            }
        }
    }

    static Shape poolOutputShape(Tensor input, PoolConfig config) {
        long height = outputSize(input.size(2), config.padding()[0], config.dilation()[0],
                config.kernel()[0], config.stride()[0]);
        long width = outputSize(input.size(3), config.padding()[1], config.dilation()[1],
                config.kernel()[1], config.stride()[1]);
        return Shape.of(input.size(0), input.size(1), height, width);
    }

    static long outputSize(long input, int padding, int dilation, int kernel, int stride) {
        long effective = (long) dilation * (kernel - 1) + 1;
        long output = (input + 2L * padding - effective) / stride + 1;
        if (output <= 0) {
            throw new ShapeException("convolution/pooling produces an empty output dimension from input size "
                    + input + ", padding " + padding + ", dilation " + dilation + ", kernel " + kernel
                    + ", stride " + stride);
        }
        return output;
    }

    static DType requireFloatingSame(Tensor left, Tensor right) {
        if (left.dtype() != right.dtype() || !left.dtype().isFloatingPoint()) {
            throw new DTypeException("convolution operands must share a floating dtype");
        }
        return left.dtype();
    }
}
