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

    record AvgPoolConfig(int[] kernel, int[] stride, int[] padding, boolean countIncludePad) {}

    record PadConfig(int pairs, int[] pads) {}

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

    static AvgPoolConfig avgPoolConfig(
            long[] kernel, long[] stride, long[] padding, boolean ceilMode, boolean countIncludePad) {
        if (ceilMode) {
            throw new JTorchException("avg_pool2d in this build does not support ceil_mode=true");
        }
        return new AvgPoolConfig(pair(kernel, "kernel_size"), pair(stride, "stride"), pair(padding, "padding"),
                countIncludePad);
    }

    static void validateAvgPool(Tensor input, AvgPoolConfig config) {
        if ((input.dim() != 3 && input.dim() != 4)
                || (!input.dtype().isFloatingPoint() && !input.dtype().isIntegral())) {
            throw new ShapeException(
                    "avg_pool2d expects a 3-D CHW or 4-D NCHW tensor with a numeric dtype, got " + input.shape()
                            + " " + input.dtype());
        }
        for (int i = 0; i < 2; i++) {
            if (config.kernel()[i] <= 0 || config.stride()[i] <= 0 || config.padding()[i] < 0) {
                throw new ShapeException("avg_pool2d kernel/stride must be positive and padding non-negative");
            }
            if (config.padding()[i] > config.kernel()[i] / 2) {
                throw new ShapeException("avg_pool2d padding must not exceed half the kernel size");
            }
        }
    }

    static Shape avgPoolOutputShape(Tensor input, AvgPoolConfig config) {
        int heightDim = input.dim() - 2;
        int widthDim = input.dim() - 1;
        long height = outputSize(input.size(heightDim), config.padding()[0], 1, config.kernel()[0],
                config.stride()[0]);
        long width = outputSize(input.size(widthDim), config.padding()[1], 1, config.kernel()[1],
                config.stride()[1]);
        if (input.dim() == 3) {
            return Shape.of(input.size(0), height, width);
        }
        return Shape.of(input.size(0), input.size(1), height, width);
    }

    static PadConfig padConfig(long[] pad, int rank) {
        if (pad.length % 2 != 0) {
            throw new ShapeException("constant_pad_nd pad must have an even number of entries");
        }
        int pairs = pad.length / 2;
        if (pairs > rank) {
            throw new ShapeException("constant_pad_nd pad length " + pad.length + " exceeds input rank " + rank);
        }
        int[] values = new int[pad.length];
        for (int i = 0; i < pad.length; i++) {
            values[i] = Math.toIntExact(pad[i]);
        }
        return new PadConfig(pairs, values);
    }

    static Shape padShape(Shape input, PadConfig config) {
        long[] shape = input.toArray();
        for (int i = 0; i < config.pairs(); i++) {
            int dim = shape.length - 1 - i;
            shape[dim] += (long) config.pads()[2 * i] + config.pads()[2 * i + 1];
            if (shape[dim] <= 0) {
                throw new ShapeException("constant_pad_nd produced a non-positive dimension " + dim);
            }
        }
        return Shape.of(shape);
    }

    static int padStart(PadConfig config, int dim, int rank) {
        int pair = rank - 1 - dim;
        if (pair < 0 || pair >= config.pairs()) {
            return 0;
        }
        return config.pads()[2 * pair];
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
        if (weight.size(2) <= 0 || weight.size(3) <= 0) {
            throw new ShapeException("conv2d kernel dimensions must be positive, got " + weight.shape());
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
        if (kernel <= 0 || stride <= 0 || dilation <= 0 || padding < 0) {
            throw new ShapeException("kernel, stride and dilation must be positive and padding non-negative");
        }
        long effective = (long) dilation * (kernel - 1) + 1;
        long output = Math.floorDiv(input + 2L * padding - effective, stride) + 1;
        if (output <= 0) {
            throw new ShapeException("convolution/pooling produces an empty output dimension from input size "
                    + input + ", padding " + padding + ", dilation " + dilation + ", kernel " + kernel
                    + ", stride " + stride);
        }
        return output;
    }

    static void validateConvBackwardInput(Tensor grad, Tensor weight, long[] inputShape, ConvConfig config) {
        if (inputShape.length != 4) {
            throw new ShapeException("conv2d input_shape must have four entries");
        }
        if (grad.dim() != 4 || weight.dim() != 4) {
            throw new ShapeException("conv2d backward expects 4-D grad and weight, got " + grad.shape() + " and "
                    + weight.shape());
        }
        if (!grad.dtype().isFloatingPoint() || grad.dtype() != weight.dtype()) {
            throw new DTypeException("conv2d backward grad and weight must share a floating dtype");
        }
        if (grad.size(1) != weight.size(0)) {
            throw new ShapeException("conv2d backward grad channels " + grad.size(1)
                    + " do not match weight output channels " + weight.size(0));
        }
        if (inputShape[1] != weight.size(1)) {
            throw new ShapeException("conv2d backward input channels " + inputShape[1]
                    + " do not match weight channels " + weight.size(1));
        }
        if (inputShape[2] <= 0 || inputShape[3] <= 0) {
            throw new ShapeException("conv2d backward input spatial dimensions must be positive");
        }
        long outputH = outputSize(inputShape[2], config.padding()[0], config.dilation()[0],
                (int) weight.size(2), config.stride()[0]);
        long outputW = outputSize(inputShape[3], config.padding()[1], config.dilation()[1],
                (int) weight.size(3), config.stride()[1]);
        if (grad.size(2) != outputH || grad.size(3) != outputW) {
            throw new ShapeException("conv2d backward grad spatial shape " + grad.shape()
                    + " does not match computed output [" + outputH + ", " + outputW + "]");
        }
    }

    static void validateConvBackwardWeight(Tensor input, Tensor grad, long[] weightShape, ConvConfig config) {
        if (weightShape.length != 4) {
            throw new ShapeException("conv2d weight_shape must have four entries");
        }
        if (input.dim() != 4 || grad.dim() != 4) {
            throw new ShapeException("conv2d backward expects 4-D input and grad");
        }
        if (!input.dtype().isFloatingPoint() || input.dtype() != grad.dtype()) {
            throw new DTypeException("conv2d backward input and grad must share a floating dtype");
        }
        if (input.size(1) != weightShape[1] || grad.size(1) != weightShape[0]) {
            throw new ShapeException("conv2d backward shapes do not match the declared weight shape");
        }
        if (weightShape[2] <= 0 || weightShape[3] <= 0) {
            throw new ShapeException("conv2d backward weight kernel dimensions must be positive");
        }
        long outputH = outputSize(input.size(2), config.padding()[0], config.dilation()[0],
                (int) weightShape[2], config.stride()[0]);
        long outputW = outputSize(input.size(3), config.padding()[1], config.dilation()[1],
                (int) weightShape[3], config.stride()[1]);
        if (grad.size(2) != outputH || grad.size(3) != outputW) {
            throw new ShapeException("conv2d backward grad spatial shape does not match the input/weight geometry");
        }
    }

    static void validateMaxPoolBackward(Tensor grad, Tensor input, PoolConfig config) {
        if (input.dim() != 4 || grad.dim() != 4) {
            throw new ShapeException("max_pool2d backward expects 4-D tensors");
        }
        validatePool(input, config);
        Shape expected = poolOutputShape(input, config);
        if (!grad.shape().equals(expected)) {
            throw new ShapeException("max_pool2d backward grad shape " + grad.shape()
                    + " does not match pooled output " + expected);
        }
        if (grad.dtype() != input.dtype() || !grad.dtype().isFloatingPoint()) {
            throw new DTypeException("max_pool2d backward grad dtype must match a floating input dtype");
        }
    }

    static DType requireFloatingSame(Tensor left, Tensor right) {
        if (left.dtype() != right.dtype() || !left.dtype().isFloatingPoint()) {
            throw new DTypeException("convolution operands must share a floating dtype");
        }
        return left.dtype();
    }
}
