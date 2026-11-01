package org.jtorch.cpu;

import org.jtorch.core.OpRegistry;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Tensor;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.dispatch.DispatchKey;
import org.jtorch.core.dispatch.DispatchKeySet;
import org.jtorch.core.value.Value;

/**
 * Composite implementations that decompose into primitive operators. Gradients and meta shapes follow from the
 * primitives, the same way PyTorch's CompositeImplicitAutograd kernels do.
 */
final class CompositeKernels {
    private CompositeKernels() {}

    static void register(Dispatcher dispatcher) {
        register(dispatcher, OpRegistry.LINEAR, CompositeKernels::linear);
        register(dispatcher, OpRegistry.SOFTMAX, CompositeKernels::softmax);
        register(dispatcher, OpRegistry.LOG_SOFTMAX, CompositeKernels::logSoftmax);
        register(dispatcher, OpRegistry.MSE_LOSS, CompositeKernels::mseLoss);
        register(dispatcher, OpRegistry.BATCH_NORM, CompositeKernels::batchNorm);
        register(dispatcher, OpRegistry.LAYER_NORM, CompositeKernels::layerNorm);
        register(dispatcher, OpRegistry.GROUP_NORM, CompositeKernels::groupNorm);
    }

    private static void register(Dispatcher dispatcher, org.jtorch.core.Op op,
            org.jtorch.core.dispatch.Kernel kernel) {
        // Composite operators decompose for every backend, including meta. Registering the same decomposition under
        // CPU/META keeps the "every operator has meta and CPU behaviour" invariant observable to the dispatcher.
        dispatcher.register(op, DispatchKey.COMPOSITE, kernel);
        dispatcher.register(op, DispatchKey.CPU, kernel);
        dispatcher.register(op, DispatchKey.META, kernel);
    }

    private static Value[] linear(org.jtorch.core.Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor weight = args[1].asTensor();
        Tensor bias = args[2].asOptionalTensor();
        if (weight.dim() != 2) {
            throw new ShapeException("linear weight must be 2-D, got " + weight.shape());
        }
        if (input.dim() == 0) {
            throw new ShapeException("linear input must have at least one dimension");
        }
        if (input.size(input.dim() - 1) != weight.size(1)) {
            throw new ShapeException("linear input " + input.shape() + " does not match weight " + weight.shape());
        }
        Tensor output = input.matmul(weight.t());
        if (bias != null) {
            output = output.add(bias);
        }
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] softmax(org.jtorch.core.Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        ShapeRules.requireFloating(input, "softmax");
        if (input.dim() == 0) {
            int requested = args[1].asInt();
            if (requested != 0 && requested != -1) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            Tensor exponentiated = input.exp();
            return new Value[] {Value.tensor(exponentiated.div(exponentiated))};
        }
        int dim = input.shape().normalizeDim(args[1].asInt());
        if (input.size(dim) == 0) {
            return new Value[] {Value.tensor(input)};
        }
        Tensor max = input.maxWithIndices(dim, true).values();
        Tensor shifted = input.sub(max);
        Tensor exponentiated = shifted.exp();
        Tensor denominator = exponentiated.sum(new int[] {dim}, true);
        return new Value[] {Value.tensor(exponentiated.div(denominator))};
    }

    private static Value[] logSoftmax(org.jtorch.core.Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        ShapeRules.requireFloating(input, "log_softmax");
        if (input.dim() == 0) {
            int requested = args[1].asInt();
            if (requested != 0 && requested != -1) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            return new Value[] {Value.tensor(input.sub(input))};
        }
        int dim = input.shape().normalizeDim(args[1].asInt());
        if (input.size(dim) == 0) {
            return new Value[] {Value.tensor(input)};
        }
        Tensor max = input.maxWithIndices(dim, true).values();
        Tensor shifted = input.sub(max);
        Tensor denominator = shifted.exp().sum(new int[] {dim}, true);
        return new Value[] {Value.tensor(shifted.sub(denominator.log()))};
    }

    private static Value[] batchNorm(org.jtorch.core.Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor weight = args[1].asOptionalTensor();
        Tensor bias = args[2].asOptionalTensor();
        Tensor runningMean = args[3].asOptionalTensor();
        Tensor runningVar = args[4].asOptionalTensor();
        boolean training = args[5].asBoolean();
        double momentum = args[6].asScalar().toDouble();
        double eps = args[7].asScalar().toDouble();
        if (input.dim() < 2) {
            throw new ShapeException("batch_norm expects input with at least two dimensions, got " + input.shape());
        }
        if (!input.dtype().isFloatingPoint()) {
            throw new org.jtorch.core.DTypeException("batch_norm requires a floating point input");
        }
        if ((runningMean == null || runningVar == null) && !training) {
            throw new ShapeException("batch_norm inference requires running_mean and running_var");
        }
        if ((runningMean != null && runningMean.requiresGrad())
                || (runningVar != null && runningVar.requiresGrad())) {
            throw new org.jtorch.core.AutogradException(
                    "batch_norm is not differentiable with respect to running_mean/running_var");
        }
        int channels = (int) input.size(1);
        long[] dimensions = new long[input.dim()];
        dimensions[1] = channels;
        for (int i = 0; i < dimensions.length; i++) {
            if (i != 1) {
                dimensions[i] = 1;
            }
        }
        Shape statShape = Shape.of(dimensions);
        for (Tensor tensor : new Tensor[] {runningMean, runningVar, weight, bias}) {
            if (tensor != null && (tensor.dim() != 1 || tensor.size(0) != channels)) {
                throw new ShapeException("batch_norm parameter/statistic must have shape [" + channels + "], got "
                        + tensor.shape());
            }
            if (tensor != null && tensor.dtype() != input.dtype()) {
                throw new org.jtorch.core.DTypeException("batch_norm parameters must match the input dtype");
            }
        }
        Tensor mean;
        Tensor variance;
        if (training) {
            int[] reductionDims = new int[input.dim() - 1];
            int position = 0;
            for (int dim = 0; dim < input.dim(); dim++) {
                if (dim != 1) {
                    reductionDims[position++] = dim;
                }
            }
            mean = input.mean(reductionDims, true);
            Tensor centered = input.sub(mean);
            variance = centered.mul(centered).mean(reductionDims, true);
            long count = input.numel() / channels;
            if (!input.device().isMeta() && runningMean != null) {
                Tensor meanFlat = mean.detach().reshape(channels);
                runningMean.mul_(1 - momentum).add_(meanFlat, momentum);
            }
            if (!input.device().isMeta() && runningVar != null) {
                double unbiased = count > 1 ? (double) count / (count - 1) : 1.0;
                Tensor varianceFlat = variance.detach().reshape(channels).mul(unbiased);
                runningVar.mul_(1 - momentum).add_(varianceFlat, momentum);
            }
        } else {
            mean = runningMean.reshape(statShape.toArray());
            variance = runningVar.reshape(statShape.toArray());
        }
        Tensor normalized = input.sub(mean).div(variance.add(eps).sqrt());
        if (weight != null) {
            normalized = normalized.mul(weight.reshape(statShape.toArray()));
        }
        if (bias != null) {
            normalized = normalized.add(bias.reshape(statShape.toArray()));
        }
        return new Value[] {Value.tensor(normalized)};
    }

    private static long[] broadcastShape(int rank, long[] normalizedShape) {
        long[] shape = new long[rank];
        int prefix = rank - normalizedShape.length;
        for (int i = 0; i < prefix; i++) {
            shape[i] = 1;
        }
        System.arraycopy(normalizedShape, 0, shape, prefix, normalizedShape.length);
        return shape;
    }

    private static Value[] layerNorm(org.jtorch.core.Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        long[] normalizedShape = args[1].asLongArray();
        Tensor weight = args[2].asOptionalTensor();
        Tensor bias = args[3].asOptionalTensor();
        double eps = args[4].asScalar().toDouble();
        if (normalizedShape.length == 0 || normalizedShape.length > input.dim()) {
            throw new ShapeException("layer_norm normalized_shape rank is invalid for " + input.shape());
        }
        if (!input.dtype().isFloatingPoint()) {
            throw new org.jtorch.core.DTypeException("layer_norm requires a floating point input");
        }
        int[] dims = new int[normalizedShape.length];
        for (int i = 0; i < normalizedShape.length; i++) {
            int dim = input.dim() - normalizedShape.length + i;
            if (input.size(dim) != normalizedShape[i]) {
                throw new ShapeException("layer_norm normalized_shape " + java.util.Arrays.toString(normalizedShape)
                        + " does not match input " + input.shape());
            }
            dims[i] = dim;
        }
        for (Tensor tensor : new Tensor[] {weight, bias}) {
            if (tensor != null && (!tensor.shape().equals(Shape.of(normalizedShape))
                    || tensor.dtype() != input.dtype())) {
                throw new ShapeException("layer_norm affine parameter must match normalized_shape and dtype");
            }
        }
        Tensor mean = input.mean(dims, true);
        Tensor centered = input.sub(mean);
        Tensor variance = centered.mul(centered).mean(dims, true);
        Tensor normalized = centered.div(variance.add(eps).sqrt());
        if (weight != null) {
            normalized = normalized.mul(weight.reshape(broadcastShape(input.dim(), normalizedShape)));
        }
        if (bias != null) {
            normalized = normalized.add(bias.reshape(broadcastShape(input.dim(), normalizedShape)));
        }
        return new Value[] {Value.tensor(normalized)};
    }

    private static Value[] groupNorm(org.jtorch.core.Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        int groups = args[1].asInt();
        Tensor weight = args[2].asOptionalTensor();
        Tensor bias = args[3].asOptionalTensor();
        double eps = args[4].asScalar().toDouble();
        if (input.dim() != 4 || !input.dtype().isFloatingPoint()) {
            throw new ShapeException("group_norm expects a floating NCHW tensor, got " + input.shape());
        }
        int channels = (int) input.size(1);
        if (groups <= 0 || channels % groups != 0) {
            throw new ShapeException("group_norm groups must divide channels");
        }
        for (Tensor tensor : new Tensor[] {weight, bias}) {
            if (tensor != null && (tensor.dim() != 1 || tensor.size(0) != channels
                    || tensor.dtype() != input.dtype())) {
                throw new ShapeException("group_norm affine parameter must have shape [" + channels + "]");
            }
        }
        long batch = input.size(0);
        Tensor grouped = input.reshape(batch, groups, channels / groups, input.size(2), input.size(3));
        Tensor mean = grouped.mean(new int[] {2, 3, 4}, true);
        Tensor centered = grouped.sub(mean);
        Tensor variance = centered.mul(centered).mean(new int[] {2, 3, 4}, true);
        Tensor normalized = centered.div(variance.add(eps).sqrt());
        normalized = normalized.reshape(input.shape().toArray());
        if (weight != null) {
            normalized = normalized.mul(weight.reshape(1, channels, 1, 1));
        }
        if (bias != null) {
            normalized = normalized.add(bias.reshape(1, channels, 1, 1));
        }
        return new Value[] {Value.tensor(normalized)};
    }

    private static Value[] mseLoss(org.jtorch.core.Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor target = args[1].asTensor();
        String reduction = args[2].asString();
        if (!reduction.equals("mean") && !reduction.equals("sum") && !reduction.equals("none")) {
            throw new IllegalArgumentException("invalid reduction '" + reduction + "'");
        }
        Tensor difference = input.sub(target);
        Tensor squared = difference.mul(difference);
        Tensor output = switch (reduction) {
            case "mean" -> squared.mean();
            case "sum" -> squared.sum();
            default -> squared;
        };
        return new Value[] {Value.tensor(output)};
    }
}
