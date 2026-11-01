package org.jtorch.cpu;

import org.jtorch.core.OpRegistry;
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
