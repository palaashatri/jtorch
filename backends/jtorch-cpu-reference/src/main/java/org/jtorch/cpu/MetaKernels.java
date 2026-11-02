package org.jtorch.cpu;

import java.util.List;
import org.jtorch.core.DType;
import org.jtorch.core.GradMode;
import org.jtorch.core.Op;
import org.jtorch.core.OpRegistry;
import org.jtorch.core.Scalar;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Strides;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorImpl;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.dispatch.DispatchKey;
import org.jtorch.core.value.Value;

/** Shape-only kernels. Every non-composite operator has a meta implementation. */
final class MetaKernels {
    private MetaKernels() {}

    static void register(Dispatcher dispatcher) {
        // Creation
        dispatcher.registerTensor(OpRegistry.ZEROS, DispatchKey.META, MetaKernels::creation);
        dispatcher.registerTensor(OpRegistry.ONES, DispatchKey.META, MetaKernels::creation);
        dispatcher.registerTensor(OpRegistry.EMPTY, DispatchKey.META, MetaKernels::creation);
        dispatcher.registerTensor(OpRegistry.FULL, DispatchKey.META, MetaKernels::creation);
        dispatcher.registerTensor(OpRegistry.ARANGE, DispatchKey.META, MetaKernels::creation);
        dispatcher.registerTensor(OpRegistry.RANDN, DispatchKey.META, MetaKernels::creation);
        dispatcher.registerTensor(OpRegistry.RAND, DispatchKey.META, MetaKernels::creation);
        dispatcher.registerTensor(OpRegistry.RANDINT, DispatchKey.META, MetaKernels::creation);
        dispatcher.registerTensor(OpRegistry.EYE, DispatchKey.META, MetaKernels::creation);

        // Views
        dispatcher.registerTensor(OpRegistry.RESHAPE, DispatchKey.META, (op, args, keys) -> reshape(op, args, false));
        dispatcher.registerTensor(OpRegistry.VIEW, DispatchKey.META, (op, args, keys) -> reshape(op, args, true));
        dispatcher.registerTensor(OpRegistry.TRANSPOSE, DispatchKey.META, MetaKernels::transpose);
        dispatcher.registerTensor(OpRegistry.PERMUTE, DispatchKey.META, MetaKernels::permute);
        dispatcher.registerTensor(OpRegistry.NARROW, DispatchKey.META, MetaKernels::narrow);
        dispatcher.registerTensor(OpRegistry.SELECT, DispatchKey.META, MetaKernels::select);
        dispatcher.registerTensor(OpRegistry.SQUEEZE, DispatchKey.META, (op, args, keys) -> squeeze(op, args, false));
        dispatcher.registerTensor(OpRegistry.SQUEEZE_DIM, DispatchKey.META, (op, args, keys) -> squeeze(op, args, true));
        dispatcher.registerTensor(OpRegistry.UNSQUEEZE, DispatchKey.META, MetaKernels::unsqueeze);
        dispatcher.registerTensor(OpRegistry.EXPAND, DispatchKey.META, MetaKernels::expand);
        dispatcher.registerTensor(OpRegistry.FLATTEN, DispatchKey.META, MetaKernels::flatten);
        dispatcher.registerTensor(OpRegistry.T, DispatchKey.META, MetaKernels::t);
        dispatcher.registerTensor(OpRegistry.CONTIGUOUS, DispatchKey.META, (op, args, keys) -> contiguous(op, args));
        dispatcher.registerTensor(OpRegistry.CLONE, DispatchKey.META, (op, args, keys) -> cloneKernel(op, args));
        dispatcher.registerTensor(OpRegistry.DETACH, DispatchKey.META, MetaKernels::detach);
        dispatcher.registerTensor(OpRegistry.TO_DTYPE, DispatchKey.META, MetaKernels::to);

        // Elementwise
        dispatcher.registerTensor(OpRegistry.ADD_TENSOR, DispatchKey.META, MetaKernels::binary);
        dispatcher.registerTensor(OpRegistry.SUB_TENSOR, DispatchKey.META, MetaKernels::binary);
        dispatcher.registerTensor(OpRegistry.MUL_TENSOR, DispatchKey.META, MetaKernels::binary);
        dispatcher.registerTensor(OpRegistry.DIV_TENSOR, DispatchKey.META, MetaKernels::binary);
        dispatcher.registerTensor(OpRegistry.POW_TENSOR_SCALAR, DispatchKey.META, MetaKernels::powScalar);
        dispatcher.registerTensor(OpRegistry.POW_TENSOR_TENSOR, DispatchKey.META, MetaKernels::binary);
        dispatcher.registerTensor(OpRegistry.MAXIMUM, DispatchKey.META, MetaKernels::binary);
        dispatcher.registerTensor(OpRegistry.MINIMUM, DispatchKey.META, MetaKernels::binary);
        dispatcher.registerTensor(OpRegistry.EQ_TENSOR, DispatchKey.META, MetaKernels::comparison);
        dispatcher.registerTensor(OpRegistry.NE_TENSOR, DispatchKey.META, MetaKernels::comparison);
        dispatcher.registerTensor(OpRegistry.LT_TENSOR, DispatchKey.META, MetaKernels::comparison);
        dispatcher.registerTensor(OpRegistry.LE_TENSOR, DispatchKey.META, MetaKernels::comparison);
        dispatcher.registerTensor(OpRegistry.GT_TENSOR, DispatchKey.META, MetaKernels::comparison);
        dispatcher.registerTensor(OpRegistry.GE_TENSOR, DispatchKey.META, MetaKernels::comparison);
        dispatcher.registerTensor(OpRegistry.LOGICAL_NOT, DispatchKey.META, MetaKernels::logicalNot);

        for (Op op : List.of(
                OpRegistry.NEG,
                OpRegistry.ABS,
                OpRegistry.SQUARE,
                OpRegistry.SQRT,
                OpRegistry.EXP,
                OpRegistry.LOG,
                OpRegistry.TANH,
                OpRegistry.SIGMOID,
                OpRegistry.RELU,
                OpRegistry.RECIPROCAL,
                OpRegistry.FLOOR,
                OpRegistry.CEIL,
                OpRegistry.SIGN,
                OpRegistry.SIN,
                OpRegistry.COS,
                OpRegistry.LOG1P)) {
            dispatcher.registerTensor(op, DispatchKey.META, MetaKernels::unary);
        }
        dispatcher.registerTensor(OpRegistry.CLAMP, DispatchKey.META, MetaKernels::clamp);

        // Reductions
        dispatcher.registerTensor(OpRegistry.SUM, DispatchKey.META, (op, args, keys) -> reduction(op, args, false, false));
        dispatcher.registerTensor(
                OpRegistry.SUM_DIM_INTLIST, DispatchKey.META, (op, args, keys) -> reduction(op, args, true, false));
        dispatcher.registerTensor(OpRegistry.MEAN, DispatchKey.META, (op, args, keys) -> reduction(op, args, false, true));
        dispatcher.registerTensor(OpRegistry.MEAN_DIM, DispatchKey.META, (op, args, keys) -> reduction(op, args, true, true));
        dispatcher.registerTensor(OpRegistry.PROD, DispatchKey.META, (op, args, keys) -> reduction(op, args, false, false));
        dispatcher.registerTensor(
                OpRegistry.PROD_DIM, DispatchKey.META, (op, args, keys) -> reduction(op, args, true, false));
        dispatcher.registerTensor(OpRegistry.MAX, DispatchKey.META, (op, args, keys) -> reduction(op, args, false, false));
        dispatcher.registerTensor(OpRegistry.MIN, DispatchKey.META, (op, args, keys) -> reduction(op, args, false, false));
        dispatcher.register(OpRegistry.MAX_DIM, DispatchKey.META, MetaKernels::maxDim);
        dispatcher.register(OpRegistry.MIN_DIM, DispatchKey.META, MetaKernels::maxDim);
        dispatcher.registerTensor(OpRegistry.ARGMAX, DispatchKey.META, MetaKernels::argReduce);
        dispatcher.registerTensor(OpRegistry.ARGMIN, DispatchKey.META, MetaKernels::argReduce);

        // Matrix
        dispatcher.registerTensor(OpRegistry.MATMUL, DispatchKey.META, MetaKernels::matmul);
        dispatcher.registerTensor(OpRegistry.MM, DispatchKey.META, MetaKernels::mm);
        dispatcher.registerTensor(OpRegistry.BMM, DispatchKey.META, MetaKernels::bmm);
        dispatcher.registerTensor(OpRegistry.ADDMM, DispatchKey.META, MetaKernels::addmm);

        // Combination
        dispatcher.registerTensor(OpRegistry.CAT, DispatchKey.META, MetaKernels::cat);
        dispatcher.registerTensor(OpRegistry.STACK, DispatchKey.META, MetaKernels::stack);
        dispatcher.registerTensor(OpRegistry.CONV2D, DispatchKey.META, MetaKernels::conv2d);
        dispatcher.registerTensor(OpRegistry.MAX_POOL2D, DispatchKey.META, MetaKernels::maxPool2d);
        dispatcher.registerTensor(OpRegistry.CONV2D_BACKWARD_INPUT, DispatchKey.META,
                MetaKernels::conv2dBackwardInput);
        dispatcher.registerTensor(OpRegistry.CONV2D_BACKWARD_WEIGHT, DispatchKey.META,
                MetaKernels::conv2dBackwardWeight);
        dispatcher.registerTensor(OpRegistry.MAX_POOL2D_BACKWARD, DispatchKey.META,
                MetaKernels::maxPool2dBackward);
        dispatcher.registerTensor(OpRegistry.AVG_POOL2D, DispatchKey.META, MetaKernels::avgPool2d);
        dispatcher.registerTensor(OpRegistry.AVG_POOL2D_BACKWARD, DispatchKey.META,
                MetaKernels::avgPool2dBackward);
        dispatcher.registerTensor(OpRegistry.ADAPTIVE_AVG_POOL2D, DispatchKey.META,
                MetaKernels::adaptiveAvgPool2d);
        dispatcher.registerTensor(OpRegistry.ADAPTIVE_AVG_POOL2D_BACKWARD, DispatchKey.META,
                MetaKernels::adaptiveAvgPool2dBackward);
        dispatcher.registerTensor(OpRegistry.INTERPOLATE, DispatchKey.META, MetaKernels::interpolate);
        dispatcher.registerTensor(OpRegistry.INTERPOLATE_BACKWARD, DispatchKey.META,
                MetaKernels::interpolateBackward);
        dispatcher.registerTensor(OpRegistry.GELU, DispatchKey.META, MetaKernels::gelu);
        dispatcher.registerTensor(OpRegistry.GELU_BACKWARD, DispatchKey.META, MetaKernels::geluBackward);
        dispatcher.registerTensor(OpRegistry.EMBEDDING, DispatchKey.META, MetaKernels::embedding);
        dispatcher.registerTensor(OpRegistry.EMBEDDING_BACKWARD, DispatchKey.META,
                MetaKernels::embeddingBackward);
        dispatcher.registerTensor(OpRegistry.CONSTANT_PAD_ND, DispatchKey.META, MetaKernels::constantPad);
        dispatcher.registerTensor(OpRegistry.CONSTANT_PAD_ND_BACKWARD, DispatchKey.META,
                MetaKernels::constantPadBackward);
        dispatcher.registerTensor(OpRegistry.FLIP, DispatchKey.META, MetaKernels::flip);
        dispatcher.registerTensor(OpRegistry.GATHER, DispatchKey.META, MetaKernels::gather);
        dispatcher.registerTensor(OpRegistry.GATHER_BACKWARD, DispatchKey.META, MetaKernels::gatherBackward);
        dispatcher.registerTensor(OpRegistry.INDEX_SELECT, DispatchKey.META, MetaKernels::indexSelect);
        dispatcher.registerTensor(OpRegistry.INDEX_SELECT_BACKWARD, DispatchKey.META,
                MetaKernels::indexSelectBackward);
        dispatcher.registerTensor(OpRegistry.SLICE_TENSOR, DispatchKey.META, MetaKernels::slice);
        dispatcher.registerTensor(OpRegistry.SLICE_BACKWARD, DispatchKey.META, MetaKernels::sliceBackward);
        dispatcher.registerTensor(OpRegistry.SCATTER, DispatchKey.META, MetaKernels::scatter);
        dispatcher.registerTensor(OpRegistry.SCATTER_BACKWARD_SELF, DispatchKey.META,
                MetaKernels::scatterBackwardSelf);
        dispatcher.registerTensor(OpRegistry.SCATTER_BACKWARD_SRC, DispatchKey.META,
                MetaKernels::scatterBackwardSrc);

        // Loss
        dispatcher.registerTensor(OpRegistry.CROSS_ENTROPY, DispatchKey.META, MetaKernels::crossEntropy);

        // In-place
        dispatcher.registerTensor(OpRegistry.ADD__TENSOR, DispatchKey.META, MetaKernels::inPlaceBinary);
        dispatcher.registerTensor(OpRegistry.SUB__TENSOR, DispatchKey.META, MetaKernels::inPlaceBinary);
        dispatcher.registerTensor(OpRegistry.MUL__TENSOR, DispatchKey.META, MetaKernels::inPlaceBinary);
        dispatcher.registerTensor(OpRegistry.DIV__TENSOR, DispatchKey.META, MetaKernels::inPlaceBinary);
        dispatcher.registerTensor(OpRegistry.COPY_, DispatchKey.META, MetaKernels::inPlaceCopy);
        dispatcher.registerTensor(OpRegistry.ZERO_, DispatchKey.META, MetaKernels::inPlaceFill);
        dispatcher.registerTensor(OpRegistry.FILL__SCALAR, DispatchKey.META, MetaKernels::inPlaceFill);
    }

    // ------------------------------------------------------------------ helpers

    private static boolean outputRequiresGrad(Op op, Value[] args) {
        if (!GradMode.isEnabled() || !op.differentiable()) {
            return false;
        }
        for (Value value : args) {
            if (value instanceof Value.TensorValue tensor && tensor.value().requiresGrad()) {
                return true;
            }
            if (value instanceof Value.TensorListValue list) {
                for (Tensor tensor : list.values()) {
                    if (tensor.requiresGrad()) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private static Tensor meta(Shape shape, DType dtype, Op op, Value[] args) {
        return Tensor.wrap(TensorImpl.meta(shape, dtype, outputRequiresGrad(op, args)));
    }

    private static Tensor creation(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        return switch (op.name()) {
            case "aten::zeros.default", "aten::ones.default", "aten::empty.default" ->
                meta(Shape.of(args[0].asLongArray()), args[1].asDType(), op, args);
            case "aten::randn.default", "aten::rand.default" -> {
                DType dtype = args[1].asDType();
                if (!dtype.isFloatingPoint()) {
                    throw new org.jtorch.core.DTypeException(op.name() + " requires a floating point dtype, got "
                            + dtype);
                }
                yield meta(Shape.of(args[0].asLongArray()), dtype, op, args);
            }
            case "aten::full.default" ->
                meta(Shape.of(args[0].asLongArray()), args[2].asDType(), op, args);
            case "aten::randint.default" ->
                meta(Shape.of(args[2].asLongArray()), args[3].asDType(), op, args);
            case "aten::eye.default" -> {
                long n = args[0].asLong();
                yield meta(Shape.of(n, n), args[1].asDType(), op, args);
            }
            case "aten::arange.default" -> {
                Scalar start = args[0].asScalar();
                Scalar end = args[1].asScalar();
                Scalar step = args[2].asScalar();
                DType dtype = args[3].asDType();
                if (dtype.isBoolean()) {
                    throw new org.jtorch.core.DTypeException("arange is not implemented for bool");
                }
                yield meta(Shape.of(arangeLength(start, end, step)), dtype, op, args);
            }
            default -> throw new IllegalStateException("Unexpected creation op " + op);
        };
    }

    static long arangeLength(Scalar start, Scalar end, Scalar step) {
        if (step.toDouble() == 0) {
            throw new IllegalArgumentException("arange: step must be nonzero");
        }
        if (start.isIntegral() && end.isIntegral() && step.isIntegral()) {
            java.math.BigInteger first = java.math.BigInteger.valueOf(start.toLong());
            java.math.BigInteger last = java.math.BigInteger.valueOf(end.toLong());
            java.math.BigInteger stride = java.math.BigInteger.valueOf(step.toLong());
            java.math.BigInteger[] quotient = last.subtract(first).divideAndRemainder(stride);
            java.math.BigInteger length = quotient[0];
            if (quotient[1].signum() != 0) {
                length = length.add(java.math.BigInteger.ONE);
            }
            if (length.signum() <= 0) {
                return 0;
            }
            if (length.compareTo(java.math.BigInteger.valueOf(Integer.MAX_VALUE)) > 0) {
                throw new ShapeException("arange produces too many elements: " + length);
            }
            return length.longValueExact();
        }
        double length = Math.ceil((end.toDouble() - start.toDouble()) / step.toDouble());
        if (length <= 0) {
            return 0;
        }
        if (length > Integer.MAX_VALUE) {
            throw new ShapeException("arange produces too many elements: " + length);
        }
        return (long) length;
    }

    private static Tensor reshape(Op op, Value[] args, boolean strict) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = Views.reshape(input, args[1].asLongArray(), strict);
        return Tensor.wrap(TensorImpl.meta(
                spec.shape(),
                spec.requiresCopy() ? Strides.contiguous(spec.shape()) : spec.strides(),
                input.dtype(),
                outputRequiresGrad(op, args)));
    }

    private static Tensor transpose(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = Views.transpose(input, args[1].asInt(), args[2].asInt());
        return metaView(input, spec, op, args);
    }

    private static Tensor permute(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = Views.permute(input, args[1].asLongArray());
        return metaView(input, spec, op, args);
    }

    private static Tensor narrow(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = Views.narrow(input, args[1].asInt(), args[2].asLong(), args[3].asLong());
        return metaView(input, spec, op, args);
    }

    private static Tensor select(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = Views.select(input, args[1].asInt(), args[2].asLong());
        return metaView(input, spec, op, args);
    }

    private static Tensor squeeze(Op op, Value[] args, boolean dim) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = dim ? Views.squeezeDim(input, args[1].asInt()) : Views.squeeze(input);
        return metaView(input, spec, op, args);
    }

    private static Tensor unsqueeze(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = Views.unsqueeze(input, args[1].asInt());
        return metaView(input, spec, op, args);
    }

    private static Tensor expand(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = Views.expand(input, args[1].asLongArray());
        return metaView(input, spec, op, args);
    }

    private static Tensor flatten(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        int start = args[1].asInt();
        int end = args[2].asInt();
        int rank = input.dim();
        if (rank == 0) {
            return Tensor.wrap(TensorImpl.meta(Shape.of(1), input.dtype(), outputRequiresGrad(op, args)));
        }
        int from = start < 0 ? start + rank : start;
        int to = end < 0 ? end + rank : end;
        if (from < 0 || to >= rank || from > to) {
            throw new IndexOutOfBoundsException("flatten has invalid dim range [" + start + ", " + end + "]");
        }
        long[] sizes = input.shape().toArray();
        long merged = 1;
        for (int i = from; i <= to; i++) {
            merged *= sizes[i];
        }
        long[] output = new long[rank - (to - from)];
        int target = 0;
        for (int i = 0; i < from; i++) {
            output[target++] = sizes[i];
        }
        output[target++] = merged;
        for (int i = to + 1; i < rank; i++) {
            output[target++] = sizes[i];
        }
        Shape shape = Shape.of(output);
        return Tensor.wrap(TensorImpl.meta(shape, input.dtype(), outputRequiresGrad(op, args)));
    }

    private static Tensor t(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        if (input.dim() <= 1) {
            return Tensor.wrap(TensorImpl.meta(input.shape(), input.strides(), input.dtype(),
                    outputRequiresGrad(op, args)));
        }
        if (input.dim() != 2) {
            throw new ShapeException("t() expects a tensor of at most 2 dimensions, got rank " + input.dim());
        }
        return metaView(input, Views.transpose(input, 0, 1), op, args);
    }

    private static Tensor contiguous(Op op, Value[] args) {
        Tensor input = args[0].asTensor();
        if (input.isContiguous()) {
            return Tensor.wrap(TensorImpl.meta(input.shape(), input.strides(), input.dtype(),
                    outputRequiresGrad(op, args)));
        }
        return Tensor.wrap(TensorImpl.meta(
                input.shape(), Strides.contiguous(input.shape()), input.dtype(), outputRequiresGrad(op, args)));
    }

    private static Tensor cloneKernel(Op op, Value[] args) {
        Tensor input = args[0].asTensor();
        Strides strides = input.isContiguous() || ShapeRules.isDense(input.shape(), input.strides())
                ? input.strides()
                : Strides.contiguous(input.shape());
        return Tensor.wrap(TensorImpl.meta(
                input.shape(), strides, input.dtype(), outputRequiresGrad(op, args)));
    }

    private static Tensor detach(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        return Tensor.wrap(TensorImpl.meta(input.shape(), input.strides(), input.dtype(), false));
    }

    private static Tensor to(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        DType target = args[1].asDType();
        boolean requires = outputRequiresGrad(op, args) && target.isFloatingPoint();
        if (target == input.dtype()) {
            if (requires == input.requiresGrad()) {
                return input;
            }
            return Tensor.wrap(TensorImpl.meta(input.shape(), input.strides(), target, requires));
        }
        Strides strides = input.isContiguous() || ShapeRules.isDense(input.shape(), input.strides())
                ? input.strides()
                : Strides.contiguous(input.shape());
        return Tensor.wrap(TensorImpl.meta(input.shape(), strides, target, requires));
    }

    private static Tensor metaView(Tensor input, Views.ViewSpec spec, Op op, Value[] args) {
        return Tensor.wrap(TensorImpl.meta(spec.shape(), spec.strides(), input.dtype(), outputRequiresGrad(op, args)));
    }

    private static Tensor binary(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor left = args[0].asTensor();
        Tensor right = args[1].asTensor();
        Shape shape = ShapeRules.broadcastShape(left, right);
        DType dtype = switch (op.name()) {
            case "aten::div.Tensor" -> ShapeRules.divisionDtype(left, right);
            case "aten::pow.Tensor_Tensor" -> ShapeRules.binaryDtype(left, right);
            default -> ShapeRules.binaryDtype(left, right);
        };
        return meta(shape, dtype, op, args);
    }

    private static Tensor powScalar(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor base = args[0].asTensor();
        Scalar exponent = args[1].asScalar();
        DType dtype = ShapeRules.scalarDtype(base, exponent);
        if (!dtype.isFloatingPoint() && exponent.isFloatingPoint()) {
            dtype = DType.defaultFloat();
        }
        if (!base.dtype().isFloatingPoint() && exponent.isFloatingPoint()) {
            dtype = DType.defaultFloat();
        }
        return meta(base.shape(), dtype, op, args);
    }

    private static Tensor comparison(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor left = args[0].asTensor();
        Tensor right = args[1].asTensor();
        return meta(ShapeRules.broadcastShape(left, right), DType.BOOL, op, args);
    }

    private static Tensor logicalNot(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        return meta(args[0].asTensor().shape(), DType.BOOL, op, args);
    }

    private static Tensor unary(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        if (input.dtype().isBoolean() && switch (op.name()) {
            case "aten::neg.default", "aten::abs.default", "aten::relu.default", "aten::floor.default",
                    "aten::ceil.default" -> true;
            default -> false;
        }) {
            throw new org.jtorch.core.DTypeException(op.name() + " is not implemented for bool tensors");
        }
        DType dtype = switch (op.name()) {
            case "aten::sqrt.default", "aten::exp.default", "aten::log.default", "aten::tanh.default",
                    "aten::sigmoid.default", "aten::reciprocal.default", "aten::sin.default", "aten::cos.default",
                    "aten::log1p.default" -> ShapeRules.floatMathDtype(input.dtype());
            case "aten::square.default" -> ShapeRules.integralPromotion(input.dtype());
            default -> input.dtype();
        };
        return meta(input.shape(), dtype, op, args);
    }

    private static Tensor clamp(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Double min = args[1].isNone() ? null : args[1].asScalar().toDouble();
        Double max = args[2].isNone() ? null : args[2].asScalar().toDouble();
        if (min != null && max != null && min > max) {
            throw new IllegalArgumentException("clamp: min (" + min + ") must not exceed max (" + max + ")");
        }
        return meta(args[0].asTensor().shape(), args[0].asTensor().dtype(), op, args);
    }

    private static Tensor reduction(Op op, Value[] args, boolean dims, boolean mean) {
        Tensor input = args[0].asTensor();
        int[] normalized;
        boolean keepdim;
        if (input.dim() == 0 && dims) {
            long[] rawDims = op.name().equals("aten::prod.dim")
                    ? new long[] {args[1].asLong()}
                    : args[1].asLongArray();
            if (!(rawDims.length == 0
                    || (rawDims.length == 1 && (rawDims[0] == 0 || rawDims[0] == -1)))) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            if (mean && !input.dtype().isFloatingPoint()) {
                throw new org.jtorch.core.DTypeException("mean(): could not infer output dtype from "
                        + input.dtype() + "; input must be floating point");
            }
            DType dtype = mean ? input.dtype() : ShapeRules.integralPromotion(input.dtype());
            return meta(Shape.scalar(), dtype, op, args);
        }
        if (dims) {
            long[] rawDims = op.name().equals("aten::prod.dim")
                    ? new long[] {args[1].asLong()}
                    : args[1].asLongArray();
            normalized = rawDims.length == 0
                    ? allDims(input.shape())
                    : ShapeRules.normalizeDims(input.shape(), rawDims);
            keepdim = args[2].asBoolean();
        } else {
            normalized = new int[input.dim()];
            for (int i = 0; i < input.dim(); i++) {
                normalized[i] = i;
            }
            keepdim = false;
        }
        Shape shape = ShapeRules.reductionShape(input.shape(), normalized, keepdim);
        if (op.name().equals("aten::mean.default") || op.name().equals("aten::mean.dim")) {
            if (!input.dtype().isFloatingPoint()) {
                throw new org.jtorch.core.DTypeException(
                        "mean(): could not infer output dtype from " + input.dtype()
                                + "; input must be floating point");
            }
        }
        DType dtype = switch (op.name()) {
            case "aten::mean.default", "aten::mean.dim" -> ShapeRules.floatMathDtype(input.dtype());
            case "aten::max.default", "aten::min.default", "aten::max.dim", "aten::min.dim" -> input.dtype();
            default -> ShapeRules.integralPromotion(input.dtype());
        };
        if ((op.name().equals("aten::max.default") || op.name().equals("aten::min.default"))
                && input.numel() == 0) {
            throw new ShapeException(op.name() + "(): Expected reduction dim to have non-zero size");
        }
        return meta(shape, dtype, op, args);
    }

    private static Value[] maxDim(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        boolean keepdim = args[2].asBoolean();
        if (input.dim() == 0) {
            int dim = args[1].asInt();
            if (dim != 0 && dim != -1) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            Shape shape = keepdim ? Shape.of(1) : Shape.scalar();
            return new Value[] {
                Value.tensor(meta(shape, input.dtype(), op, args)),
                Value.tensor(Tensor.wrap(TensorImpl.meta(shape, DType.INT64, false)))
            };
        }
        int dim = input.shape().normalizeDim(args[1].asInt());
        if (input.size(dim) == 0) {
            throw new ShapeException(op.name() + "(): Expected reduction dim to have non-zero size");
        }
        Shape shape = ShapeRules.reductionShape(input.shape(), new int[] {dim}, keepdim);
        return new Value[] {
            Value.tensor(meta(shape, input.dtype(), op, args)),
            Value.tensor(Tensor.wrap(TensorImpl.meta(shape, DType.INT64, false)))
        };
    }

    private static Tensor argReduce(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        if (input.numel() == 0) {
            throw new ShapeException(op.name() + "(): cannot reduce an empty tensor");
        }
        if (input.dtype().isBoolean()) {
            throw new org.jtorch.core.DTypeException(op.name() + "(): does not support bool input");
        }
        Integer dim = args[1].asOptionalInt();
        boolean keepdim = args[2].asBoolean();
        if (input.dim() == 0) {
            if (dim != null && dim != 0 && dim != -1) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            return meta(keepdim && dim != null ? Shape.of(1) : Shape.scalar(), DType.INT64, op, args);
        }
        Shape shape;
        if (dim == null) {
            shape = Shape.scalar();
        } else {
            int normalized = input.shape().normalizeDim(dim);
            if (input.size(normalized) == 0) {
                throw new ShapeException(op.name() + "(): Expected reduction dim to have non-zero size");
            }
            shape = ShapeRules.reductionShape(input.shape(), new int[] {normalized}, keepdim);
        }
        return meta(shape, DType.INT64, op, args);
    }

    private static Tensor matmul(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor left = args[0].asTensor();
        Tensor right = args[1].asTensor();
        if (left.dtype().isBoolean() || right.dtype().isBoolean()) {
            throw new org.jtorch.core.DTypeException("matmul is not implemented for bool tensors");
        }
        return meta(ShapeRules.matmulShape(left.shape(), right.shape()), ShapeRules.binaryDtype(left, right), op, args);
    }

    private static Tensor mm(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor left = args[0].asTensor();
        Tensor right = args[1].asTensor();
        if (left.dim() != 2 || right.dim() != 2) {
            throw new ShapeException("mm expects two 2-D tensors");
        }
        if (left.size(1) != right.size(0)) {
            throw new ShapeException("mat1 and mat2 shapes cannot be multiplied (" + left.shape() + " and "
                    + right.shape() + ")");
        }
        return meta(Shape.of(left.size(0), right.size(1)), ShapeRules.binaryDtype(left, right), op, args);
    }

    private static Tensor bmm(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor left = args[0].asTensor();
        Tensor right = args[1].asTensor();
        if (left.dim() != 3 || right.dim() != 3) {
            throw new ShapeException("bmm expects two 3-D tensors");
        }
        if (left.size(0) != right.size(0) || left.size(2) != right.size(1)) {
            throw new ShapeException("bmm shapes are incompatible: " + left.shape() + " and " + right.shape());
        }
        return meta(
                Shape.of(left.size(0), left.size(1), right.size(2)), ShapeRules.binaryDtype(left, right), op, args);
    }

    private static Tensor addmm(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        Tensor mat1 = args[1].asTensor();
        Tensor mat2 = args[2].asTensor();
        if (mat1.dim() != 2 || mat2.dim() != 2) {
            throw new ShapeException("addmm expects 2-D matrices, got " + mat1.shape() + " and " + mat2.shape());
        }
        Shape mmShape = ShapeRules.matmulShape(mat1.shape(), mat2.shape());
        Shape broadcast = Shape.broadcast(self.shape(), mmShape);
        if (!broadcast.equals(mmShape)) {
            throw new ShapeException("addmm: self shape " + self.shape() + " cannot broadcast to " + mmShape);
        }
        DType dtype = ShapeRules.binaryDtype(self, mat1);
        dtype = DType.resultType(dtype, mat2.dtype());
        return meta(mmShape, dtype, op, args);
    }

    private static Tensor cat(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor[] tensors = args[0].asTensorList();
        int dim = args[1].asInt();
        if (tensors.length == 0) {
            throw new ShapeException("cat expects a non-empty tensor list");
        }
        int rank = tensors[0].dim();
        int normalized = tensors[0].shape().normalizeDim(dim);
        long[] sizes = tensors[0].shape().toArray();
        DType dtype = tensors[0].dtype();
        long total = 0;
        for (Tensor tensor : tensors) {
            if (tensor.dim() != rank) {
                throw new ShapeException("cat: all tensors must have the same rank");
            }
            for (int i = 0; i < rank; i++) {
                if (i != normalized && tensor.size(i) != sizes[i]) {
                    throw new ShapeException("cat: sizes must match except in dimension " + normalized);
                }
            }
            total += tensor.size(normalized);
            dtype = DType.resultType(dtype, tensor.dtype());
        }
        sizes[normalized] = total;
        return meta(Shape.of(sizes), dtype, op, args);
    }

    private static Tensor stack(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor[] tensors = args[0].asTensorList();
        int dim = args[1].asInt();
        if (tensors.length == 0) {
            throw new ShapeException("stack expects a non-empty tensor list");
        }
        Shape base = tensors[0].shape();
        DType dtype = tensors[0].dtype();
        for (Tensor tensor : tensors) {
            if (!tensor.shape().equals(base)) {
                throw new ShapeException("stack: all tensors must have the same shape");
            }
            dtype = DType.resultType(dtype, tensor.dtype());
        }
        Shape output = base.withDim(dim < 0 ? dim + base.rank() + 1 : dim, tensors.length);
        return meta(output, dtype, op, args);
    }

    private static Tensor conv2d(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor weight = args[1].asTensor();
        Tensor bias = args[2].asOptionalTensor();
        Convolution.ConvConfig config = Convolution.convConfig(
                args[3].asLongArray(), args[4].asLongArray(), args[5].asLongArray(), args[6].asInt());
        Convolution.validateConv(input, weight, bias, config);
        return meta(Convolution.convOutputShape(input, weight, config), input.dtype(), op, args);
    }

    private static Tensor maxPool2d(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Convolution.PoolConfig config = Convolution.poolConfig(
                args[1].asLongArray(), args[2].asLongArray(), args[3].asLongArray(), args[4].asLongArray(),
                args[5].asBoolean());
        Convolution.validatePool(input, config);
        return meta(Convolution.poolOutputShape(input, config), input.dtype(), op, args);
    }

    private static Tensor conv2dBackwardInput(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor weight = args[1].asTensor();
        long[] inputShape = args[2].asLongArray();
        Convolution.ConvConfig config = Convolution.convConfig(
                args[3].asLongArray(), args[4].asLongArray(), args[5].asLongArray(), args[6].asInt());
        Convolution.validateConvBackwardInput(grad, weight, inputShape, config);
        return meta(Shape.of(inputShape), grad.dtype(), op, args);
    }

    private static Tensor conv2dBackwardWeight(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor grad = args[1].asTensor();
        long[] weightShape = args[2].asLongArray();
        Convolution.ConvConfig config = Convolution.convConfig(
                args[3].asLongArray(), args[4].asLongArray(), args[5].asLongArray(), args[6].asInt());
        Convolution.validateConvBackwardWeight(input, grad, weightShape, config);
        return meta(Shape.of(weightShape), grad.dtype(), op, args);
    }

    private static Tensor maxPool2dBackward(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor input = args[1].asTensor();
        Convolution.PoolConfig config = Convolution.poolConfig(
                args[2].asLongArray(), args[3].asLongArray(), args[4].asLongArray(), args[5].asLongArray(),
                args[6].asBoolean());
        Convolution.validateMaxPoolBackward(grad, input, config);
        return meta(input.shape(), grad.dtype(), op, args);
    }

    private static Tensor avgPool2d(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Convolution.AvgPoolConfig config = Convolution.avgPoolConfig(
                args[1].asLongArray(), args[2].asLongArray(), args[3].asLongArray(), args[4].asBoolean(),
                args[5].asBoolean(), args[6].asOptionalInt());
        Convolution.validateAvgPool(input, config);
        return meta(Convolution.avgPoolOutputShape(input, config), input.dtype(), op, args);
    }

    private static Tensor avgPool2dBackward(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor input = args[1].asTensor();
        Convolution.AvgPoolConfig config = Convolution.avgPoolConfig(
                args[2].asLongArray(), args[3].asLongArray(), args[4].asLongArray(), args[5].asBoolean(),
                args[6].asBoolean(), args[7].asOptionalInt());
        Convolution.validateAvgPool(input, config);
        Shape expected = Convolution.avgPoolOutputShape(input, config);
        if (!grad.shape().equals(expected)) {
            throw new org.jtorch.core.ShapeException("avg_pool2d backward grad shape mismatch");
        }
        return meta(input.shape(), grad.dtype(), op, args);
    }

    private static Tensor constantPad(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Convolution.PadConfig config = Convolution.padConfig(args[1].asLongArray(), input.dim());
        Shape shape = Convolution.padShape(input.shape(), config);
        return meta(shape, input.dtype(), op, args);
    }

    private static Tensor constantPadBackward(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Convolution.PadConfig config = Convolution.padConfig(args[1].asLongArray(), grad.dim());
        long[] shape = grad.shape().toArray();
        for (int i = 0; i < config.pairs(); i++) {
            int dim = shape.length - 1 - i;
            shape[dim] -= (long) config.pads()[2 * i] + config.pads()[2 * i + 1];
            if (shape[dim] <= 0) {
                throw new org.jtorch.core.ShapeException(
                        "constant_pad_nd backward produced a non-positive shape");
            }
        }
        return meta(Shape.of(shape), grad.dtype(), op, args);
    }

    private static Tensor flip(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        boolean[] seen = new boolean[input.dim()];
        for (long raw : args[1].asLongArray()) {
            int dim = input.shape().normalizeDim((int) raw);
            if (seen[dim]) {
                throw new org.jtorch.core.ShapeException("flip received duplicate dimension " + dim);
            }
            seen[dim] = true;
        }
        return meta(input.shape(), input.dtype(), op, args);
    }

    private static Tensor gather(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor index = args[2].asTensor();
        if (index.dtype() != org.jtorch.core.DType.INT64 && index.dtype() != org.jtorch.core.DType.INT32) {
            throw new org.jtorch.core.DTypeException("gather index must be int32 or int64");
        }
        if (input.dim() == 0) {
            if (index.dim() != 0 || (args[1].asInt() != 0 && args[1].asInt() != -1)) {
                throw new org.jtorch.core.ShapeException("gather on a 0-D tensor requires a 0-D index and dimension 0");
            }
            return meta(Shape.scalar(), input.dtype(), op, args);
        }
        int dim = input.shape().normalizeDim(args[1].asInt());
        if (index.dim() != input.dim()) {
            throw new org.jtorch.core.ShapeException("gather index rank must match input rank");
        }
        for (int d = 0; d < input.dim(); d++) {
            if (d != dim && index.size(d) > input.size(d)) {
                throw new org.jtorch.core.ShapeException("gather index size exceeds input at dimension " + d);
            }
        }
        return meta(index.shape(), input.dtype(), op, args);
    }

    private static Tensor gatherBackward(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor index = args[1].asTensor();
        long[] shape = args[2].asLongArray();
        int dim = Shape.of(shape).normalizeDim(args[3].asInt());
        if (grad.dim() != shape.length || index.dim() != shape.length
                || !grad.shape().equals(index.shape())) {
            throw new org.jtorch.core.ShapeException("gather backward shape mismatch");
        }
        for (int d = 0; d < shape.length; d++) {
            if (d != dim && index.size(d) > shape[d]) {
                throw new org.jtorch.core.ShapeException("gather backward index exceeds self");
            }
        }
        return meta(Shape.of(shape), grad.dtype(), op, args);
    }

    private static Tensor indexSelect(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        int dim = input.shape().normalizeDim(args[1].asInt());
        Tensor index = args[2].asTensor();
        if ((index.dtype() != org.jtorch.core.DType.INT64 && index.dtype() != org.jtorch.core.DType.INT32)
                || (index.dim() != 0 && index.dim() != 1)) {
            throw new org.jtorch.core.DTypeException("index_select index must be an int32/int64 scalar or vector");
        }
        long[] shape = input.shape().toArray();
        shape[dim] = index.numel();
        return meta(Shape.of(shape), input.dtype(), op, args);
    }

    private static Tensor indexSelectBackward(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor index = args[1].asTensor();
        long[] shape = args[2].asLongArray();
        int dim = Shape.of(shape).normalizeDim(args[3].asInt());
        if (grad.dim() != shape.length
                || (index.dim() != 0 && (index.dim() != 1 || grad.size(dim) != index.numel()))) {
            throw new org.jtorch.core.ShapeException("index_select backward shape mismatch");
        }
        return meta(Shape.of(shape), grad.dtype(), op, args);
    }

    private static Tensor slice(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        int dim = input.shape().normalizeDim(args[1].asInt());
        Integer startValue = args[2].asOptionalInt();
        Integer endValue = args[3].asOptionalInt();
        int step = args[4].asInt();
        if (step <= 0) {
            throw new ShapeException("slice step must be positive");
        }
        long size = input.size(dim);
        long start = startValue == null ? 0 : startValue;
        long end = endValue == null ? size : endValue;
        if (start < 0) {
            start += size;
        }
        if (end < 0) {
            end += size;
        }
        start = Math.max(0, Math.min(size, start));
        end = Math.max(0, Math.min(size, end));
        long length = end > start ? (end - start + step - 1) / step : 0;
        long[] shape = input.shape().toArray();
        shape[dim] = length;
        long[] strides = input.strides().toArray();
        strides[dim] *= step;
        return Tensor.wrap(TensorImpl.meta(Shape.of(shape), Strides.of(strides), input.dtype(),
                outputRequiresGrad(op, args)));
    }

    private static Tensor sliceBackward(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        long[] selfShape = args[1].asLongArray();
        int dim = Shape.of(selfShape).normalizeDim(args[2].asInt());
        int start = args[3].asInt();
        int length = args[4].asInt();
        int step = args[5].asInt();
        if (step <= 0 || length < 0 || start < 0 || start > selfShape[dim]
                || (length > 0 && start + (long) (length - 1) * step >= selfShape[dim])) {
            throw new ShapeException("invalid slice backward geometry");
        }
        long[] expected = selfShape.clone();
        expected[dim] = length;
        if (grad.dim() != selfShape.length || !grad.shape().equals(Shape.of(expected))) {
            throw new ShapeException("slice backward grad shape mismatch");
        }
        return meta(Shape.of(selfShape), grad.dtype(), op, args);
    }

    private static Tensor scatter(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        int dim = self.shape().normalizeDim(args[1].asInt());
        Tensor index = args[2].asTensor();
        Tensor src = args[3].asTensor();
        if (index.dtype() != org.jtorch.core.DType.INT64 && index.dtype() != org.jtorch.core.DType.INT32) {
            throw new org.jtorch.core.DTypeException("scatter index must be int32 or int64");
        }
        if (index.dim() != self.dim() || src.dim() != self.dim()) {
            throw new ShapeException("scatter expects index/src with the same rank as self");
        }
        for (int d = 0; d < self.dim(); d++) {
            if ((d != dim && index.size(d) > self.size(d)) || index.size(d) > src.size(d)) {
                throw new ShapeException("scatter index/src exceeds self at dimension " + d);
            }
        }
        return meta(self.shape(), self.dtype(), op, args);
    }

    private static Tensor scatterBackwardSelf(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor index = args[1].asTensor();
        int dim = grad.shape().normalizeDim(args[2].asInt());
        if (index.dim() != grad.dim()) {
            throw new ShapeException("scatter backward index rank must match grad rank");
        }
        for (int d = 0; d < grad.dim(); d++) {
            if (d != dim && index.size(d) > grad.size(d)) {
                throw new ShapeException("scatter backward index exceeds grad");
            }
        }
        return meta(grad.shape(), grad.dtype(), op, args);
    }

    private static Tensor scatterBackwardSrc(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor index = args[1].asTensor();
        int dim = grad.shape().normalizeDim(args[2].asInt());
        long[] srcShape = args[3].asLongArray();
        if (srcShape.length != grad.dim() || index.dim() != grad.dim()) {
            throw new ShapeException("scatter backward src/index rank mismatch");
        }
        for (int d = 0; d < grad.dim(); d++) {
            if ((d != dim && index.size(d) > grad.size(d)) || index.size(d) > srcShape[d]) {
                throw new ShapeException("scatter backward src/index geometry mismatch");
            }
        }
        return meta(Shape.of(srcShape), grad.dtype(), op, args);
    }


    private static Tensor adaptiveAvgPool2d(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        ImageOps.validatePoolInput(input, "adaptive_avg_pool2d");
        int[] size = ImageOps.adaptiveOutputSize(args[1].asLongArray(),
                input.size(input.dim() - 2), input.size(input.dim() - 1));
        return meta(ImageOps.adaptiveShape(input, size), input.dtype(), op, args);
    }

    private static Tensor adaptiveAvgPool2dBackward(
            Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        long[] inputShape = args[1].asLongArray();
        if (inputShape.length != 3 && inputShape.length != 4) {
            throw new ShapeException("adaptive_avg_pool2d input_shape must have three or four entries");
        }
        int height = (int) inputShape[inputShape.length - 2];
        int width = (int) inputShape[inputShape.length - 1];
        int[] size = ImageOps.adaptiveOutputSize(args[2].asLongArray(), height, width);
        if (inputShape.length == 4) {
            Shape expected = Shape.of(inputShape[0], inputShape[1], size[0], size[1]);
            if (!grad.shape().equals(expected)) {
                throw new ShapeException("adaptive_avg_pool2d backward grad shape mismatch");
            }
        } else {
            Shape expected = Shape.of(inputShape[0], size[0], size[1]);
            if (!grad.shape().equals(expected)) {
                throw new ShapeException("adaptive_avg_pool2d backward grad shape mismatch");
            }
        }
        return meta(Shape.of(inputShape), grad.dtype(), op, args);
    }

    private static Tensor interpolate(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        ImageOps.validateImageInput(input);
        int[] size = ImageOps.interpolateSize(args[1].asLongArray(), input);
        ImageOps.validateMode(args[2].asString(), args[3].asBoolean());
        return meta(ImageOps.interpolateShape(input, size), input.dtype(), op, args);
    }

    private static Tensor interpolateBackward(
            Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        long[] inputShape = args[1].asLongArray();
        if (inputShape.length != 4) {
            throw new ShapeException("interpolate input_shape must have four entries");
        }
        int[] size = ImageOps.interpolateSize(args[2].asLongArray(), grad);
        ImageOps.validateMode(args[3].asString(), args[4].asBoolean());
        Shape expected = Shape.of(inputShape[0], inputShape[1], size[0], size[1]);
        if (!grad.shape().equals(expected)) {
            throw new ShapeException("interpolate backward grad shape mismatch");
        }
        return meta(Shape.of(inputShape), grad.dtype(), op, args);
    }

    private static Tensor gelu(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        ImageOps.tanhApproximation(args[1].asString());
        if (!input.dtype().isFloatingPoint()) {
            throw new org.jtorch.core.DTypeException("gelu requires a floating point tensor");
        }
        return meta(input.shape(), input.dtype(), op, args);
    }

    private static Tensor geluBackward(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor input = args[1].asTensor();
        ImageOps.tanhApproximation(args[2].asString());
        if (!grad.shape().equals(input.shape())) {
            throw new ShapeException("gelu backward grad shape mismatch");
        }
        return meta(input.shape(), grad.dtype(), op, args);
    }

    private static Tensor embedding(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor weight = args[0].asTensor();
        Tensor indices = args[1].asTensor();
        if (weight.dim() != 2 || !weight.dtype().isFloatingPoint()) {
            throw new ShapeException("embedding weight must be a floating 2-D tensor");
        }
        if (indices.dtype() != org.jtorch.core.DType.INT64
                && indices.dtype() != org.jtorch.core.DType.INT32) {
            throw new org.jtorch.core.DTypeException("embedding indices must be int32 or int64");
        }
        if (args[4].asBoolean()) {
            throw new org.jtorch.core.NotSupportedException("sparse embedding gradients are not implemented");
        }
        if (!args[2].isNone()) {
            int padding = args[2].asInt();
            long normalized = padding < 0 ? padding + weight.size(0) : padding;
            if (normalized < 0 || normalized >= weight.size(0)) {
                throw new IndexOutOfBoundsException("embedding padding_idx is out of range");
            }
        }
        long[] shape = new long[indices.dim() + 1];
        for (int d = 0; d < indices.dim(); d++) {
            shape[d] = indices.size(d);
        }
        shape[indices.dim()] = weight.size(1);
        return meta(Shape.of(shape), weight.dtype(), op, args);
    }

    private static Tensor embeddingBackward(
            Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        long[] weightShape = args[2].asLongArray();
        if (weightShape.length != 2) {
            throw new ShapeException("embedding weight_shape must have two entries");
        }
        return meta(Shape.of(weightShape), args[0].asTensor().dtype(), op, args);
    }

    private static Tensor crossEntropy(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor logits = args[0].asTensor();
        Tensor target = args[1].asTensor();
        String reduction = args[2].asString();
        validateCrossEntropy(logits, target, reduction);
        Shape shape = reduction.equals("none") ? Shape.of(logits.size(0)) : Shape.scalar();
        return meta(shape, logits.dtype(), op, args);
    }

    static int[] allDims(Shape shape) {
        int[] dims = new int[shape.rank()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = i;
        }
        return dims;
    }

    static void validateCrossEntropy(Tensor logits, Tensor target, String reduction) {
        if (!logits.dtype().isFloatingPoint()) {
            throw new org.jtorch.core.DTypeException(
                    "cross_entropy expects floating point logits, got " + logits.dtype());
        }
        if (logits.dim() != 2) {
            throw new ShapeException(
                    "cross_entropy in this build supports 2-D logits only, got " + logits.shape());
        }
        if (!target.dtype().isIntegral() || target.dtype() == DType.INT32) {
            throw new org.jtorch.core.DTypeException("cross_entropy target must be int64, got " + target.dtype());
        }
        if (target.dim() != logits.dim() - 1) {
            throw new ShapeException("cross_entropy target rank must be logits rank - 1");
        }
        for (int i = 0; i < target.dim(); i++) {
            if (target.size(i) != logits.size(i)) {
                throw new ShapeException("cross_entropy target shape " + target.shape() + " does not match logits "
                        + logits.shape());
            }
        }
        if (!reduction.equals("mean") && !reduction.equals("sum") && !reduction.equals("none")) {
            throw new IllegalArgumentException("invalid reduction '" + reduction + "'");
        }
    }

    private static Tensor inPlaceBinary(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        Tensor other = args[1].asTensor();
        ShapeRules.requireScaling(self, op.name());
        org.jtorch.core.DType resultDtype = op.name().equals("aten::div_.Tensor")
                ? ShapeRules.divisionDtype(self, other)
                : ShapeRules.binaryDtype(self, other);
        if (resultDtype != self.dtype()) {
            throw new org.jtorch.core.DTypeException("result type " + resultDtype + " can't be cast to "
                    + self.dtype() + " for in-place operation " + op.name());
        }
        if (hasInternalOverlap(self)) {
            throw new org.jtorch.core.JTorchException("unsupported operation: more than one element of the "
                    + "written-to tensor refers to a single memory location");
        }
        Shape broadcast = Shape.broadcast(self.shape(), other.shape());
        if (!broadcast.equals(self.shape())) {
            throw new ShapeException("in-place op cannot expand self from " + self.shape() + " to " + broadcast);
        }
        self.impl().bumpVersion();
        return self;
    }

    private static boolean hasInternalOverlap(Tensor tensor) {
        for (int dim = 0; dim < tensor.dim(); dim++) {
            if (tensor.size(dim) > 1 && tensor.stride(dim) == 0) {
                return true;
            }
        }
        return false;
    }

    private static Tensor inPlaceCopy(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        ShapeRules.requireScaling(self, op.name());
        Tensor source = args[1].asTensor();
        if (source.dim() > self.dim()) {
            throw new ShapeException("copy_: source rank exceeds target rank");
        }
        if (hasInternalOverlap(self)) {
            throw new org.jtorch.core.JTorchException("unsupported operation: more than one element of the "
                    + "written-to tensor refers to a single memory location");
        }
        int offset = self.dim() - source.dim();
        for (int i = 0; i < source.dim(); i++) {
            long targetSize = self.size(offset + i);
            long sourceSize = source.size(i);
            if (sourceSize != targetSize && sourceSize != 1) {
                throw new ShapeException("copy_: source shape " + source.shape()
                        + " cannot broadcast to target " + self.shape());
            }
        }
        self.impl().bumpVersion();
        return self;
    }

    private static Tensor inPlaceFill(Op op, Value[] args, org.jtorch.core.dispatch.DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        ShapeRules.requireScaling(self, op.name());
        if (hasInternalOverlap(self)) {
            throw new org.jtorch.core.JTorchException("unsupported operation: more than one element of the "
                    + "written-to tensor refers to a single memory location");
        }
        self.impl().bumpVersion();
        return self;
    }
}
