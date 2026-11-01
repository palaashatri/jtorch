package org.jtorch.cpu;

import java.util.List;
import org.jtorch.core.CpuAllocator;
import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
import org.jtorch.core.Device;
import org.jtorch.core.DeviceException;
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
import org.jtorch.core.dispatch.DispatchKeySet;
import org.jtorch.core.memory.TensorAccess;
import org.jtorch.core.value.Value;

/** The CPU reference backend: straightforward, strided, pure-Java kernels that define JTorch semantics. */
final class CpuKernels {
    private CpuKernels() {}

    static void register(Dispatcher dispatcher) {
        dispatcher.register(OpRegistry.ZEROS, DispatchKey.CPU, CpuKernels::creation);
        dispatcher.register(OpRegistry.ONES, DispatchKey.CPU, CpuKernels::creation);
        dispatcher.register(OpRegistry.EMPTY, DispatchKey.CPU, CpuKernels::creation);
        dispatcher.register(OpRegistry.FULL, DispatchKey.CPU, CpuKernels::creation);
        dispatcher.register(OpRegistry.ARANGE, DispatchKey.CPU, CpuKernels::creation);
        dispatcher.register(OpRegistry.RANDN, DispatchKey.CPU, CpuKernels::creation);
        dispatcher.register(OpRegistry.RAND, DispatchKey.CPU, CpuKernels::creation);
        dispatcher.register(OpRegistry.RANDINT, DispatchKey.CPU, CpuKernels::creation);
        dispatcher.register(OpRegistry.EYE, DispatchKey.CPU, CpuKernels::creation);

        dispatcher.register(OpRegistry.RESHAPE, DispatchKey.CPU, (op, args, keys) -> reshape(op, args, false));
        dispatcher.register(OpRegistry.VIEW, DispatchKey.CPU, (op, args, keys) -> reshape(op, args, true));
        dispatcher.register(OpRegistry.TRANSPOSE, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.PERMUTE, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.NARROW, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.SELECT, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.SQUEEZE, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.SQUEEZE_DIM, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.UNSQUEEZE, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.EXPAND, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.FLATTEN, DispatchKey.CPU, CpuKernels::flatten);
        dispatcher.register(OpRegistry.T, DispatchKey.CPU, CpuKernels::view);
        dispatcher.register(OpRegistry.CONTIGUOUS, DispatchKey.CPU, CpuKernels::contiguous);
        dispatcher.register(OpRegistry.CLONE, DispatchKey.CPU, CpuKernels::cloneKernel);
        dispatcher.register(OpRegistry.DETACH, DispatchKey.CPU, CpuKernels::detach);
        dispatcher.register(OpRegistry.TO_DTYPE, DispatchKey.CPU, CpuKernels::to);

        dispatcher.register(OpRegistry.ADD_TENSOR, DispatchKey.CPU, CpuKernels::binary);
        dispatcher.register(OpRegistry.SUB_TENSOR, DispatchKey.CPU, CpuKernels::binary);
        dispatcher.register(OpRegistry.MUL_TENSOR, DispatchKey.CPU, CpuKernels::binary);
        dispatcher.register(OpRegistry.DIV_TENSOR, DispatchKey.CPU, CpuKernels::binary);
        dispatcher.register(OpRegistry.POW_TENSOR_SCALAR, DispatchKey.CPU, CpuKernels::powScalar);
        dispatcher.register(OpRegistry.POW_TENSOR_TENSOR, DispatchKey.CPU, CpuKernels::binary);
        dispatcher.register(OpRegistry.MAXIMUM, DispatchKey.CPU, CpuKernels::binary);
        dispatcher.register(OpRegistry.MINIMUM, DispatchKey.CPU, CpuKernels::binary);
        dispatcher.register(OpRegistry.EQ_TENSOR, DispatchKey.CPU, CpuKernels::comparison);
        dispatcher.register(OpRegistry.NE_TENSOR, DispatchKey.CPU, CpuKernels::comparison);
        dispatcher.register(OpRegistry.LT_TENSOR, DispatchKey.CPU, CpuKernels::comparison);
        dispatcher.register(OpRegistry.LE_TENSOR, DispatchKey.CPU, CpuKernels::comparison);
        dispatcher.register(OpRegistry.GT_TENSOR, DispatchKey.CPU, CpuKernels::comparison);
        dispatcher.register(OpRegistry.GE_TENSOR, DispatchKey.CPU, CpuKernels::comparison);
        dispatcher.register(OpRegistry.LOGICAL_NOT, DispatchKey.CPU, CpuKernels::logicalNot);

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
            dispatcher.register(op, DispatchKey.CPU, CpuKernels::unary);
        }
        dispatcher.register(OpRegistry.CLAMP, DispatchKey.CPU, CpuKernels::clamp);

        dispatcher.register(OpRegistry.SUM, DispatchKey.CPU, (op, args, keys) -> reduce(op, args, false, false));
        dispatcher.register(OpRegistry.SUM_DIM_INTLIST, DispatchKey.CPU,
                (op, args, keys) -> reduce(op, args, true, false));
        dispatcher.register(OpRegistry.MEAN, DispatchKey.CPU, (op, args, keys) -> reduce(op, args, false, true));
        dispatcher.register(OpRegistry.MEAN_DIM, DispatchKey.CPU, (op, args, keys) -> reduce(op, args, true, true));
        dispatcher.register(OpRegistry.PROD, DispatchKey.CPU, (op, args, keys) -> reduce(op, args, false, false));
        dispatcher.register(OpRegistry.PROD_DIM, DispatchKey.CPU, (op, args, keys) -> reduce(op, args, true, false));
        dispatcher.register(OpRegistry.MAX, DispatchKey.CPU, (op, args, keys) -> reduce(op, args, false, false));
        dispatcher.register(OpRegistry.MIN, DispatchKey.CPU, (op, args, keys) -> reduce(op, args, false, false));
        dispatcher.register(OpRegistry.MAX_DIM, DispatchKey.CPU, CpuKernels::maxDim);
        dispatcher.register(OpRegistry.MIN_DIM, DispatchKey.CPU, CpuKernels::maxDim);
        dispatcher.register(OpRegistry.ARGMAX, DispatchKey.CPU, CpuKernels::argReduce);
        dispatcher.register(OpRegistry.ARGMIN, DispatchKey.CPU, CpuKernels::argReduce);

        dispatcher.register(OpRegistry.MATMUL, DispatchKey.CPU, CpuKernels::matmul);
        dispatcher.register(OpRegistry.MM, DispatchKey.CPU, CpuKernels::matmul);
        dispatcher.register(OpRegistry.BMM, DispatchKey.CPU, CpuKernels::matmul);
        dispatcher.register(OpRegistry.ADDMM, DispatchKey.CPU, CpuKernels::addmm);

        dispatcher.register(OpRegistry.CAT, DispatchKey.CPU, CpuKernels::cat);
        dispatcher.register(OpRegistry.STACK, DispatchKey.CPU, CpuKernels::stack);
        dispatcher.register(OpRegistry.CONV2D, DispatchKey.CPU, CpuKernels::conv2d);
        dispatcher.register(OpRegistry.MAX_POOL2D, DispatchKey.CPU, CpuKernels::maxPool2d);
        dispatcher.register(OpRegistry.CONV2D_BACKWARD_INPUT, DispatchKey.CPU, CpuKernels::conv2dBackwardInput);
        dispatcher.register(OpRegistry.CONV2D_BACKWARD_WEIGHT, DispatchKey.CPU, CpuKernels::conv2dBackwardWeight);
        dispatcher.register(OpRegistry.MAX_POOL2D_BACKWARD, DispatchKey.CPU, CpuKernels::maxPool2dBackward);
        dispatcher.register(OpRegistry.CROSS_ENTROPY, DispatchKey.CPU, CpuKernels::crossEntropy);

        dispatcher.register(OpRegistry.ADD__TENSOR, DispatchKey.CPU, CpuKernels::inPlaceBinary);
        dispatcher.register(OpRegistry.SUB__TENSOR, DispatchKey.CPU, CpuKernels::inPlaceBinary);
        dispatcher.register(OpRegistry.MUL__TENSOR, DispatchKey.CPU, CpuKernels::inPlaceBinary);
        dispatcher.register(OpRegistry.DIV__TENSOR, DispatchKey.CPU, CpuKernels::inPlaceBinary);
        dispatcher.register(OpRegistry.COPY_, DispatchKey.CPU, CpuKernels::inPlaceCopy);
        dispatcher.register(OpRegistry.ZERO_, DispatchKey.CPU, CpuKernels::inPlaceFill);
        dispatcher.register(OpRegistry.FILL__SCALAR, DispatchKey.CPU, CpuKernels::inPlaceFill);
    }

    // ------------------------------------------------------------------ helpers

    static boolean outputRequiresGrad(Op op, Value[] args) {
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

    static Tensor fresh(Shape shape, DType dtype, boolean requiresGrad) {
        return Tensor.wrap(TensorImpl.fresh(
                CpuAllocator.allocate(dtype, shape.numel()), shape, Strides.contiguous(shape), requiresGrad));
    }

    static Device requireCpuDevice(Device device) {
        if (device.isMeta()) {
            throw new DeviceException("CPU kernel received a meta device request");
        }
        if (!device.isCpu()) {
            throw new DeviceException("Device " + device + " has no JTorch backend in this build");
        }
        return device;
    }

    private static void fill(Tensor tensor, double value) {
        long numel = tensor.numel();
        for (long i = 0; i < numel; i++) {
            TensorAccess.setDouble(tensor.impl(), tensor.elementOffset(i), value);
        }
    }

    private static void fillScalar(Tensor tensor, Scalar value) {
        if (tensor.dtype().isIntegral() || tensor.dtype().isBoolean()) {
            fillLong(tensor, value.toLong(tensor.dtype()));
        } else {
            fill(tensor, value.toDouble(tensor.dtype()));
        }
    }

    private static void fillLong(Tensor tensor, long value) {
        long numel = tensor.numel();
        for (long i = 0; i < numel; i++) {
            TensorAccess.setLong(tensor.impl(), tensor.elementOffset(i), value);
        }
    }

    static long[] allDims(Shape shape) {
        long[] dims = new long[shape.rank()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = i;
        }
        return dims;
    }

    // ------------------------------------------------------------------ creation

    private static Value[] creation(Op op, Value[] args, DispatchKeySet keys) {
        Tensor result;
        switch (op.name()) {
            case "aten::zeros.default", "aten::ones.default", "aten::empty.default", "aten::randn.default",
                    "aten::rand.default" -> {
                Shape shape = Shape.of(args[0].asLongArray());
                DType dtype = args[1].asDType();
                requireCpuDevice(args[2].asDevice());
                if ((op.name().equals("aten::randn.default") || op.name().equals("aten::rand.default"))
                        && !dtype.isFloatingPoint()) {
                    throw new DTypeException(op.name() + " requires a floating point dtype, got " + dtype);
                }
                result = fresh(shape, dtype, false);
                switch (op.name()) {
                    case "aten::ones.default" -> fill(result, 1);
                    case "aten::randn.default" -> fillRandomNormal(result);
                    case "aten::rand.default" -> fillRandomUniform(result);
                    default -> {
                        // empty leaves memory uninitialised, like PyTorch.
                    }
                }
            }
            case "aten::full.default" -> {
                Shape shape = Shape.of(args[0].asLongArray());
                Scalar value = args[1].asScalar();
                DType dtype = args[2].asDType();
                requireCpuDevice(args[3].asDevice());
                result = fresh(shape, dtype, false);
                fillScalar(result, value);
            }
            case "aten::randint.default" -> {
                long low = args[0].asLong();
                long high = args[1].asLong();
                Shape shape = Shape.of(args[2].asLongArray());
                DType dtype = args[3].asDType();
                requireCpuDevice(args[4].asDevice());
                if (high <= low) {
                    throw new IllegalArgumentException("randint: high must be greater than low");
                }
                result = fresh(shape, dtype, false);
                for (long i = 0; i < result.numel(); i++) {
                    long sample = low + CpuRandom.nextLong(high - low);
                    TensorAccess.setLong(result.impl(), result.elementOffset(i), sample);
                }
            }
            case "aten::eye.default" -> {
                long n = args[0].asLong();
                DType dtype = args[1].asDType();
                requireCpuDevice(args[2].asDevice());
                if (n < 0) {
                    throw new IllegalArgumentException("eye: n must be non-negative");
                }
                result = fresh(Shape.of(n, n), dtype, false);
                fill(result, 0);
                for (long i = 0; i < n; i++) {
                    TensorAccess.setDouble(result.impl(), result.elementOffset(i * n + i), 1);
                }
            }
            case "aten::arange.default" -> {
                Scalar start = args[0].asScalar();
                Scalar end = args[1].asScalar();
                Scalar step = args[2].asScalar();
                DType dtype = args[3].asDType();
                requireCpuDevice(args[4].asDevice());
                if (dtype.isBoolean()) {
                    throw new DTypeException("arange is not implemented for bool");
                }
                long length = MetaKernels.arangeLength(start, end, step);
                result = fresh(Shape.of(length), dtype, false);
                boolean exact = dtype.isIntegral() && start.isIntegral() && step.isIntegral();
                if (exact) {
                    long value = start.toLong();
                    long increment = step.toLong();
                    for (long i = 0; i < length; i++) {
                        TensorAccess.setLong(result.impl(), i, value);
                        value += increment;
                    }
                } else {
                    double value = start.toDouble();
                    double increment = step.toDouble();
                    for (long i = 0; i < length; i++) {
                        TensorAccess.setDouble(result.impl(), i, value);
                        value += increment;
                    }
                }
            }
            default -> throw new IllegalStateException("Unexpected creation op " + op);
        }
        return new Value[] {Value.tensor(result)};
    }

    private static void fillRandomNormal(Tensor tensor) {
        for (long i = 0; i < tensor.numel(); i++) {
            TensorAccess.setDouble(tensor.impl(), tensor.elementOffset(i), CpuRandom.nextNormal());
        }
    }

    private static void fillRandomUniform(Tensor tensor) {
        for (long i = 0; i < tensor.numel(); i++) {
            TensorAccess.setDouble(tensor.impl(), tensor.elementOffset(i), CpuRandom.nextDouble());
        }
    }

    // ------------------------------------------------------------------ views

    private static Value[] reshape(Op op, Value[] args, boolean strict) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = Views.reshape(input, args[1].asLongArray(), strict);
        Tensor output;
        if (spec.requiresCopy()) {
            Tensor copy = Views.contiguousCopy(input, input.shape(), input.requiresGrad());
            output = Views.cpuView(copy, Views.copied(spec.shape()), outputRequiresGrad(op, args));
        } else {
            output = Views.cpuView(input, spec, outputRequiresGrad(op, args));
        }
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] flatten(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        int rank = input.dim();
        if (rank == 0) {
            Views.ViewSpec spec = Views.reshape(input, new long[] {1}, false);
            return new Value[] {Value.tensor(Views.cpuView(input, spec, outputRequiresGrad(op, args)))};
        }
        int from = args[1].asInt();
        int to = args[2].asInt();
        from = from < 0 ? from + rank : from;
        to = to < 0 ? to + rank : to;
        if (from < 0 || to >= rank || from > to) {
            throw new IndexOutOfBoundsException("flatten has invalid dim range");
        }
        long[] sizes = input.shape().toArray();
        long merged = 1;
        for (int i = from; i <= to; i++) {
            merged *= sizes[i];
        }
        long[] target = new long[rank - (to - from)];
        int index = 0;
        for (int i = 0; i < from; i++) {
            target[index++] = sizes[i];
        }
        target[index++] = merged;
        for (int i = to + 1; i < rank; i++) {
            target[index++] = sizes[i];
        }
        Views.ViewSpec spec = Views.reshape(input, target, false);
        Tensor output = spec.requiresCopy()
                ? Views.cpuView(
                        Views.contiguousCopy(input, input.shape(), input.requiresGrad()),
                        Views.copied(spec.shape()),
                        outputRequiresGrad(op, args))
                : Views.cpuView(input, spec, outputRequiresGrad(op, args));
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] view(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Views.ViewSpec spec = switch (op.name()) {
            case "aten::transpose.default" -> Views.transpose(input, args[1].asInt(), args[2].asInt());
            case "aten::permute.default" -> Views.permute(input, args[1].asLongArray());
            case "aten::narrow.default" ->
                Views.narrow(input, args[1].asInt(), args[2].asLong(), args[3].asLong());
            case "aten::select.default" -> Views.select(input, args[1].asInt(), args[2].asLong());
            case "aten::squeeze.default" -> Views.squeeze(input);
            case "aten::squeeze.dim" -> Views.squeezeDim(input, args[1].asInt());
            case "aten::unsqueeze.default" -> Views.unsqueeze(input, args[1].asInt());
            case "aten::expand.default" -> Views.expand(input, args[1].asLongArray());
            case "aten::t.default" -> {
                if (input.dim() <= 1) {
                    yield new Views.ViewSpec(input.shape(), input.strides(), input.storageOffset(), false);
                }
                if (input.dim() != 2) {
                    throw new ShapeException("t() expects a tensor of at most 2 dimensions, got rank " + input.dim());
                }
                yield Views.transpose(input, 0, 1);
            }
            default -> throw new IllegalStateException("Unexpected view op " + op);
        };
        return new Value[] {Value.tensor(Views.cpuView(input, spec, outputRequiresGrad(op, args)))};
    }

    private static Value[] contiguous(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        if (input.isContiguous()) {
            return new Value[] {Value.tensor(input)};
        }
        Tensor output = Views.contiguousCopy(input, input.shape(), outputRequiresGrad(op, args));
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] cloneKernel(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        if (!input.isContiguous() && !isDense(input)) {
            return new Value[] {Value.tensor(
                    Views.contiguousCopy(input, input.shape(), outputRequiresGrad(op, args)))};
        }
        Tensor output = Views.empty(input, input.shape(), input.strides(), outputRequiresGrad(op, args));
        Copy.copy(input, output);
        return new Value[] {Value.tensor(output)};
    }

    private static boolean isDense(Tensor tensor) {
        if (tensor.numel() == 0) {
            return false;
        }
        long min = tensor.storageOffset();
        long max = min;
        for (int dim = 0; dim < tensor.dim(); dim++) {
            max += (tensor.size(dim) - 1) * tensor.stride(dim);
        }
        return max - min + 1 == tensor.numel();
    }

    private static Value[] detach(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor output = Tensor.wrap(new TensorImpl(
                input.impl().storage(),
                input.storageOffset(),
                input.shape(),
                input.strides(),
                input.versionCounter(),
                false,
                true));
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] to(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        DType dtype = args[1].asDType();
        boolean requiresGrad = outputRequiresGrad(op, args) && dtype.isFloatingPoint();
        if (input.dtype() == dtype) {
            if (requiresGrad == input.requiresGrad()) {
                return new Value[] {Value.tensor(input)};
            }
            return new Value[] {Value.tensor(Tensor.wrap(new TensorImpl(
                    input.impl().storage(),
                    input.storageOffset(),
                    input.shape(),
                    input.strides(),
                    input.versionCounter(),
                    requiresGrad,
                    true)))};
        }
        boolean preserveStrides = input.isContiguous() || ShapeRules.isDense(input.shape(), input.strides());
        Tensor output = Views.empty(
                dtype,
                input.shape(),
                preserveStrides ? input.strides() : Strides.contiguous(input.shape()),
                requiresGrad);
        Copy.copy(input, output);
        return new Value[] {Value.tensor(output)};
    }

    // ------------------------------------------------------------------ elementwise

    private static Value[] binary(Op op, Value[] args, DispatchKeySet keys) {
        Tensor left = args[0].asTensor();
        Tensor right = args[1].asTensor();
        Shape shape = ShapeRules.broadcastShape(left, right);
        DType dtype;
        if (op.name().equals("aten::div.Tensor")) {
            dtype = ShapeRules.divisionDtype(left, right);
        } else if (op.name().equals("aten::pow.Tensor_Tensor")) {
            dtype = ShapeRules.binaryDtype(left, right);
        } else {
            dtype = ShapeRules.binaryDtype(left, right);
        }
        Tensor output = fresh(shape, dtype, outputRequiresGrad(op, args));
        double alpha = 1;
        if (op.name().equals("aten::add.Tensor") || op.name().equals("aten::sub.Tensor")) {
            alpha = args[2].asScalar().toDouble();
        }
        final double alphaValue = alpha;
        final Scalar alphaScalar = op.name().equals("aten::add.Tensor") || op.name().equals("aten::sub.Tensor")
                ? args[2].asScalar()
                : Scalar.of(1);
        if ((dtype.isIntegral() || dtype.isBoolean()) && alphaScalar.isFloatingPoint()) {
            throw new DTypeException("For integral input tensors, argument alpha must not be a floating point number");
        }
        if (dtype.isIntegral() || dtype.isBoolean()) {
            final long alphaLong = alphaScalar.toLong();
            final boolean exactAlpha = alphaScalar.isIntegral() || alphaValue == Math.rint(alphaValue);
            Iteration.forEach(output, new Tensor[] {left, right}, (outOffset, in) -> {
                long a = TensorAccess.getLong(left.impl(), in[0]);
                long b = TensorAccess.getLong(right.impl(), in[1]);
                long result = switch (op.name()) {
                    case "aten::add.Tensor" -> exactAlpha
                            ? a + alphaLong * b
                            : (long) (a + alphaValue * (double) b);
                    case "aten::sub.Tensor" -> exactAlpha
                            ? a - alphaLong * b
                            : (long) (a - alphaValue * (double) b);
                    case "aten::mul.Tensor" -> a * b;
                    case "aten::pow.Tensor_Tensor" -> {
                        if (b < 0) {
                            throw new DTypeException("Integers to negative integer powers are not allowed");
                        }
                        yield integerPower(a, b);
                    }
                    case "aten::maximum.default" -> Math.max(a, b);
                    case "aten::minimum.default" -> Math.min(a, b);
                    default -> throw new IllegalStateException("Unexpected integral binary op " + op);
                };
                TensorAccess.setLong(output.impl(), outOffset, result);
            });
        } else {
            Iteration.forEach(output, new Tensor[] {left, right}, (outOffset, in) -> {
                double a = TensorAccess.getDouble(left.impl(), in[0]);
                double b = TensorAccess.getDouble(right.impl(), in[1]);
                double value = switch (op.name()) {
                    case "aten::add.Tensor" -> a + alphaValue * b;
                    case "aten::sub.Tensor" -> a - alphaValue * b;
                    case "aten::mul.Tensor" -> a * b;
                    case "aten::div.Tensor" -> a / b;
                    case "aten::pow.Tensor_Tensor" -> Math.pow(a, b);
                    case "aten::maximum.default" -> Math.max(a, b);
                    case "aten::minimum.default" -> Math.min(a, b);
                    default -> throw new IllegalStateException("Unexpected binary op " + op);
                };
                TensorAccess.setDouble(output.impl(), outOffset, value);
            });
        }
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] powScalar(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Scalar exponent = args[1].asScalar();
        DType dtype = ShapeRules.scalarDtype(input, exponent);
        if (!dtype.isFloatingPoint() && exponent.isFloatingPoint()) {
            dtype = DType.defaultFloat();
        }
        if (!input.dtype().isFloatingPoint() && exponent.isFloatingPoint()) {
            dtype = DType.defaultFloat();
        }
        if (!dtype.isFloatingPoint() && exponent.toDouble() < 0) {
            throw new DTypeException("Integers to negative integer powers are not allowed");
        }
        Tensor output = fresh(input.shape(), dtype, outputRequiresGrad(op, args));
        if (!dtype.isFloatingPoint()) {
            final long exponentLong = exponent.toLong();
            Iteration.forEach(output, new Tensor[] {input}, (outOffset, in) -> TensorAccess.setLong(
                    output.impl(), outOffset, integerPower(TensorAccess.getLong(input.impl(), in[0]), exponentLong)));
            return new Value[] {Value.tensor(output)};
        }
        double exponentValue = exponent.toDouble();
        Iteration.forEach(output, new Tensor[] {input}, (outOffset, in) -> TensorAccess.setDouble(
                output.impl(), outOffset, Math.pow(TensorAccess.getDouble(input.impl(), in[0]), exponentValue)));
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] comparison(Op op, Value[] args, DispatchKeySet keys) {
        Tensor left = args[0].asTensor();
        Tensor right = args[1].asTensor();
        Shape shape = ShapeRules.broadcastShape(left, right);
        Tensor output = fresh(shape, DType.BOOL, false);
        boolean integral = left.dtype().isIntegral() && right.dtype().isIntegral();
        Iteration.forEach(output, new Tensor[] {left, right}, (outOffset, in) -> {
            boolean value;
            if (integral) {
                long a = TensorAccess.getLong(left.impl(), in[0]);
                long b = TensorAccess.getLong(right.impl(), in[1]);
                value = switch (op.name()) {
                    case "aten::eq.Tensor" -> a == b;
                    case "aten::ne.Tensor" -> a != b;
                    case "aten::lt.Tensor" -> a < b;
                    case "aten::le.Tensor" -> a <= b;
                    case "aten::gt.Tensor" -> a > b;
                    case "aten::ge.Tensor" -> a >= b;
                    default -> throw new IllegalStateException("Unexpected comparison " + op);
                };
            } else {
                double a = TensorAccess.getDouble(left.impl(), in[0]);
                double b = TensorAccess.getDouble(right.impl(), in[1]);
                value = switch (op.name()) {
                    case "aten::eq.Tensor" -> a == b;
                    case "aten::ne.Tensor" -> a != b;
                    case "aten::lt.Tensor" -> a < b;
                    case "aten::le.Tensor" -> a <= b;
                    case "aten::gt.Tensor" -> a > b;
                    case "aten::ge.Tensor" -> a >= b;
                    default -> throw new IllegalStateException("Unexpected comparison " + op);
                };
            }
            TensorAccess.setBoolean(output.impl(), outOffset, value);
        });
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] logicalNot(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor output = fresh(input.shape(), DType.BOOL, false);
        Iteration.forEach(output, new Tensor[] {input}, (outOffset, in) ->
                TensorAccess.setBoolean(output.impl(), outOffset, !TensorAccess.getBoolean(input.impl(), in[0])));
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] unary(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        DType dtype = switch (op.name()) {
            case "aten::sqrt.default", "aten::exp.default", "aten::log.default", "aten::tanh.default",
                    "aten::sigmoid.default", "aten::reciprocal.default", "aten::sin.default", "aten::cos.default",
                    "aten::log1p.default" -> ShapeRules.floatMathDtype(input.dtype());
            case "aten::square.default" -> ShapeRules.integralPromotion(input.dtype());
            default -> input.dtype();
        };
        if (input.dtype().isBoolean() && switch (op.name()) {
            case "aten::neg.default", "aten::abs.default", "aten::relu.default", "aten::floor.default",
                    "aten::ceil.default" -> true;
            default -> false;
        }) {
            throw new DTypeException(op.name() + " is not implemented for bool tensors");
        }
        Tensor output = fresh(input.shape(), dtype, outputRequiresGrad(op, args));
        boolean longPath = (dtype.isIntegral() || dtype.isBoolean())
                && switch (op.name()) {
                    case "aten::neg.default", "aten::abs.default", "aten::square.default", "aten::relu.default",
                            "aten::floor.default", "aten::ceil.default", "aten::sign.default" -> true;
                    default -> false;
                };
        Iteration.forEach(output, new Tensor[] {input}, (outOffset, in) -> {
            if (longPath) {
                long value = TensorAccess.getLong(input.impl(), in[0]);
                long result = switch (op.name()) {
                    case "aten::neg.default" -> -value;
                    case "aten::abs.default" -> Math.abs(value);
                    case "aten::square.default" -> value * value;
                    case "aten::relu.default" -> Math.max(value, 0);
                    case "aten::sign.default" -> Long.compare(value, 0);
                    default -> value;
                };
                TensorAccess.setLong(output.impl(), outOffset, result);
                return;
            }
            double value = TensorAccess.getDouble(input.impl(), in[0]);
            double result = switch (op.name()) {
                case "aten::neg.default" -> -value;
                case "aten::abs.default" -> Math.abs(value);
                case "aten::square.default" -> value * value;
                case "aten::sqrt.default" -> Math.sqrt(value);
                case "aten::exp.default" -> Math.exp(value);
                case "aten::log.default" -> Math.log(value);
                case "aten::tanh.default" -> Math.tanh(value);
                case "aten::sigmoid.default" -> 1.0 / (1.0 + Math.exp(-value));
                case "aten::relu.default" -> Math.max(value, 0);
                case "aten::reciprocal.default" -> 1.0 / value;
                case "aten::floor.default" -> Math.floor(value);
                case "aten::ceil.default" -> Math.ceil(value);
                case "aten::sign.default" -> Double.isNaN(value) ? 0 : Math.signum(value);
                case "aten::sin.default" -> Math.sin(value);
                case "aten::cos.default" -> Math.cos(value);
                case "aten::log1p.default" -> Math.log1p(value);
                default -> throw new IllegalStateException("Unexpected unary op " + op);
            };
            TensorAccess.setDouble(output.impl(), outOffset, result);
        });
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] clamp(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Double min = args[1].isNone() ? null : args[1].asScalar().toDouble();
        Double max = args[2].isNone() ? null : args[2].asScalar().toDouble();
        if (min != null && max != null && min > max) {
            throw new IllegalArgumentException("clamp: min (" + min + ") must not exceed max (" + max + ")");
        }
        Tensor output = fresh(input.shape(), input.dtype(), outputRequiresGrad(op, args));
        if (input.dtype().isIntegral() || input.dtype().isBoolean()) {
            Long minLong = min == null ? null : (long) Math.ceil(min);
            Long maxLong = max == null ? null : (long) Math.floor(max);
            Iteration.forEach(output, new Tensor[] {input}, (outOffset, in) -> {
                long value = TensorAccess.getLong(input.impl(), in[0]);
                if (minLong != null) {
                    value = Math.max(value, minLong);
                }
                if (maxLong != null) {
                    value = Math.min(value, maxLong);
                }
                TensorAccess.setLong(output.impl(), outOffset, value);
            });
        } else {
            Iteration.forEach(output, new Tensor[] {input}, (outOffset, in) -> {
                double value = TensorAccess.getDouble(input.impl(), in[0]);
                if (min != null) {
                    value = Math.max(value, min);
                }
                if (max != null) {
                    value = Math.min(value, max);
                }
                TensorAccess.setDouble(output.impl(), outOffset, value);
            });
        }
        return new Value[] {Value.tensor(output)};
    }

    // ------------------------------------------------------------------ reductions

    private static Value[] reduce(Op op, Value[] args, boolean dims, boolean mean) {
        Tensor input = args[0].asTensor();
        if (op.name().startsWith("aten::prod") && input.requiresGrad() && GradMode.isEnabled()) {
            throw new org.jtorch.core.NotSupportedException(
                    "prod is forward-only in this build; run it under GradMode.noGrad or use a supported reduction");
        }
        boolean keepdim;
        int[] normalized;
        if (input.dim() == 0 && dims) {
            long[] rawDims = op.name().equals("aten::prod.dim")
                    ? new long[] {args[1].asLong()}
                    : args[1].asLongArray();
            if (!(rawDims.length == 0
                    || (rawDims.length == 1 && (rawDims[0] == 0 || rawDims[0] == -1)))) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            boolean meanReduction = op.name().startsWith("aten::mean");
            if (meanReduction && !input.dtype().isFloatingPoint()) {
                throw new DTypeException("mean(): could not infer output dtype from " + input.dtype()
                        + "; input must be floating point");
            }
            DType dtype = meanReduction ? input.dtype() : ShapeRules.integralPromotion(input.dtype());
            Tensor scalar = fresh(Shape.scalar(), dtype, outputRequiresGrad(op, args));
            if (dtype.isFloatingPoint()) {
                TensorAccess.setDouble(scalar.impl(), 0,
                        TensorAccess.getDouble(input.impl(), input.storageOffset()));
            } else {
                TensorAccess.setLong(scalar.impl(), 0, TensorAccess.getLong(input.impl(), input.storageOffset()));
            }
            return new Value[] {Value.tensor(scalar)};
        }
        if (dims) {
            long[] rawDims = op.name().equals("aten::prod.dim")
                    ? new long[] {args[1].asLong()}
                    : args[1].asLongArray();
            normalized = rawDims.length == 0
                    ? MetaKernels.allDims(input.shape())
                    : ShapeRules.normalizeDims(input.shape(), rawDims);
            keepdim = args[2].asBoolean();
        } else {
            normalized = new int[input.dim()];
            for (int i = 0; i < input.dim(); i++) {
                normalized[i] = i;
            }
            keepdim = false;
        }
        Shape keepShape = ShapeRules.keepDimShape(input.shape(), normalized);
        Shape outputShape = ShapeRules.reductionShape(input.shape(), normalized, keepdim);
        if (op.name().equals("aten::mean.default") || op.name().equals("aten::mean.dim")) {
            if (!input.dtype().isFloatingPoint()) {
                throw new DTypeException("mean(): could not infer output dtype from " + input.dtype()
                        + "; input must be floating point");
            }
        }
        DType dtype = switch (op.name()) {
            case "aten::mean.default", "aten::mean.dim" -> ShapeRules.floatMathDtype(input.dtype());
            case "aten::max.default", "aten::min.default" -> input.dtype();
            default -> ShapeRules.integralPromotion(input.dtype());
        };
        Tensor keepOutput = fresh(keepShape, dtype, outputRequiresGrad(op, args));
        long outNumel = keepShape.numel();
        if (outNumel > Integer.MAX_VALUE) {
            throw new ShapeException("Reduction output is too large for the reference backend: " + outNumel);
        }
        int outSize = (int) outNumel;
        long count = ShapeRules.reductionCount(input.shape(), normalized);
        boolean isMax = op.name().startsWith("aten::max");
        boolean isMin = op.name().startsWith("aten::min");
        boolean isProd = op.name().startsWith("aten::prod");
        if ((isMax || isMin) && count == 0) {
            throw new ShapeException(op.name() + "(): Expected reduction dim to have non-zero size");
        }

        boolean integral = !dtype.isFloatingPoint();
        long[] longAcc = integral ? new long[outSize] : null;
        double[] doubleAcc = integral ? null : new double[outSize];
        boolean[] booleanAcc = dtype.isBoolean() ? new boolean[outSize] : null;
        if (longAcc != null) {
            long identity = isProd ? 1 : (isMax ? Long.MIN_VALUE : isMin ? Long.MAX_VALUE : 0);
            java.util.Arrays.fill(longAcc, identity);
        }
        if (doubleAcc != null) {
            double identity = isProd ? 1 : (isMax ? Double.NEGATIVE_INFINITY : isMin ? Double.POSITIVE_INFINITY : 0);
            java.util.Arrays.fill(doubleAcc, identity);
        }
        if (booleanAcc != null) {
            java.util.Arrays.fill(booleanAcc, isProd);
        }

        long numel = input.numel();
        boolean[] reduced = new boolean[input.dim()];
        for (int dim : normalized) {
            reduced[dim] = true;
        }
        long[] outStrides = keepOutput.strides().toArray();
        long[] index = new long[input.dim()];
        for (long linear = 0; linear < numel; linear++) {
            long inOffset = input.elementOffset(linear);
            decompose(linear, input.shape(), index);
            long outOffset = 0;
            for (int d = 0; d < index.length; d++) {
                if (!reduced[d]) {
                    outOffset += index[d] * outStrides[d];
                }
            }
            int outIndex = (int) outOffset;
            double value = TensorAccess.getDouble(input.impl(), inOffset);
            if (longAcc != null) {
                long longValue = TensorAccess.getLong(input.impl(), inOffset);
                if (isProd) {
                    longAcc[outIndex] *= longValue;
                } else if (isMax) {
                    if (longValue > longAcc[outIndex]) {
                        longAcc[outIndex] = longValue;
                    }
                } else if (isMin) {
                    if (longValue < longAcc[outIndex]) {
                        longAcc[outIndex] = longValue;
                    }
                } else {
                    longAcc[outIndex] += longValue;
                }
            } else if (booleanAcc != null) {
                boolean booleanValue = value != 0;
                booleanAcc[outIndex] = (isProd || isMin)
                        ? booleanAcc[outIndex] && booleanValue
                        : booleanAcc[outIndex] || booleanValue;
            } else {
                if (isProd) {
                    doubleAcc[outIndex] *= value;
                } else if (isMax) {
                    if (value > doubleAcc[outIndex] || (Double.isNaN(value) && !Double.isNaN(doubleAcc[outIndex]))) {
                        doubleAcc[outIndex] = value;
                    }
                } else if (isMin) {
                    if (value < doubleAcc[outIndex] || (Double.isNaN(value) && !Double.isNaN(doubleAcc[outIndex]))) {
                        doubleAcc[outIndex] = value;
                    }
                } else {
                    doubleAcc[outIndex] += value;
                }
            }
        }

        for (int i = 0; i < outSize; i++) {
            if (longAcc != null) {
                TensorAccess.setLong(keepOutput.impl(), i, longAcc[i]);
            } else if (booleanAcc != null) {
                TensorAccess.setBoolean(keepOutput.impl(), i, booleanAcc[i]);
            } else {
                double value = doubleAcc[i];
                if (mean) {
                    value = count == 0 ? Double.NaN : value / count;
                }
                TensorAccess.setDouble(keepOutput.impl(), i, value);
            }
        }
        Tensor result = keepdim ? keepOutput : Views.contiguousReshape(keepOutput, outputShape);
        return new Value[] {Value.tensor(result)};
    }

    private static long offset(Tensor tensor, long... indices) {
        long offset = tensor.storageOffset();
        for (int dim = 0; dim < indices.length; dim++) {
            offset += indices[dim] * tensor.stride(dim);
        }
        return offset;
    }

    private static void decompose(long linear, Shape shape, long[] index) {
        long remaining = linear;
        for (int d = shape.rank() - 1; d >= 0; d--) {
            long size = shape.dim(d);
            index[d] = size == 0 ? 0 : remaining % size;
            remaining = size == 0 ? 0 : remaining / size;
        }
    }

    private static Value[] maxDim(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        boolean keepdim = args[2].asBoolean();
        if (input.dim() == 0) {
            int requested = args[1].asInt();
            if (requested != 0 && requested != -1) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            Shape shape = keepdim ? Shape.of(1) : Shape.scalar();
            Tensor values = fresh(shape, input.dtype(), outputRequiresGrad(op, args));
            Tensor indices = fresh(shape, DType.INT64, false);
            TensorAccess.setDouble(values.impl(), 0, TensorAccess.getDouble(input.impl(), input.storageOffset()));
            TensorAccess.setLong(indices.impl(), 0, 0);
            return new Value[] {Value.tensor(values), Value.tensor(indices)};
        }
        int dim = input.shape().normalizeDim(args[1].asInt());
        boolean max = op.name().startsWith("aten::max");
        Shape keepShape = ShapeRules.keepDimShape(input.shape(), new int[] {dim});
        Shape outputShape = ShapeRules.reductionShape(input.shape(), new int[] {dim}, keepdim);
        if (input.size(dim) == 0) {
            throw new ShapeException(op.name() + "(): Expected reduction dim to have non-zero size");
        }
        Tensor keepValues = fresh(keepShape, input.dtype(), outputRequiresGrad(op, args));
        Tensor keepIndices = fresh(keepShape, DType.INT64, false);
        int outSize = (int) keepShape.numel();
        boolean integral = input.dtype().isIntegral() || input.dtype().isBoolean();
        double[] best = integral ? null : new double[outSize];
        long[] bestLong = integral ? new long[outSize] : null;
        long[] bestIndex = new long[outSize];
        if (integral) {
            java.util.Arrays.fill(bestLong, max ? Long.MIN_VALUE : Long.MAX_VALUE);
        } else {
            java.util.Arrays.fill(best, max ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY);
        }

        long[] outStrides = keepValues.strides().toArray();
        long[] index = new long[input.dim()];
        for (long linear = 0; linear < input.numel(); linear++) {
            decompose(linear, input.shape(), index);
            long outOffset = 0;
            for (int d = 0; d < index.length; d++) {
                if (d != dim) {
                    outOffset += index[d] * outStrides[d];
                }
            }
            int outIndex = (int) outOffset;
            long inOffset = input.elementOffset(linear);
            boolean better;
            if (integral) {
                long value = TensorAccess.getLong(input.impl(), inOffset);
                better = max ? value > bestLong[outIndex] : value < bestLong[outIndex];
                if (better) {
                    bestLong[outIndex] = value;
                }
            } else {
                double value = TensorAccess.getDouble(input.impl(), inOffset);
                better = max
                        ? (value > best[outIndex]
                                || (Double.isNaN(value) && !Double.isNaN(best[outIndex])))
                        : (value < best[outIndex]
                                || (Double.isNaN(value) && !Double.isNaN(best[outIndex])));
                if (better) {
                    best[outIndex] = value;
                }
            }
            if (better) {
                bestIndex[outIndex] = index[dim];
            }
        }
        for (int i = 0; i < outSize; i++) {
            if (integral) {
                TensorAccess.setLong(keepValues.impl(), i, bestLong[i]);
            } else {
                TensorAccess.setDouble(keepValues.impl(), i, best[i]);
            }
            TensorAccess.setLong(keepIndices.impl(), i, bestIndex[i]);
        }
        Tensor values = keepdim ? keepValues : Views.contiguousReshape(keepValues, outputShape);
        Tensor indices = keepdim ? keepIndices : Views.contiguousReshape(keepIndices, outputShape);
        return new Value[] {Value.tensor(values), Value.tensor(indices)};
    }

    private static Value[] argReduce(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        if (input.dtype().isBoolean()) {
            throw new DTypeException(op.name() + "(): does not support bool input");
        }
        Integer requestedDim = args[1].asOptionalInt();
        boolean keepdim = args[2].asBoolean();
        boolean max = op.name().startsWith("aten::argmax");
        if (input.numel() == 0) {
            throw new ShapeException(op.name() + "(): cannot reduce an empty tensor");
        }
        if (input.dim() == 0) {
            if (requestedDim != null && requestedDim != 0 && requestedDim != -1) {
                throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
            }
            Shape shape = requestedDim != null && keepdim ? Shape.of(1) : Shape.scalar();
            Tensor output = fresh(shape, DType.INT64, false);
            TensorAccess.setLong(output.impl(), 0, 0);
            return new Value[] {Value.tensor(output)};
        }
        if (requestedDim == null) {
            boolean integral = input.dtype().isIntegral() || input.dtype().isBoolean();
            double best = integral ? 0 : (max ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY);
            long bestLong = integral ? (max ? Long.MIN_VALUE : Long.MAX_VALUE) : 0;
            long bestIndex = 0;
            for (long linear = 0; linear < input.numel(); linear++) {
                long offset = input.elementOffset(linear);
                boolean better;
                if (integral) {
                    long value = TensorAccess.getLong(input.impl(), offset);
                    better = max ? value > bestLong : value < bestLong;
                    if (better) {
                        bestLong = value;
                    }
                } else {
                    double value = TensorAccess.getDouble(input.impl(), offset);
                    better = max
                            ? (value > best || (Double.isNaN(value) && !Double.isNaN(best)))
                            : (value < best || (Double.isNaN(value) && !Double.isNaN(best)));
                    if (better) {
                        best = value;
                    }
                }
                if (better) {
                    bestIndex = linear;
                }
            }
            Tensor output = fresh(Shape.scalar(), DType.INT64, false);
            TensorAccess.setLong(output.impl(), 0, bestIndex);
            return new Value[] {Value.tensor(output)};
        }
        int dim = input.shape().normalizeDim(requestedDim);
        if (input.size(dim) == 0) {
            throw new ShapeException(op.name() + "(): Expected reduction dim to have non-zero size");
        }
        Shape keepShape = ShapeRules.keepDimShape(input.shape(), new int[] {dim});
        Shape outputShape = ShapeRules.reductionShape(input.shape(), new int[] {dim}, keepdim);
        Tensor keep = fresh(keepShape, DType.INT64, false);
        int outSize = (int) keepShape.numel();
        boolean integral = input.dtype().isIntegral() || input.dtype().isBoolean();
        double[] best = integral ? null : new double[outSize];
        long[] bestLong = integral ? new long[outSize] : null;
        long[] bestIndex = new long[outSize];
        if (integral) {
            java.util.Arrays.fill(bestLong, max ? Long.MIN_VALUE : Long.MAX_VALUE);
        } else {
            java.util.Arrays.fill(best, max ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY);
        }
        long[] outStrides = keep.strides().toArray();
        long[] index = new long[input.dim()];
        for (long linear = 0; linear < input.numel(); linear++) {
            decompose(linear, input.shape(), index);
            long outOffset = 0;
            for (int d = 0; d < index.length; d++) {
                if (d != dim) {
                    outOffset += index[d] * outStrides[d];
                }
            }
            int outIndex = (int) outOffset;
            long inOffset = input.elementOffset(linear);
            boolean better;
            if (integral) {
                long value = TensorAccess.getLong(input.impl(), inOffset);
                better = max ? value > bestLong[outIndex] : value < bestLong[outIndex];
                if (better) {
                    bestLong[outIndex] = value;
                }
            } else {
                double value = TensorAccess.getDouble(input.impl(), inOffset);
                better = max
                        ? (value > best[outIndex] || (Double.isNaN(value) && !Double.isNaN(best[outIndex])))
                        : (value < best[outIndex] || (Double.isNaN(value) && !Double.isNaN(best[outIndex])));
                if (better) {
                    best[outIndex] = value;
                }
            }
            if (better) {
                bestIndex[outIndex] = index[dim];
            }
        }
        for (int i = 0; i < outSize; i++) {
            TensorAccess.setLong(keep.impl(), i, bestIndex[i]);
        }
        Tensor result = keepdim ? keep : Views.contiguousReshape(keep, outputShape);
        return new Value[] {Value.tensor(result)};
    }

    // ------------------------------------------------------------------ matmul

    private static Value[] matmul(Op op, Value[] args, DispatchKeySet keys) {
        Tensor left = args[0].asTensor();
        Tensor right = args[1].asTensor();
        if (left.dtype().isBoolean() || right.dtype().isBoolean()) {
            throw new DTypeException("matmul is not implemented for bool tensors");
        }
        if (op.name().equals("aten::mm.default") && (left.dim() != 2 || right.dim() != 2)) {
            throw new ShapeException("mm expects two 2-D tensors, got " + left.shape() + " and " + right.shape());
        }
        if (op.name().equals("aten::bmm.default")) {
            if (left.dim() != 3 || right.dim() != 3) {
                throw new ShapeException("bmm expects two 3-D tensors, got " + left.shape() + " and " + right.shape());
            }
            if (left.size(0) != right.size(0)) {
                throw new ShapeException("bmm expects equal batch sizes, got " + left.shape() + " and " + right.shape());
            }
        }
        Shape outputShape = ShapeRules.matmulShape(left.shape(), right.shape());
        DType dtype = ShapeRules.binaryDtype(left, right);
        Tensor output = fresh(outputShape, dtype, outputRequiresGrad(op, args));
        computeMatmul(left, right, output);
        return new Value[] {Value.tensor(output)};
    }

    private static void computeMatmul(Tensor left, Tensor right, Tensor output) {
        int leftRank = left.dim();
        int rightRank = right.dim();
        long m = leftRank == 1 ? 1 : left.size(leftRank - 2);
        long k = left.size(leftRank - 1);
        long n = rightRank == 1 ? 1 : right.size(rightRank - 1);
        Shape leftBatch = leftRank <= 2 ? Shape.scalar() : prefixShape(left.shape(), leftRank - 2);
        Shape rightBatch = rightRank <= 2 ? Shape.scalar() : prefixShape(right.shape(), rightRank - 2);
        Shape batch = Shape.broadcast(leftBatch, rightBatch);
        int batchRank = batch.rank();
        long batchSize = batch.numel();

        long leftStrideM = leftRank == 1 ? 0 : left.stride(leftRank - 2);
        long leftStrideK = left.stride(leftRank - 1);
        long rightStrideK = rightRank == 1 ? right.stride(0) : right.stride(rightRank - 2);
        long rightStrideN = rightRank == 1 ? 0 : right.stride(rightRank - 1);

        long[] leftBatchStrides = batchStrides(left, batch);
        long[] rightBatchStrides = batchStrides(right, batch);
        long[] outputStrides = output.strides().toArray();
        int outputRank = output.dim();
        long[] index = new long[batchRank];
        boolean integral = output.dtype().isIntegral();

        for (long batchLinear = 0; batchLinear < batchSize; batchLinear++) {
            decompose(batchLinear, batch, index);
            long leftBatchOffset = left.storageOffset();
            long rightBatchOffset = right.storageOffset();
            long outputBatchOffset = output.storageOffset();
            for (int d = 0; d < batchRank; d++) {
                leftBatchOffset += index[d] * leftBatchStrides[d];
                rightBatchOffset += index[d] * rightBatchStrides[d];
                outputBatchOffset += index[d] * outputStrides[d];
            }
            int mDim = leftRank == 1 ? -1 : batchRank;
            int nDim = rightRank == 1 ? -1 : (leftRank == 1 ? batchRank : batchRank + 1);
            long outputStrideM = mDim >= 0 ? outputStrides[mDim] : 0;
            long outputStrideN = nDim >= 0 && nDim < outputRank ? outputStrides[nDim] : 0;
            for (long i = 0; i < m; i++) {
                for (long j = 0; j < n; j++) {
                    if (integral) {
                        long sum = 0;
                        for (long p = 0; p < k; p++) {
                            long a = TensorAccess.getLong(
                                    left.impl(), leftBatchOffset + i * leftStrideM + p * leftStrideK);
                            long b = TensorAccess.getLong(
                                    right.impl(), rightBatchOffset + p * rightStrideK + j * rightStrideN);
                            sum += a * b;
                        }
                        TensorAccess.setLong(
                                output.impl(), outputBatchOffset + i * outputStrideM + j * outputStrideN, sum);
                    } else if (output.dtype() == DType.FLOAT32) {
                        float sum = 0f;
                        for (long p = 0; p < k; p++) {
                            float a = TensorAccess.getFloat(
                                    left.impl(), leftBatchOffset + i * leftStrideM + p * leftStrideK);
                            float b = TensorAccess.getFloat(
                                    right.impl(), rightBatchOffset + p * rightStrideK + j * rightStrideN);
                            sum += a * b;
                        }
                        TensorAccess.setFloat(
                                output.impl(), outputBatchOffset + i * outputStrideM + j * outputStrideN, sum);
                    } else {
                        double sum = 0;
                        for (long p = 0; p < k; p++) {
                            double a = TensorAccess.getDouble(
                                    left.impl(), leftBatchOffset + i * leftStrideM + p * leftStrideK);
                            double b = TensorAccess.getDouble(
                                    right.impl(), rightBatchOffset + p * rightStrideK + j * rightStrideN);
                            sum += a * b;
                        }
                        TensorAccess.setDouble(
                                output.impl(), outputBatchOffset + i * outputStrideM + j * outputStrideN, sum);
                    }
                }
            }
        }
    }

    private static Shape prefixShape(Shape shape, int rank) {
        long[] dims = shape.toArray();
        long[] prefix = new long[rank];
        System.arraycopy(dims, 0, prefix, 0, rank);
        return Shape.of(prefix);
    }

    private static long[] batchStrides(Tensor tensor, Shape batch) {
        int batchRank = batch.rank();
        long[] strides = new long[batchRank];
        int tensorBatchRank = Math.max(0, tensor.dim() - 2);
        for (int d = 0; d < batchRank; d++) {
            int fromEnd = batchRank - 1 - d;
            int tensorDim = tensorBatchRank - 1 - fromEnd;
            if (tensorDim < 0) {
                strides[d] = 0;
            } else if (tensor.size(tensorDim) == 1 && batch.dim(d) != 1) {
                strides[d] = 0;
            } else {
                strides[d] = tensor.stride(tensorDim);
            }
        }
        return strides;
    }

    private static Value[] addmm(Op op, Value[] args, DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        Tensor mat1 = args[1].asTensor();
        Tensor mat2 = args[2].asTensor();
        double beta = args[3].asScalar().toDouble();
        double alpha = args[4].asScalar().toDouble();
        if (mat1.dim() != 2 || mat2.dim() != 2) {
            throw new ShapeException("addmm expects 2-D matrices, got " + mat1.shape() + " and " + mat2.shape());
        }
        Shape mmShape = ShapeRules.matmulShape(mat1.shape(), mat2.shape());
        Shape broadcast = Shape.broadcast(self.shape(), mmShape);
        if (!broadcast.equals(mmShape)) {
            throw new ShapeException("addmm: self shape " + self.shape() + " cannot broadcast to " + mmShape);
        }
        DType dtype = DType.resultType(ShapeRules.binaryDtype(self, mat1), mat2.dtype());
        Tensor output = fresh(mmShape, dtype, outputRequiresGrad(op, args));
        Tensor product = fresh(mmShape, dtype, false);
        computeMatmul(mat1, mat2, product);
        boolean exactScalars = beta == (long) beta && alpha == (long) alpha;
        if ((dtype.isIntegral() || dtype.isBoolean()) && exactScalars) {
            final long betaLong = (long) beta;
            final long alphaLong = (long) alpha;
            Iteration.forEach(output, new Tensor[] {self, product}, (outOffset, in) -> {
                long value = 0;
                if (betaLong != 0) {
                    value += betaLong * TensorAccess.getLong(self.impl(), in[0]);
                }
                if (alphaLong != 0) {
                    value += alphaLong * TensorAccess.getLong(product.impl(), in[1]);
                }
                TensorAccess.setLong(output.impl(), outOffset, value);
            });
        } else {
            Iteration.forEach(output, new Tensor[] {self, product}, (outOffset, in) -> {
                double value = 0;
                if (beta != 0) {
                    value += beta * TensorAccess.getDouble(self.impl(), in[0]);
                }
                if (alpha != 0) {
                    value += alpha * TensorAccess.getDouble(product.impl(), in[1]);
                }
                TensorAccess.setDouble(output.impl(), outOffset, value);
            });
        }
        return new Value[] {Value.tensor(output)};
    }

    // ------------------------------------------------------------------ combination

    private static Value[] cat(Op op, Value[] args, DispatchKeySet keys) {
        Tensor[] tensors = args[0].asTensorList();
        int dim = args[1].asInt();
        Shape shape = catShape(tensors, dim);
        DType dtype = tensors[0].dtype();
        for (Tensor tensor : tensors) {
            dtype = DType.resultType(dtype, tensor.dtype());
        }
        Tensor output = fresh(shape, dtype, outputRequiresGrad(op, args));
        int normalized = tensors[0].shape().normalizeDim(dim);
        long offset = 0;
        for (Tensor tensor : tensors) {
            long[] index = new long[output.dim()];
            long[] outStrides = output.strides().toArray();
            for (long linear = 0; linear < tensor.numel(); linear++) {
                decompose(linear, tensor.shape(), index);
                long outOffset = output.storageOffset();
                for (int d = 0; d < index.length; d++) {
                    long value = d == normalized ? index[d] + offset : index[d];
                    outOffset += value * outStrides[d];
                }
                copyValue(output, outOffset, tensor, tensor.elementOffset(linear));
            }
            offset += tensor.size(normalized);
        }
        return new Value[] {Value.tensor(output)};
    }

    private static Shape catShape(Tensor[] tensors, int dim) {
        if (tensors.length == 0) {
            throw new ShapeException("cat expects a non-empty tensor list");
        }
        int normalized = tensors[0].shape().normalizeDim(dim);
        long[] sizes = tensors[0].shape().toArray();
        long total = 0;
        for (Tensor tensor : tensors) {
            if (tensor.dim() != sizes.length) {
                throw new ShapeException("cat: all tensors must have the same rank");
            }
            for (int i = 0; i < sizes.length; i++) {
                if (i != normalized && tensor.size(i) != sizes[i]) {
                    throw new ShapeException("cat: sizes must match except in dimension " + normalized);
                }
            }
            total += tensor.size(normalized);
        }
        sizes[normalized] = total;
        return Shape.of(sizes);
    }

    private static Value[] stack(Op op, Value[] args, DispatchKeySet keys) {
        Tensor[] tensors = args[0].asTensorList();
        int dim = args[1].asInt();
        if (tensors.length == 0) {
            throw new ShapeException("stack expects a non-empty tensor list");
        }
        Shape base = tensors[0].shape();
        for (Tensor tensor : tensors) {
            if (!tensor.shape().equals(base)) {
                throw new ShapeException("stack: all tensors must have the same shape");
            }
        }
        int rank = base.rank();
        int normalized = dim < 0 ? dim + rank + 1 : dim;
        if (normalized < 0 || normalized > rank) {
            throw new IndexOutOfBoundsException("stack: dimension " + dim + " out of range");
        }
        long[] sizes = new long[rank + 1];
        for (int i = 0; i < normalized; i++) {
            sizes[i] = base.dim(i);
        }
        sizes[normalized] = tensors.length;
        for (int i = normalized; i < rank; i++) {
            sizes[i + 1] = base.dim(i);
        }
        Shape shape = Shape.of(sizes);
        DType dtype = tensors[0].dtype();
        for (Tensor tensor : tensors) {
            dtype = DType.resultType(dtype, tensor.dtype());
        }
        Tensor output = fresh(shape, dtype, outputRequiresGrad(op, args));
        long[] outStrides = output.strides().toArray();
        for (int t = 0; t < tensors.length; t++) {
            Tensor tensor = tensors[t];
            long[] index = new long[tensor.dim()];
            for (long linear = 0; linear < tensor.numel(); linear++) {
                decompose(linear, tensor.shape(), index);
                long outOffset = output.storageOffset() + t * outStrides[normalized];
                for (int d = 0; d < index.length; d++) {
                    int outDim = d < normalized ? d : d + 1;
                    outOffset += index[d] * outStrides[outDim];
                }
                copyValue(output, outOffset, tensor, tensor.elementOffset(linear));
            }
        }
        return new Value[] {Value.tensor(output)};
    }

    // ------------------------------------------------------------------ convolution and pooling

    private static Value[] conv2d(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor weight = args[1].asTensor();
        Tensor bias = args[2].asOptionalTensor();
        Convolution.ConvConfig config = Convolution.convConfig(
                args[3].asLongArray(), args[4].asLongArray(), args[5].asLongArray(), args[6].asInt());
        Convolution.validateConv(input, weight, bias, config);
        Shape shape = Convolution.convOutputShape(input, weight, config);
        Tensor output = fresh(shape, input.dtype(), outputRequiresGrad(op, args));
        int batch = (int) input.size(0);
        int outChannels = (int) weight.size(0);
        int inChannels = (int) input.size(1);
        int kernelH = (int) weight.size(2);
        int kernelW = (int) weight.size(3);
        int outH = (int) shape.dim(2);
        int outW = (int) shape.dim(3);
        long inputOffset = input.storageOffset();
        long weightOffset = weight.storageOffset();
        long strideN = input.stride(0);
        long strideC = input.stride(1);
        long strideH = input.stride(2);
        long strideW = input.stride(3);
        long weightStrideO = weight.stride(0);
        long weightStrideC = weight.stride(1);
        long weightStrideH = weight.stride(2);
        long weightStrideW = weight.stride(3);
        for (int n = 0; n < batch; n++) {
            for (int o = 0; o < outChannels; o++) {
                for (int y = 0; y < outH; y++) {
                    for (int x = 0; x < outW; x++) {
                        double sum = bias == null ? 0 : TensorAccess.getDouble(bias.impl(), o);
                        for (int c = 0; c < inChannels; c++) {
                            for (int ky = 0; ky < kernelH; ky++) {
                                int iy = y * config.stride()[0] - config.padding()[0] + ky * config.dilation()[0];
                                if (iy < 0 || iy >= input.size(2)) {
                                    continue;
                                }
                                for (int kx = 0; kx < kernelW; kx++) {
                                    int ix = x * config.stride()[1] - config.padding()[1]
                                            + kx * config.dilation()[1];
                                    if (ix < 0 || ix >= input.size(3)) {
                                        continue;
                                    }
                                    long in = inputOffset + n * strideN + c * strideC + iy * strideH + ix * strideW;
                                    long w = weightOffset + o * weightStrideO + c * weightStrideC
                                            + ky * weightStrideH + kx * weightStrideW;
                                    sum += TensorAccess.getDouble(input.impl(), in)
                                            * TensorAccess.getDouble(weight.impl(), w);
                                }
                            }
                        }
                        TensorAccess.setDouble(output.impl(), output.elementOffset(
                                ((long) n * outChannels + o) * outH * outW + (long) y * outW + x), sum);
                    }
                }
            }
        }
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] maxPool2d(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Convolution.PoolConfig config = Convolution.poolConfig(
                args[1].asLongArray(), args[2].asLongArray(), args[3].asLongArray(), args[4].asLongArray(),
                args[5].asBoolean());
        Convolution.validatePool(input, config);
        Shape shape = Convolution.poolOutputShape(input, config);
        Tensor output = fresh(shape, input.dtype(), outputRequiresGrad(op, args));
        int batch = (int) input.size(0);
        int channels = (int) input.size(1);
        int outH = (int) shape.dim(2);
        int outW = (int) shape.dim(3);
        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < outH; y++) {
                    for (int x = 0; x < outW; x++) {
                        double best = Double.NEGATIVE_INFINITY;
                        for (int ky = 0; ky < config.kernel()[0]; ky++) {
                            int iy = y * config.stride()[0] - config.padding()[0] + ky * config.dilation()[0];
                            if (iy < 0 || iy >= input.size(2)) {
                                continue;
                            }
                            for (int kx = 0; kx < config.kernel()[1]; kx++) {
                                int ix = x * config.stride()[1] - config.padding()[1]
                                        + kx * config.dilation()[1];
                                if (ix < 0 || ix >= input.size(3)) {
                                    continue;
                                }
                                double value = TensorAccess.getDouble(input.impl(),
                                        offset(input,n, c, iy, ix));
                                if (value > best || (Double.isNaN(value) && !Double.isNaN(best))) {
                                    best = value;
                                }
                            }
                        }
                        TensorAccess.setDouble(output.impl(),
                                offset(output,n, c, y, x), best);
                    }
                }
            }
        }
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] conv2dBackwardInput(Op op, Value[] args, DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor weight = args[1].asTensor();
        long[] inputShape = args[2].asLongArray();
        Convolution.ConvConfig config = Convolution.convConfig(
                args[3].asLongArray(), args[4].asLongArray(), args[5].asLongArray(), 1);
        Tensor output = fresh(Shape.of(inputShape), grad.dtype(), false);
        fillLongDouble(output, 0);
        int batch = (int) inputShape[0];
        int inChannels = (int) inputShape[1];
        int height = (int) inputShape[2];
        int width = (int) inputShape[3];
        int outChannels = (int) weight.size(0);
        int kernelH = (int) weight.size(2);
        int kernelW = (int) weight.size(3);
        int outH = (int) grad.size(2);
        int outW = (int) grad.size(3);
        for (int n = 0; n < batch; n++) {
            for (int o = 0; o < outChannels; o++) {
                for (int y = 0; y < outH; y++) {
                    for (int x = 0; x < outW; x++) {
                        double g = TensorAccess.getDouble(grad.impl(), offset(grad,n, o, y, x));
                        if (g == 0) {
                            continue;
                        }
                        for (int c = 0; c < inChannels; c++) {
                            for (int ky = 0; ky < kernelH; ky++) {
                                int iy = y * config.stride()[0] - config.padding()[0] + ky * config.dilation()[0];
                                if (iy < 0 || iy >= height) {
                                    continue;
                                }
                                for (int kx = 0; kx < kernelW; kx++) {
                                    int ix = x * config.stride()[1] - config.padding()[1]
                                            + kx * config.dilation()[1];
                                    if (ix < 0 || ix >= width) {
                                        continue;
                                    }
                                    double w = TensorAccess.getDouble(weight.impl(),
                                            offset(weight,o, c, ky, kx));
                                    long offset = offset(output,n, c, iy, ix);
                                    TensorAccess.setDouble(output.impl(), offset,
                                            TensorAccess.getDouble(output.impl(), offset) + g * w);
                                }
                            }
                        }
                    }
                }
            }
        }
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] conv2dBackwardWeight(Op op, Value[] args, DispatchKeySet keys) {
        Tensor input = args[0].asTensor();
        Tensor grad = args[1].asTensor();
        long[] weightShape = args[2].asLongArray();
        Convolution.ConvConfig config = Convolution.convConfig(
                args[3].asLongArray(), args[4].asLongArray(), args[5].asLongArray(), 1);
        Tensor output = fresh(Shape.of(weightShape), grad.dtype(), false);
        fillLongDouble(output, 0);
        int batch = (int) input.size(0);
        int inChannels = (int) input.size(1);
        int height = (int) input.size(2);
        int width = (int) input.size(3);
        int outChannels = (int) weightShape[0];
        int kernelH = (int) weightShape[2];
        int kernelW = (int) weightShape[3];
        int outH = (int) grad.size(2);
        int outW = (int) grad.size(3);
        for (int n = 0; n < batch; n++) {
            for (int o = 0; o < outChannels; o++) {
                for (int y = 0; y < outH; y++) {
                    for (int x = 0; x < outW; x++) {
                        double g = TensorAccess.getDouble(grad.impl(), offset(grad,n, o, y, x));
                        if (g == 0) {
                            continue;
                        }
                        for (int c = 0; c < inChannels; c++) {
                            for (int ky = 0; ky < kernelH; ky++) {
                                int iy = y * config.stride()[0] - config.padding()[0] + ky * config.dilation()[0];
                                if (iy < 0 || iy >= height) {
                                    continue;
                                }
                                for (int kx = 0; kx < kernelW; kx++) {
                                    int ix = x * config.stride()[1] - config.padding()[1]
                                            + kx * config.dilation()[1];
                                    if (ix < 0 || ix >= width) {
                                        continue;
                                    }
                                    long offset = offset(output,o, c, ky, kx);
                                    TensorAccess.setDouble(output.impl(), offset,
                                            TensorAccess.getDouble(output.impl(), offset)
                                                    + g * TensorAccess.getDouble(input.impl(),
                                                            offset(input,n, c, iy, ix)));
                                }
                            }
                        }
                    }
                }
            }
        }
        return new Value[] {Value.tensor(output)};
    }

    private static Value[] maxPool2dBackward(Op op, Value[] args, DispatchKeySet keys) {
        Tensor grad = args[0].asTensor();
        Tensor input = args[1].asTensor();
        Convolution.PoolConfig config = Convolution.poolConfig(
                args[2].asLongArray(), args[3].asLongArray(), args[4].asLongArray(), args[5].asLongArray(), false);
        Tensor output = fresh(input.shape(), grad.dtype(), false);
        fillLongDouble(output, 0);
        int batch = (int) input.size(0);
        int channels = (int) input.size(1);
        int outH = (int) grad.size(2);
        int outW = (int) grad.size(3);
        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < outH; y++) {
                    for (int x = 0; x < outW; x++) {
                        double best = Double.NEGATIVE_INFINITY;
                        int bestY = -1;
                        int bestX = -1;
                        for (int ky = 0; ky < config.kernel()[0]; ky++) {
                            int iy = y * config.stride()[0] - config.padding()[0] + ky * config.dilation()[0];
                            if (iy < 0 || iy >= input.size(2)) {
                                continue;
                            }
                            for (int kx = 0; kx < config.kernel()[1]; kx++) {
                                int ix = x * config.stride()[1] - config.padding()[1]
                                        + kx * config.dilation()[1];
                                if (ix < 0 || ix >= input.size(3)) {
                                    continue;
                                }
                                double value = TensorAccess.getDouble(input.impl(), offset(input,n, c, iy, ix));
                                if (value > best || (Double.isNaN(value) && !Double.isNaN(best))) {
                                    best = value;
                                    bestY = iy;
                                    bestX = ix;
                                }
                            }
                        }
                        if (bestY >= 0) {
                            long offset = offset(output,n, c, bestY, bestX);
                            TensorAccess.setDouble(output.impl(), offset,
                                    TensorAccess.getDouble(output.impl(), offset)
                                            + TensorAccess.getDouble(grad.impl(), offset(grad,n, c, y, x)));
                        }
                    }
                }
            }
        }
        return new Value[] {Value.tensor(output)};
    }

    private static void fillLongDouble(Tensor tensor, double value) {
        for (long i = 0; i < tensor.numel(); i++) {
            TensorAccess.setDouble(tensor.impl(), tensor.elementOffset(i), value);
        }
    }

    // ------------------------------------------------------------------ losses

    private static Value[] crossEntropy(Op op, Value[] args, DispatchKeySet keys) {
        Tensor logits = args[0].asTensor();
        Tensor target = args[1].asTensor();
        String reduction = args[2].asString();
        MetaKernels.validateCrossEntropy(logits, target, reduction);
        if (logits.dim() != 2) {
            throw new ShapeException("cross_entropy in this build supports 2-D logits only, got " + logits.shape());
        }
        if (logits.size(0) > Integer.MAX_VALUE || logits.size(1) > Integer.MAX_VALUE) {
            throw new ShapeException("cross_entropy dimensions exceed the reference backend limit");
        }
        int rows = (int) logits.size(0);
        int classes = (int) logits.size(1);
        boolean none = reduction.equals("none");
        Tensor output = fresh(none ? Shape.of(rows) : Shape.scalar(), logits.dtype(), outputRequiresGrad(op, args));
        double total = 0;
        for (int row = 0; row < rows; row++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                max = Math.max(max, TensorAccess.getDouble(logits.impl(), logits.elementOffset((long) row * classes + c)));
            }
            double sumExp = 0;
            for (int c = 0; c < classes; c++) {
                sumExp += Math.exp(TensorAccess.getDouble(logits.impl(), logits.elementOffset((long) row * classes + c)) - max);
            }
            long label = TensorAccess.getLong(target.impl(), target.elementOffset(row));
            if (label < 0 || label >= classes) {
                throw new IndexOutOfBoundsException(
                        "Target " + label + " is out of bounds for " + classes + " classes");
            }
            double logit = TensorAccess.getDouble(logits.impl(), logits.elementOffset((long) row * classes + label));
            double loss = Math.log(sumExp) + max - logit;
            if (none) {
                TensorAccess.setDouble(output.impl(), row, loss);
            } else {
                total += loss;
            }
        }
        if (!none) {
            double value = reduction.equals("mean") ? total / rows : total;
            TensorAccess.setDouble(output.impl(), 0, value);
        }
        return new Value[] {Value.tensor(output)};
    }

    // ------------------------------------------------------------------ in-place

    private static Value[] inPlaceBinary(Op op, Value[] args, DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        Tensor other = args[1].asTensor();
        ShapeRules.requireScaling(self, op.name());
        Shape broadcast = Shape.broadcast(self.shape(), other.shape());
        if (!broadcast.equals(self.shape())) {
            throw new ShapeException("in-place op cannot expand self from " + self.shape() + " to " + broadcast);
        }
        org.jtorch.core.DType resultDtype = op.name().equals("aten::div_.Tensor")
                ? ShapeRules.divisionDtype(self, other)
                : ShapeRules.binaryDtype(self, other);
        if (resultDtype != self.dtype()) {
            throw new DTypeException("result type " + resultDtype + " can't be cast to " + self.dtype()
                    + " for in-place operation " + op.name());
        }
        if (hasInternalOverlap(self)) {
            throw new org.jtorch.core.JTorchException("unsupported operation: more than one element of the "
                    + "written-to tensor refers to a single memory location");
        }
        if (self.impl() != other.impl() && self.impl().sharesStorage(other.impl())
                && !sameGeometry(self, other) && rangesOverlap(self, other)) {
            throw new org.jtorch.core.JTorchException("unsupported operation: some elements of the input tensor and "
                    + "the written-to tensor refer to a single memory location; clone the tensor first");
        }
        double alpha = 1;
        if (op.name().equals("aten::add_.Tensor") || op.name().equals("aten::sub_.Tensor")) {
            alpha = args[2].asScalar().toDouble();
        }
        final double alphaValue = alpha;
        final long alphaLong = (long) alpha;
        Iteration.forEach(self, new Tensor[] {self, other}, (outOffset, in) -> {
            if (self.dtype().isIntegral() || self.dtype().isBoolean()) {
                long a = TensorAccess.getLong(self.impl(), in[0]);
                long b = TensorAccess.getLong(other.impl(), in[1]);
                long result = switch (op.name()) {
                    case "aten::add_.Tensor" -> a + alphaLong * b;
                    case "aten::sub_.Tensor" -> a - alphaLong * b;
                    case "aten::mul_.Tensor" -> a * b;
                    default -> throw new IllegalStateException("Unexpected integral in-place op " + op);
                };
                TensorAccess.setLong(self.impl(), outOffset, result);
            } else {
                double a = TensorAccess.getDouble(self.impl(), in[0]);
                double b = TensorAccess.getDouble(other.impl(), in[1]);
                double value = switch (op.name()) {
                    case "aten::add_.Tensor" -> a + alphaValue * b;
                    case "aten::sub_.Tensor" -> a - alphaValue * b;
                    case "aten::mul_.Tensor" -> a * b;
                    case "aten::div_.Tensor" -> a / b;
                    default -> throw new IllegalStateException("Unexpected in-place op " + op);
                };
                TensorAccess.setDouble(self.impl(), outOffset, value);
            }
        });
        self.impl().bumpVersion();
        return new Value[] {Value.tensor(self)};
    }

    private static Value[] inPlaceCopy(Op op, Value[] args, DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        Tensor source = args[1].asTensor();
        ShapeRules.requireScaling(self, op.name());
        if (hasInternalOverlap(self)) {
            throw new org.jtorch.core.JTorchException("unsupported operation: more than one element of the "
                    + "written-to tensor refers to a single memory location");
        }
        if (self.impl() != source.impl() && self.impl().sharesStorage(source.impl())
                && !sameGeometry(self, source) && rangesOverlap(self, source)) {
            throw new org.jtorch.core.JTorchException("unsupported operation: some elements of the input tensor and "
                    + "the written-to tensor refer to a single memory location; clone the tensor first");
        }
        Copy.copyBroadcast(source, self);
        self.impl().bumpVersion();
        return new Value[] {Value.tensor(self)};
    }

    private static Value[] inPlaceFill(Op op, Value[] args, DispatchKeySet keys) {
        Tensor self = args[0].asTensor();
        ShapeRules.requireScaling(self, op.name());
        if (hasInternalOverlap(self)) {
            throw new org.jtorch.core.JTorchException("unsupported operation: more than one element of the "
                    + "written-to tensor refers to a single memory location");
        }
        Scalar scalar = op.name().equals("aten::fill_.Scalar") ? args[1].asScalar() : Scalar.of(0);
        boolean integral = self.dtype().isIntegral() || self.dtype().isBoolean();
        long longValue = scalar.toLong(self.dtype());
        double doubleValue = scalar.toDouble(self.dtype());
        Iteration.forEach(self, new Tensor[0], (outOffset, ignored) -> {
            if (integral) {
                TensorAccess.setLong(self.impl(), outOffset, longValue);
            } else {
                TensorAccess.setDouble(self.impl(), outOffset, doubleValue);
            }
        });
        self.impl().bumpVersion();
        return new Value[] {Value.tensor(self)};
    }

    private static void copyValue(Tensor output, long outputOffset, Tensor source, long sourceOffset) {
        if (output.dtype().isIntegral() || output.dtype().isBoolean()) {
            TensorAccess.setLong(output.impl(), outputOffset, TensorAccess.getLong(source.impl(), sourceOffset));
        } else {
            TensorAccess.setDouble(output.impl(), outputOffset, TensorAccess.getDouble(source.impl(), sourceOffset));
        }
    }

    private static boolean sameGeometry(Tensor left, Tensor right) {
        return left.shape().equals(right.shape()) && left.strides().equals(right.strides())
                && left.storageOffset() == right.storageOffset();
    }

    private static boolean hasInternalOverlap(Tensor tensor) {
        for (int dim = 0; dim < tensor.dim(); dim++) {
            if (tensor.size(dim) > 1 && tensor.stride(dim) == 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean rangesOverlap(Tensor left, Tensor right) {
        long[] leftRange = range(left);
        long[] rightRange = range(right);
        return leftRange != null && rightRange != null
                && leftRange[0] <= rightRange[1] && rightRange[0] <= leftRange[1];
    }

    private static long[] range(Tensor tensor) {
        if (tensor.numel() == 0) {
            return null;
        }
        long min = tensor.storageOffset();
        long max = min;
        for (int dim = 0; dim < tensor.dim(); dim++) {
            max += (tensor.size(dim) - 1) * tensor.stride(dim);
        }
        return new long[] {min, max};
    }

    private static long integerPower(long base, long exponent) {
        long result = 1;
        long factor = base;
        long remaining = exponent;
        while (remaining > 0) {
            if ((remaining & 1) != 0) {
                result *= factor;
            }
            factor *= factor;
            remaining >>= 1;
        }
        return result;
    }

}
