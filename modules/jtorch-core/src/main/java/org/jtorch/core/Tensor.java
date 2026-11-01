package org.jtorch.core;

import java.util.List;
import java.util.function.Consumer;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.memory.TensorAccess;
import org.jtorch.core.value.Value;

/**
 * A tensor handle: lightweight, non-{@code AutoCloseable}, and sharing storage with its views.
 *
 * <p>Every computational method goes through the dispatcher. Data extraction methods are metadata/read helpers and
 * never bypass operator semantics for computation.
 */
public class Tensor {
    final TensorImpl impl;

    protected Tensor(TensorImpl impl) {
        this.impl = impl;
    }

    public static Tensor wrap(TensorImpl impl) {
        return new Tensor(impl);
    }

    public TensorImpl impl() {
        return impl;
    }

    // ---------------------------------------------------------------- metadata

    public Shape shape() {
        return impl.sizes();
    }

    public Strides strides() {
        return impl.strides();
    }

    public DType dtype() {
        return impl.dtype();
    }

    public Device device() {
        return impl.device();
    }

    public Layout layout() {
        return impl.layout();
    }

    public long numel() {
        return impl.numel();
    }

    public int dim() {
        return impl.sizes().rank();
    }

    public long size(int dim) {
        return impl.sizes().dim(dim);
    }

    public long stride(int dim) {
        return impl.strides().stride(impl.sizes().normalizeDim(dim));
    }

    public long storageOffset() {
        return impl.storageOffset();
    }

    public boolean isContiguous() {
        return impl.isContiguous();
    }

    public boolean isView() {
        return impl.isView();
    }

    public boolean requiresGrad() {
        return impl.requiresGrad();
    }

    public boolean isLeaf() {
        return impl.isLeaf();
    }

    public Tensor requiresGrad(boolean value) {
        impl.setRequiresGrad(value);
        return this;
    }

    public Tensor grad() {
        return impl.grad();
    }

    public VersionCounter versionCounter() {
        return impl.version();
    }

    /** True when this tensor and the other share the same underlying storage object. */
    public boolean sharesStorageWith(Tensor other) {
        return impl.sharesStorage(other.impl);
    }

    public Tensor registerPostAccumulateHook(Consumer<Tensor> hook) {
        impl.addPostAccumulateHook(hook);
        return this;
    }

    // ---------------------------------------------------------------- factories used by graph code

    public static Tensor onesLike(Tensor tensor) {
        Shape shape = tensor.shape();
        long[] dims = shape.toArray();
        return Dispatcher.get()
                .call(OpRegistry.ONES, Value.integers(dims), Value.dtype(tensor.dtype()), Value.device(tensor.device()))[0]
                .asTensor();
    }

    public static Tensor zerosLike(Tensor tensor) {
        Shape shape = tensor.shape();
        long[] dims = shape.toArray();
        return Dispatcher.get()
                .call(OpRegistry.ZEROS, Value.integers(dims), Value.dtype(tensor.dtype()), Value.device(tensor.device()))[0]
                .asTensor();
    }

    // ---------------------------------------------------------------- views and shape

    public Tensor reshape(long... shape) {
        return first(OpRegistry.RESHAPE, Value.tensor(this), Value.integers(shape));
    }

    public Tensor view(long... shape) {
        return first(OpRegistry.VIEW, Value.tensor(this), Value.integers(shape));
    }

    public Tensor transpose(int dim0, int dim1) {
        return first(OpRegistry.TRANSPOSE, Value.tensor(this), Value.integer(dim0), Value.integer(dim1));
    }

    public Tensor permute(int... dims) {
        return first(OpRegistry.PERMUTE, Value.tensor(this), Value.integers(toLongs(dims)));
    }

    public Tensor narrow(int dim, long start, long length) {
        return first(
                OpRegistry.NARROW,
                Value.tensor(this),
                Value.integer(dim),
                Value.integer(start),
                Value.integer(length));
    }

    public Tensor select(int dim, long index) {
        return first(OpRegistry.SELECT, Value.tensor(this), Value.integer(dim), Value.integer(index));
    }

    public Tensor squeeze() {
        return first(OpRegistry.SQUEEZE, Value.tensor(this));
    }

    public Tensor squeeze(int dim) {
        return first(OpRegistry.SQUEEZE_DIM, Value.tensor(this), Value.integer(dim));
    }

    public Tensor unsqueeze(int dim) {
        return first(OpRegistry.UNSQUEEZE, Value.tensor(this), Value.integer(dim));
    }

    public Tensor expand(long... sizes) {
        return first(OpRegistry.EXPAND, Value.tensor(this), Value.integers(sizes));
    }

    public Tensor flatten() {
        return first(OpRegistry.FLATTEN, Value.tensor(this), Value.integer(0), Value.integer(-1));
    }

    public Tensor flatten(int startDim, int endDim) {
        return first(
                OpRegistry.FLATTEN, Value.tensor(this), Value.integer(startDim), Value.integer(endDim));
    }

    public Tensor t() {
        return first(OpRegistry.T, Value.tensor(this));
    }

    public Tensor contiguous() {
        return first(OpRegistry.CONTIGUOUS, Value.tensor(this));
    }

    public Tensor clone() {
        return first(OpRegistry.CLONE, Value.tensor(this));
    }

    public Tensor detach() {
        return first(OpRegistry.DETACH, Value.tensor(this));
    }

    public Tensor to(DType dtype) {
        return first(OpRegistry.TO_DTYPE, Value.tensor(this), Value.dtype(dtype));
    }

    public Tensor toFloat() {
        return to(DType.FLOAT32);
    }

    public Tensor toDouble() {
        return to(DType.FLOAT64);
    }

    public Tensor toLong() {
        return to(DType.INT64);
    }

    public Tensor toInt() {
        return to(DType.INT32);
    }

    public Tensor toBool() {
        return to(DType.BOOL);
    }

    // ---------------------------------------------------------------- elementwise

    public Tensor add(Tensor other) {
        return first(OpRegistry.ADD_TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(1)));
    }

    public Tensor add(Tensor other, double alpha) {
        return first(
                OpRegistry.ADD_TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(alpha)));
    }

    public Tensor add(Tensor other, long alpha) {
        return first(
                OpRegistry.ADD_TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(alpha)));
    }

    public Tensor sub(Tensor other) {
        return first(OpRegistry.SUB_TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(1)));
    }

    public Tensor sub(Tensor other, double alpha) {
        return first(
                OpRegistry.SUB_TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(alpha)));
    }

    public Tensor sub(Tensor other, long alpha) {
        return first(
                OpRegistry.SUB_TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(alpha)));
    }

    public Tensor mul(Tensor other) {
        return first(OpRegistry.MUL_TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor mul(double scalar) {
        return mul(scalarTensor(Scalar.of(scalar)));
    }

    public Tensor div(Tensor other) {
        return first(OpRegistry.DIV_TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor div(double scalar) {
        return div(scalarTensor(Scalar.of(scalar)));
    }

    public Tensor add(double scalar) {
        return add(scalarTensor(Scalar.of(scalar)));
    }

    public Tensor add(long scalar) {
        return add(scalarTensor(Scalar.of(scalar)));
    }

    public Tensor sub(double scalar) {
        return sub(scalarTensor(Scalar.of(scalar)));
    }

    public Tensor sub(long scalar) {
        return sub(scalarTensor(Scalar.of(scalar)));
    }

    public Tensor mul(long scalar) {
        return mul(scalarTensor(Scalar.of(scalar)));
    }

    public Tensor div(long scalar) {
        return div(scalarTensor(Scalar.of(scalar)));
    }

    public Tensor pow(long exponent) {
        return first(OpRegistry.POW_TENSOR_SCALAR, Value.tensor(this), Value.scalar(Scalar.of(exponent)));
    }

    private Tensor scalarTensor(Scalar scalar) {
        // PyTorch's weak-scalar rule: a fractional scalar lifts an integer tensor to the default float dtype, while
        // an integer scalar must not silently truncate. Build the scalar in the promoted dtype so the binary op
        // applies normal tensor promotion.
        DType promoted = DType.resultTypeWithScalar(dtype(), scalar);
        return first(OpRegistry.FULL, Value.integers(), Value.scalar(scalar), Value.dtype(promoted),
                Value.device(device()));
    }

    public Tensor pow(double exponent) {
        return first(OpRegistry.POW_TENSOR_SCALAR, Value.tensor(this), Value.scalar(Scalar.of(exponent)));
    }

    public Tensor pow(Tensor exponent) {
        return first(OpRegistry.POW_TENSOR_TENSOR, Value.tensor(this), Value.tensor(exponent));
    }

    public Tensor maximum(Tensor other) {
        return first(OpRegistry.MAXIMUM, Value.tensor(this), Value.tensor(other));
    }

    public Tensor minimum(Tensor other) {
        return first(OpRegistry.MINIMUM, Value.tensor(this), Value.tensor(other));
    }

    public Tensor eq(Tensor other) {
        return first(OpRegistry.EQ_TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor ne(Tensor other) {
        return first(OpRegistry.NE_TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor lt(Tensor other) {
        return first(OpRegistry.LT_TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor le(Tensor other) {
        return first(OpRegistry.LE_TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor gt(Tensor other) {
        return first(OpRegistry.GT_TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor ge(Tensor other) {
        return first(OpRegistry.GE_TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor logicalNot() {
        return first(OpRegistry.LOGICAL_NOT, Value.tensor(this));
    }

    public Tensor neg() {
        return unary(OpRegistry.NEG);
    }

    public Tensor abs() {
        return unary(OpRegistry.ABS);
    }

    public Tensor square() {
        return unary(OpRegistry.SQUARE);
    }

    public Tensor sqrt() {
        return unary(OpRegistry.SQRT);
    }

    public Tensor exp() {
        return unary(OpRegistry.EXP);
    }

    public Tensor log() {
        return unary(OpRegistry.LOG);
    }

    public Tensor tanh() {
        return unary(OpRegistry.TANH);
    }

    public Tensor sigmoid() {
        return unary(OpRegistry.SIGMOID);
    }

    public Tensor relu() {
        return unary(OpRegistry.RELU);
    }

    public Tensor reciprocal() {
        return unary(OpRegistry.RECIPROCAL);
    }

    public Tensor floor() {
        return unary(OpRegistry.FLOOR);
    }

    public Tensor ceil() {
        return unary(OpRegistry.CEIL);
    }

    public Tensor sign() {
        return unary(OpRegistry.SIGN);
    }

    public Tensor sin() {
        return unary(OpRegistry.SIN);
    }

    public Tensor cos() {
        return unary(OpRegistry.COS);
    }

    public Tensor log1p() {
        return unary(OpRegistry.LOG1P);
    }

    public Tensor clamp(Scalar min, Scalar max) {
        if (min == null && max == null) {
            throw new JTorchException("clamp requires at least one of min or max");
        }
        return first(
                OpRegistry.CLAMP,
                Value.tensor(this),
                min == null ? Value.none() : Value.scalar(min),
                max == null ? Value.none() : Value.scalar(max));
    }

    public Tensor clamp(double min, double max) {
        return clamp(Scalar.of(min), Scalar.of(max));
    }

    public Tensor clamp(long min, long max) {
        return clamp(Scalar.of(min), Scalar.of(max));
    }

    public Tensor clampMin(double min) {
        return clamp(Scalar.of(min), null);
    }

    public Tensor clampMin(long min) {
        return clamp(Scalar.of(min), null);
    }

    public Tensor clampMax(double max) {
        return clamp(null, Scalar.of(max));
    }

    public Tensor clampMax(long max) {
        return clamp(null, Scalar.of(max));
    }

    // ---------------------------------------------------------------- reductions

    public Tensor sum() {
        return first(OpRegistry.SUM, Value.tensor(this));
    }

    public Tensor sum(int dim, boolean keepdim) {
        return sum(new int[] {dim}, keepdim);
    }

    public Tensor sum(int[] dims, boolean keepdim) {
        return first(
                OpRegistry.SUM_DIM_INTLIST, Value.tensor(this), Value.integers(toLongs(dims)), Value.bool(keepdim));
    }

    public Tensor mean() {
        return first(OpRegistry.MEAN, Value.tensor(this));
    }

    public Tensor mean(int dim, boolean keepdim) {
        return mean(new int[] {dim}, keepdim);
    }

    public Tensor mean(int[] dims, boolean keepdim) {
        return first(
                OpRegistry.MEAN_DIM, Value.tensor(this), Value.integers(toLongs(dims)), Value.bool(keepdim));
    }

    public Tensor prod() {
        return first(OpRegistry.PROD, Value.tensor(this));
    }

    public Tensor prod(int dim, boolean keepdim) {
        return first(OpRegistry.PROD_DIM, Value.tensor(this), Value.integer(dim), Value.bool(keepdim));
    }

    public Tensor max() {
        return first(OpRegistry.MAX, Value.tensor(this));
    }

    public Tensor min() {
        return first(OpRegistry.MIN, Value.tensor(this));
    }

    /** PyTorch's {@code max(dim)} tuple: values plus int64 indices. */
    public TensorTuple maxWithIndices(int dim, boolean keepdim) {
        Value[] outputs =
                Dispatcher.get().call(OpRegistry.MAX_DIM, Value.tensor(this), Value.integer(dim), Value.bool(keepdim));
        return new TensorTuple(outputs[0].asTensor(), outputs[1].asTensor());
    }

    public TensorTuple minWithIndices(int dim, boolean keepdim) {
        Value[] outputs =
                Dispatcher.get().call(OpRegistry.MIN_DIM, Value.tensor(this), Value.integer(dim), Value.bool(keepdim));
        return new TensorTuple(outputs[0].asTensor(), outputs[1].asTensor());
    }

    public Tensor argmax() {
        return first(OpRegistry.ARGMAX, Value.tensor(this), Value.none(), Value.bool(false));
    }

    public Tensor argmax(int dim, boolean keepdim) {
        return first(OpRegistry.ARGMAX, Value.tensor(this), Value.integer(dim), Value.bool(keepdim));
    }

    public Tensor argmin() {
        return first(OpRegistry.ARGMIN, Value.tensor(this), Value.none(), Value.bool(false));
    }

    public Tensor argmin(int dim, boolean keepdim) {
        return first(OpRegistry.ARGMIN, Value.tensor(this), Value.integer(dim), Value.bool(keepdim));
    }

    public Tensor softmax(int dim) {
        return first(OpRegistry.SOFTMAX, Value.tensor(this), Value.integer(dim));
    }

    public Tensor logSoftmax(int dim) {
        return first(OpRegistry.LOG_SOFTMAX, Value.tensor(this), Value.integer(dim));
    }

    // ---------------------------------------------------------------- matrix and shape combination

    public Tensor matmul(Tensor other) {
        return first(OpRegistry.MATMUL, Value.tensor(this), Value.tensor(other));
    }

    public Tensor mm(Tensor other) {
        return first(OpRegistry.MM, Value.tensor(this), Value.tensor(other));
    }

    public Tensor bmm(Tensor other) {
        return first(OpRegistry.BMM, Value.tensor(this), Value.tensor(other));
    }

    public Tensor addmm(Tensor mat1, Tensor mat2, double beta, double alpha) {
        return first(
                OpRegistry.ADDMM,
                Value.tensor(this),
                Value.tensor(mat1),
                Value.tensor(mat2),
                Value.scalar(Scalar.of(beta)),
                Value.scalar(Scalar.of(alpha)));
    }

    public Tensor addmm(Tensor mat1, Tensor mat2) {
        return addmm(mat1, mat2, 1.0, 1.0);
    }

    public Tensor linear(Tensor weight, Tensor bias) {
        return first(
                OpRegistry.LINEAR,
                Value.tensor(this),
                Value.tensor(weight),
                bias == null ? Value.none() : Value.tensor(bias));
    }

    public Tensor mseLoss(Tensor target, String reduction) {
        return first(OpRegistry.MSE_LOSS, Value.tensor(this), Value.tensor(target), Value.string(reduction));
    }

    public Tensor crossEntropy(Tensor target, String reduction) {
        return first(OpRegistry.CROSS_ENTROPY, Value.tensor(this), Value.tensor(target), Value.string(reduction));
    }

    public static Tensor cat(List<Tensor> tensors, int dim) {
        return firstOf(OpRegistry.CAT, Value.tensors(tensors.toArray(Tensor[]::new)), Value.integer(dim));
    }

    public static Tensor stack(List<Tensor> tensors, int dim) {
        return firstOf(OpRegistry.STACK, Value.tensors(tensors.toArray(Tensor[]::new)), Value.integer(dim));
    }

    // ---------------------------------------------------------------- in-place

    public Tensor add_(Tensor other) {
        return inPlace(OpRegistry.ADD__TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(1)));
    }

    public Tensor add_(double scalar) {
        return inPlace(OpRegistry.ADD__TENSOR, Value.tensor(this), Value.tensor(scalarTensor(Scalar.of(scalar))),
                Value.scalar(Scalar.of(1)));
    }

    public Tensor add_(long scalar) {
        return add_((double) scalar);
    }

    public Tensor sub_(double scalar) {
        return inPlace(OpRegistry.SUB__TENSOR, Value.tensor(this), Value.tensor(scalarTensor(Scalar.of(scalar))),
                Value.scalar(Scalar.of(1)));
    }

    public Tensor sub_(long scalar) {
        return sub_((double) scalar);
    }

    public Tensor mul_(double scalar) {
        return inPlace(OpRegistry.MUL__TENSOR, Value.tensor(this), Value.tensor(scalarTensor(Scalar.of(scalar))));
    }

    public Tensor mul_(long scalar) {
        return inPlace(OpRegistry.MUL__TENSOR, Value.tensor(this), Value.tensor(scalarTensor(Scalar.of(scalar))));
    }

    public Tensor div_(double scalar) {
        return inPlace(OpRegistry.DIV__TENSOR, Value.tensor(this), Value.tensor(scalarTensor(Scalar.of(scalar))));
    }

    public Tensor div_(long scalar) {
        return inPlace(OpRegistry.DIV__TENSOR, Value.tensor(this), Value.tensor(scalarTensor(Scalar.of(scalar))));
    }

    public Tensor add_(Tensor other, double alpha) {
        return inPlace(
                OpRegistry.ADD__TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(alpha)));
    }

    public Tensor sub_(Tensor other) {
        return inPlace(OpRegistry.SUB__TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(1)));
    }

    public Tensor sub_(Tensor other, double alpha) {
        return inPlace(
                OpRegistry.SUB__TENSOR, Value.tensor(this), Value.tensor(other), Value.scalar(Scalar.of(alpha)));
    }

    public Tensor mul_(Tensor other) {
        return inPlace(OpRegistry.MUL__TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor div_(Tensor other) {
        return inPlace(OpRegistry.DIV__TENSOR, Value.tensor(this), Value.tensor(other));
    }

    public Tensor copy_(Tensor source) {
        return inPlace(OpRegistry.COPY_, Value.tensor(this), Value.tensor(source));
    }

    public Tensor zero_() {
        return inPlace(OpRegistry.ZERO_, Value.tensor(this));
    }

    public Tensor fill_(Scalar value) {
        return inPlace(OpRegistry.FILL__SCALAR, Value.tensor(this), Value.scalar(value));
    }

    public Tensor fill_(double value) {
        return fill_(Scalar.of(value));
    }

    public Tensor fill_(long value) {
        return fill_(Scalar.of(value));
    }

    // ---------------------------------------------------------------- autograd

    public void backward() {
        JTorch.gradientEngine().backward(this, null, false, false);
    }

    public void backward(Tensor gradient) {
        JTorch.gradientEngine().backward(this, gradient, false, false);
    }

    public void backward(boolean retainGraph) {
        JTorch.gradientEngine().backward(this, null, retainGraph, false);
    }

    public void backward(Tensor gradient, boolean retainGraph) {
        JTorch.gradientEngine().backward(this, gradient, retainGraph, false);
    }

    // ---------------------------------------------------------------- data access

    public double item() {
        checkSingleElement();
        if (dtype().isIntegral()) {
            long value = TensorAccess.getLong(impl, impl.storageOffset());
            double converted = (double) value;
            if ((long) converted != value) {
                throw new JTorchException("item() cannot represent int64 value " + value
                        + " exactly as a double; use itemLong()");
            }
            return converted;
        }
        if (dtype().isBoolean()) {
            return TensorAccess.getBoolean(impl, impl.storageOffset()) ? 1 : 0;
        }
        return TensorAccess.getDouble(impl, impl.storageOffset());
    }

    public float itemFloat() {
        checkSingleElement();
        return TensorAccess.getFloat(impl, impl.storageOffset());
    }

    public long itemLong() {
        checkSingleElement();
        return TensorAccess.getLong(impl, impl.storageOffset());
    }

    public boolean itemBoolean() {
        checkSingleElement();
        return TensorAccess.getBoolean(impl, impl.storageOffset());
    }

    public double get(long... indices) {
        long[] normalized = normalizeIndices(indices);
        return TensorAccess.getDouble(impl, TensorAccess.offsetOf(impl, normalized));
    }

    public Tensor set(double value, long... indices) {
        requireWritable();
        long[] normalized = normalizeIndices(indices);
        if (dtype().isIntegral() || dtype().isBoolean()) {
            TensorAccess.setLong(impl, TensorAccess.offsetOf(impl, normalized), (long) value);
        } else {
            TensorAccess.setDouble(impl, TensorAccess.offsetOf(impl, normalized), value);
        }
        impl.bumpVersion();
        return this;
    }

    public Tensor set(long value, long... indices) {
        requireWritable();
        long[] normalized = normalizeIndices(indices);
        TensorAccess.setLong(impl, TensorAccess.offsetOf(impl, normalized), value);
        impl.bumpVersion();
        return this;
    }

    private void requireWritable() {
        if (impl.requiresGrad() && GradMode.isEnabled()) {
            throw new AutogradException("a tensor that requires grad is being mutated outside GradMode.noGrad");
        }
    }

    public float[] toFloatArray() {
        int size = checkedArraySize();
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = TensorAccess.getFloat(impl, elementOffset(i));
        }
        return values;
    }

    public double[] toDoubleArray() {
        int size = checkedArraySize();
        double[] values = new double[size];
        for (int i = 0; i < size; i++) {
            values[i] = TensorAccess.getDouble(impl, elementOffset(i));
        }
        return values;
    }

    public long[] toLongArray() {
        int size = checkedArraySize();
        long[] values = new long[size];
        for (int i = 0; i < size; i++) {
            values[i] = TensorAccess.getLong(impl, elementOffset(i));
        }
        return values;
    }

    public boolean[] toBooleanArray() {
        int size = checkedArraySize();
        boolean[] values = new boolean[size];
        for (int i = 0; i < size; i++) {
            values[i] = TensorAccess.getBoolean(impl, elementOffset(i));
        }
        return values;
    }

    /** Storage element offset of a logical linear index in row-major order. */
    public long elementOffset(long linearIndex) {
        int rank = dim();
        long offset = impl.storageOffset();
        long remaining = linearIndex;
        for (int d = rank - 1; d >= 0; d--) {
            long size = size(d);
            if (size == 0) {
                continue;
            }
            long index = remaining % size;
            remaining /= size;
            offset += index * impl.strides().stride(d);
        }
        return offset;
    }

    private void checkSingleElement() {
        if (numel() != 1) {
            throw new JTorchException("item() requires a tensor with exactly one element, got " + numel());
        }
    }

    private int checkedArraySize() {
        long numel = numel();
        if (numel > Integer.MAX_VALUE) {
            throw new JTorchException("Tensor has " + numel + " elements, too many for a Java array");
        }
        return (int) numel;
    }

    private long[] normalizeIndices(long[] indices) {
        if (indices.length != dim()) {
            throw new IndexOutOfBoundsException(
                    "Expected " + dim() + " indices but received " + indices.length);
        }
        long[] normalized = new long[indices.length];
        for (int i = 0; i < indices.length; i++) {
            long index = indices[i];
            if (index < 0) {
                index += size(i);
            }
            if (index < 0 || index >= size(i)) {
                throw new IndexOutOfBoundsException(
                        "Index " + indices[i] + " is out of bounds for dimension " + i + " with size " + size(i));
            }
            normalized[i] = index;
        }
        return normalized;
    }

    // ---------------------------------------------------------------- helpers

    private Tensor first(Op op, Value... args) {
        return firstOf(op, args);
    }

    private static Tensor firstOf(Op op, Value... args) {
        return Dispatcher.get().call(op, args)[0].asTensor();
    }

    private Tensor unary(Op op) {
        return first(op, Value.tensor(this));
    }

    private Tensor inPlace(Op op, Value... args) {
        Dispatcher.get().call(op, args);
        return this;
    }

    private static long[] toLongs(int[] values) {
        long[] result = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder out = new StringBuilder("tensor(");
        long numel = numel();
        if (numel == 0) {
            out.append("[]");
        } else if (numel <= 64 && dim() <= 2) {
            appendValues(out);
        } else {
            out.append("<").append(numel).append(" elements>");
        }
        out.append(", dtype=").append(dtype());
        if (!device().isCpu()) {
            out.append(", device=").append(device());
        }
        if (requiresGrad()) {
            out.append(", requires_grad=true");
        }
        if (impl.gradFn() != null) {
            out.append(", grad_fn=<").append(impl.gradFn().name()).append(">");
        }
        return out.append(')').toString();
    }

    private void appendValues(StringBuilder out) {
        if (dim() == 0) {
            appendNumber(out, 0);
            return;
        }
        if (dim() == 1) {
            out.append('[');
            for (long i = 0; i < size(0); i++) {
                if (i > 0) {
                    out.append(", ");
                }
                appendNumber(out, elementOffset(i));
            }
            out.append(']');
            return;
        }
        out.append('[');
        for (long i = 0; i < size(0); i++) {
            if (i > 0) {
                out.append(",\n       ");
            }
            out.append('[');
            for (long j = 0; j < size(1); j++) {
                if (j > 0) {
                    out.append(", ");
                }
                appendNumber(out, elementOffset(i * size(1) + j));
            }
            out.append(']');
        }
        out.append(']');
    }

    private void appendNumber(StringBuilder out, long offset) {
        switch (dtype()) {
            case BOOL -> out.append(TensorAccess.getBoolean(impl, offset));
            case INT32, INT64 -> out.append(TensorAccess.getLong(impl, offset));
            case FLOAT32 -> out.append(TensorAccess.getFloat(impl, offset));
            case FLOAT64 -> out.append(TensorAccess.getDouble(impl, offset));
        }
    }

    /** A pair result for operators that return values and indices. */
    public record TensorTuple(Tensor values, Tensor indices) {}
}
