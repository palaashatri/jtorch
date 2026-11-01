package org.jtorch.api;

import java.nio.file.Path;
import java.util.function.Supplier;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.GradMode;
import org.jtorch.core.JTorch;
import org.jtorch.core.OpRegistry;
import org.jtorch.core.Scalar;
import org.jtorch.core.Shape;
import org.jtorch.core.Strides;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorImpl;
import org.jtorch.core.TensorOptions;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.memory.MappedStorage;
import org.jtorch.core.memory.NativeStorage;
import org.jtorch.core.memory.Storage;
import org.jtorch.core.memory.TensorAccess;
import org.jtorch.core.value.Value;

/** The canonical Java facade, mirroring {@code torch} entry points. */
public final class Torch {
    private Torch() {}

    static {
        // Fail fast at first use rather than at the first operator call.
        JTorch.ensureInitialized();
    }

    // ---------------------------------------------------------------- creation

    public static Tensor tensor(float[] data, Shape shape) {
        return tensor(data, shape, TensorOptions.defaults());
    }

    public static Tensor tensor(float[] data, Shape shape, TensorOptions options) {
        checkDataLength(data.length, shape);
        Tensor result = empty(shape, options);
        requireHost(result);
        for (long i = 0; i < result.numel(); i++) {
            TensorAccess.setFloat(result.impl(), result.elementOffset(i), data[(int) i]);
        }
        applyRequiresGrad(result, options);
        return result;
    }

    public static Tensor tensor(double[] data, Shape shape) {
        return tensor(data, shape, TensorOptions.defaults().dtype(DType.FLOAT64));
    }

    public static Tensor tensor(double[] data, Shape shape, TensorOptions options) {
        checkDataLength(data.length, shape);
        Tensor result = empty(shape, options);
        requireHost(result);
        for (long i = 0; i < result.numel(); i++) {
            TensorAccess.setDouble(result.impl(), result.elementOffset(i), data[(int) i]);
        }
        applyRequiresGrad(result, options);
        return result;
    }

    public static Tensor tensor(long[] data, Shape shape) {
        return tensor(data, shape, TensorOptions.defaults().dtype(DType.INT64));
    }

    public static Tensor tensor(long[] data, Shape shape, TensorOptions options) {
        checkDataLength(data.length, shape);
        Tensor result = empty(shape, options);
        requireHost(result);
        for (long i = 0; i < result.numel(); i++) {
            TensorAccess.setLong(result.impl(), result.elementOffset(i), data[(int) i]);
        }
        applyRequiresGrad(result, options);
        return result;
    }

    public static Tensor tensor(int[] data, Shape shape) {
        return tensor(data, shape, TensorOptions.defaults().dtype(DType.INT32));
    }

    public static Tensor tensor(int[] data, Shape shape, TensorOptions options) {
        checkDataLength(data.length, shape);
        Tensor result = empty(shape, options);
        requireHost(result);
        for (long i = 0; i < result.numel(); i++) {
            TensorAccess.setLong(result.impl(), result.elementOffset(i), data[(int) i]);
        }
        applyRequiresGrad(result, options);
        return result;
    }

    public static Tensor tensor(boolean[] data, Shape shape) {
        return tensor(data, shape, TensorOptions.defaults().dtype(DType.BOOL));
    }

    public static Tensor tensor(boolean[] data, Shape shape, TensorOptions options) {
        checkDataLength(data.length, shape);
        Tensor result = empty(shape, options);
        requireHost(result);
        for (long i = 0; i < result.numel(); i++) {
            TensorAccess.setBoolean(result.impl(), result.elementOffset(i), data[(int) i]);
        }
        applyRequiresGrad(result, options);
        return result;
    }

    public static Tensor zeros(Shape shape) {
        return zeros(shape, TensorOptions.defaults());
    }

    public static Tensor zeros(Shape shape, TensorOptions options) {
        return factory(OpRegistry.ZEROS, options, Value.integers(shape.toArray()), dtypeValue(options),
                Value.device(options.device()));
    }

    public static Tensor ones(Shape shape) {
        return ones(shape, TensorOptions.defaults());
    }

    public static Tensor ones(Shape shape, TensorOptions options) {
        return factory(OpRegistry.ONES, options, Value.integers(shape.toArray()), dtypeValue(options),
                Value.device(options.device()));
    }

    public static Tensor empty(Shape shape) {
        return empty(shape, TensorOptions.defaults());
    }

    public static Tensor empty(Shape shape, TensorOptions options) {
        return factory(OpRegistry.EMPTY, options, Value.integers(shape.toArray()), dtypeValue(options),
                Value.device(options.device()));
    }

    public static Tensor full(Shape shape, Scalar value) {
        DType dtype = switch (value.kind()) {
            case BOOL -> DType.BOOL;
            case INT -> DType.INT64;
            case FLOAT -> DType.defaultFloat();
        };
        return full(shape, value, TensorOptions.defaults().dtype(dtype));
    }

    public static Tensor full(Shape shape, Scalar value, TensorOptions options) {
        return factory(OpRegistry.FULL, options, Value.integers(shape.toArray()), Value.scalar(value),
                dtypeValue(options), Value.device(options.device()));
    }

    public static Tensor arange(long start, long end) {
        return arange(Scalar.of(start), Scalar.of(end), Scalar.of(1),
                TensorOptions.defaults().dtype(DType.INT64));
    }

    public static Tensor arange(double start, double end, double step) {
        return arange(Scalar.of(start), Scalar.of(end), Scalar.of(step), TensorOptions.defaults());
    }

    public static Tensor arange(Scalar start, Scalar end, Scalar step, TensorOptions options) {
        return factory(OpRegistry.ARANGE, options, Value.scalar(start), Value.scalar(end), Value.scalar(step),
                dtypeValue(options), Value.device(options.device()));
    }

    public static Tensor randn(Shape shape) {
        return randn(shape, TensorOptions.defaults());
    }

    public static Tensor randn(Shape shape, TensorOptions options) {
        requireFloatingDtype(options, "randn");
        return factory(OpRegistry.RANDN, options, Value.integers(shape.toArray()), dtypeValue(options),
                Value.device(options.device()));
    }

    public static Tensor rand(Shape shape) {
        return rand(shape, TensorOptions.defaults());
    }

    public static Tensor rand(Shape shape, TensorOptions options) {
        requireFloatingDtype(options, "rand");
        return factory(OpRegistry.RAND, options, Value.integers(shape.toArray()), dtypeValue(options),
                Value.device(options.device()));
    }

    public static Tensor randint(long low, long high, Shape shape) {
        return randint(low, high, shape, TensorOptions.defaults().dtype(DType.INT64));
    }

    public static Tensor randint(long low, long high, Shape shape, TensorOptions options) {
        return factory(OpRegistry.RANDINT, options, Value.integer(low), Value.integer(high),
                Value.integers(shape.toArray()), dtypeValue(options), Value.device(options.device()));
    }

    public static Tensor eye(int n) {
        return eye(n, TensorOptions.defaults());
    }

    public static Tensor eye(int n, TensorOptions options) {
        return factory(OpRegistry.EYE, options, Value.integer(n), dtypeValue(options),
                Value.device(options.device()));
    }

    /** Maps a file of raw little-endian host-dtype values as a tensor. */
    public static Tensor fromMappedFile(Path path, DType dtype, Shape shape, boolean writable) {
        Storage storage = MappedStorage.map(path, dtype, shape.numel(), writable);
        return Tensor.wrap(TensorImpl.fresh(storage, shape, Strides.contiguous(shape), false));
    }

    private static Tensor factory(org.jtorch.core.Op op, TensorOptions options, Value... args) {
        Tensor tensor = Dispatcher.get().call(op, args)[0].asTensor();
        applyRequiresGrad(tensor, options);
        return tensor;
    }

    private static Value dtypeValue(TensorOptions options) {
        return Value.dtype(options.dtype());
    }

    private static void applyRequiresGrad(Tensor tensor, TensorOptions options) {
        if (options.requiresGrad()) {
            tensor.requiresGrad(true);
        }
    }

    private static void requireFloatingDtype(TensorOptions options, String op) {
        if (!options.dtype().isFloatingPoint()) {
            throw new org.jtorch.core.DTypeException(op + " requires a floating point dtype, got " + options.dtype());
        }
    }

    private static void checkDataLength(int actual, Shape shape) {
        if (actual != shape.numel()) {
            throw new IllegalArgumentException(
                    "Expected " + shape.numel() + " values for shape " + shape + " but received " + actual);
        }
    }

    private static void requireHost(Tensor tensor) {
        if (!tensor.device().isCpu()) {
            throw new org.jtorch.core.DeviceException(
                    "Cannot fill a tensor from a Java array on device " + tensor.device());
        }
    }

    // ---------------------------------------------------------------- modes and randomness

    public static void manualSeed(long seed) {
        org.jtorch.cpu.CpuRandom.manualSeed(seed);
    }

    public static void noGrad(Runnable action) {
        GradMode.noGrad(action);
    }

    public static <T> T noGrad(Supplier<T> action) {
        return GradMode.noGrad(action);
    }

    public static void enableGrad(Runnable action) {
        GradMode.enableGrad(action);
    }

    public static boolean isGradEnabled() {
        return GradMode.isEnabled();
    }

    public static void inferenceMode(Runnable action) {
        GradMode.inferenceMode(action);
    }

    public static DType getDefaultDtype() {
        return DType.defaultFloat();
    }

    public static Device defaultDevice() {
        return Device.cpu();
    }

    /** Number of bytes currently held by live native allocations; used by leak diagnostics. */
    public static long liveAllocatedBytes() {
        return NativeStorage.liveBytes();
    }
}
