package org.jtorch.core.dispatch;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import org.jtorch.core.AutogradException;
import org.jtorch.core.Device;
import org.jtorch.core.DeviceException;
import org.jtorch.core.GradMode;
import org.jtorch.core.JTorch;
import org.jtorch.core.JTorchException;
import org.jtorch.core.NotSupportedException;
import org.jtorch.core.Op;
import org.jtorch.core.Tensor;
import org.jtorch.core.value.Value;

/**
 * Prioritised, composable operator dispatcher.
 *
 * <p>A call derives its dispatch key set from operator arguments (backend device plus autograd), then consults keys
 * from highest to lowest priority. Wrappers redispatch by removing their own key; backend kernels are reachable only
 * through this class.
 */
public final class Dispatcher {
    private static final Dispatcher INSTANCE = new Dispatcher();
    private static final Map<DispatchKey, Kernel> NO_KERNELS = Map.of();

    private final Map<Op, Map<DispatchKey, Kernel>> table = new ConcurrentHashMap<>();

    private Dispatcher() {}

    public static Dispatcher get() {
        return INSTANCE;
    }

    /** Registers a kernel, failing loudly on duplicate registration. */
    public synchronized void register(Op op, DispatchKey key, Kernel kernel) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(kernel, "kernel");
        Map<DispatchKey, Kernel> kernels =
                table.computeIfAbsent(op, ignored -> new ConcurrentHashMap<>());
        Kernel previous = kernels.putIfAbsent(key, kernel);
        if (previous != null) {
            throw new IllegalStateException("Duplicate " + key + " kernel registration for " + op.name());
        }
    }

    /** Convenience registration for single-output kernels. */
    @FunctionalInterface
    public interface TensorKernel {
        Tensor invoke(Op op, Value[] args, DispatchKeySet keys);
    }

    public void registerTensor(Op op, DispatchKey key, TensorKernel kernel) {
        register(op, key, (called, args, keys) -> new Value[] {Value.tensor(kernel.invoke(called, args, keys))});
    }

    public boolean hasKernel(Op op, DispatchKey key) {
        Map<DispatchKey, Kernel> kernels = table.get(op);
        return kernels != null && kernels.containsKey(key);
    }

    public int kernelCount() {
        int total = 0;
        for (Map<DispatchKey, Kernel> kernels : table.values()) {
            total += kernels.size();
        }
        return total;
    }

    /** Immutable view of registrations, used by diagnostics and compatibility reports. */
    public Map<Op, Map<DispatchKey, Kernel>> registrations() {
        Map<Op, Map<DispatchKey, Kernel>> copy = new ConcurrentHashMap<>();
        table.forEach((op, kernels) -> copy.put(op, Collections.unmodifiableMap(Map.copyOf(kernels))));
        return Collections.unmodifiableMap(copy);
    }

    public Value[] call(Op op, Value... args) {
        DispatchKeySet keys = computeKeys(op, args);
        return callWithKeys(op, args, keys);
    }

    /** Executes an operator with an explicit key set. Wrappers use this to redispatch after removing their key. */
    public Value[] callWithKeys(Op op, Value[] args, DispatchKeySet keys) {
        JTorch.ensureInitialized();
        validateArity(op, args);
        if (keys.has(DispatchKey.AUTOGRAD)) {
            Kernel autograd = lookup(op, DispatchKey.AUTOGRAD);
            if (autograd == null) {
                throw new AutogradException("No autograd kernel is registered for " + op.name());
            }
            return autograd.invoke(op, args, keys);
        }
        DispatchKey backend = keys.has(DispatchKey.META) ? DispatchKey.META : DispatchKey.CPU;
        Kernel kernel = lookup(op, backend);
        if (kernel == null) {
            kernel = lookup(op, DispatchKey.COMPOSITE);
        }
        if (kernel == null) {
            throw new NotSupportedException(
                    "No implementation for " + op.name() + " on " + backend + " (keys " + keys + ")");
        }
        return kernel.invoke(op, args, keys);
    }

    private Kernel lookup(Op op, DispatchKey key) {
        Map<DispatchKey, Kernel> kernels = table.get(op);
        return kernels == null ? null : kernels.get(key);
    }

    private static void validateArity(Op op, Value[] args) {
        int expected = op.schema().args().size();
        if (args.length != expected) {
            throw new JTorchException("Operator " + op.name() + " expects " + expected + " arguments but received "
                    + args.length);
        }
    }

    /**
     * Derives the dispatch key set: the backend key from tensor devices and the autograd key when grad mode is on,
     * the operator has a native backward formula and at least one tensor input tracks gradients.
     */
    public static DispatchKeySet computeKeys(Op op, Value[] args) {
        Device device = null;
        boolean requiresGrad = false;
        for (Value arg : args) {
            if (arg instanceof Value.TensorValue tensorValue) {
                device = merge(device, tensorValue.value().device());
                requiresGrad |= tensorValue.value().requiresGrad();
            } else if (arg instanceof Value.TensorListValue list) {
                for (Tensor tensor : list.values()) {
                    device = merge(device, tensor.device());
                    requiresGrad |= tensor.requiresGrad();
                }
            }
        }
        if (device == null) {
            for (Value arg : args) {
                if (arg instanceof Value.DeviceValue deviceValue) {
                    device = deviceValue.value();
                    break;
                }
            }
        }
        if (device == null) {
            device = Device.cpu();
        }
        if (!device.isCpu() && !device.isMeta()) {
            throw new DeviceException("Device " + device + " has no JTorch backend in this build");
        }
        DispatchKey backend = device.isMeta() ? DispatchKey.META : DispatchKey.CPU;
        DispatchKeySet keys = DispatchKeySet.of(backend);
        boolean differentiable = op.differentiable() && !op.composite();
        if (differentiable && requiresGrad && GradMode.isEnabled()) {
            keys = keys.union(DispatchKey.AUTOGRAD);
        }
        return keys;
    }

    private static Device merge(Device current, Device next) {
        if (current == null) {
            return next;
        }
        if (!current.equals(next)) {
            throw new DeviceException(
                    "Expected all tensors to be on the same device, but found " + current + " and " + next);
        }
        return current;
    }
}
