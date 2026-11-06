package org.jtorch.core;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;
import org.jtorch.core.autograd.Node;
import org.jtorch.core.memory.Storage;

/** The metadata and storage behind a {@link Tensor} handle. */
public final class TensorImpl {
    private final Storage storage;
    private final long storageOffset;
    private final Shape sizes;
    private final Strides strides;
    private final DType dtype;
    private final Device device;
    private final VersionCounter version;
    private final boolean view;
    private final List<Consumer<Tensor>> postAccumulateHooks = new CopyOnWriteArrayList<>();

    private boolean requiresGrad;
    private Tensor grad;
    private Node gradFn;
    private int outputNr;
    private boolean graphFreed;

    public TensorImpl(
            Storage storage,
            long storageOffset,
            Shape sizes,
            Strides strides,
            VersionCounter version,
            boolean requiresGrad,
            boolean view) {
        if (storage == null) {
            throw new JTorchException("Storage-backed tensors require storage");
        }
        if (sizes.rank() != strides.rank()) {
            throw new ShapeException("Shape rank " + sizes.rank() + " does not match strides rank " + strides.rank());
        }
        this.storage = storage;
        this.storageOffset = storageOffset;
        this.sizes = sizes;
        this.strides = strides;
        this.dtype = storage.dtype();
        this.device = storage.device();
        this.version = version;
        this.view = view;
        validateBounds();
        setRequiresGradInternal(requiresGrad);
    }

    private TensorImpl(
            Shape sizes,
            Strides strides,
            DType dtype,
            Device device,
            VersionCounter version,
            boolean requiresGrad,
            boolean view) {
        this.storage = null;
        this.storageOffset = 0;
        this.sizes = sizes;
        this.strides = strides;
        this.dtype = dtype;
        this.device = device;
        this.version = version;
        this.view = view;
        setRequiresGradInternal(requiresGrad);
    }

    public static TensorImpl fresh(Storage storage, Shape sizes, Strides strides, boolean requiresGrad) {
        return new TensorImpl(storage, 0, sizes, strides, new VersionCounter(), requiresGrad, false);
    }

    public static TensorImpl meta(Shape sizes, DType dtype, boolean requiresGrad) {
        return new TensorImpl(
                sizes, Strides.contiguous(sizes), dtype, Device.meta(), new VersionCounter(), requiresGrad, false);
    }

    public static TensorImpl meta(Shape sizes, Strides strides, DType dtype, boolean requiresGrad) {
        return new TensorImpl(sizes, strides, dtype, Device.meta(), new VersionCounter(), requiresGrad, true);
    }

    /** Creates an alias with the same storage, version counter and storage offset. */
    public TensorImpl alias(Shape newSizes, Strides newStrides) {
        return new TensorImpl(storage, storageOffset, newSizes, newStrides, version, requiresGrad, true);
    }

    public TensorImpl alias(Shape newSizes, Strides newStrides, long newOffset) {
        return new TensorImpl(storage, newOffset, newSizes, newStrides, version, requiresGrad, true);
    }

    private void validateBounds() {
        long elements = storage.byteSize() / dtype.sizeBytes();
        if (storageOffset < 0 || storageOffset > elements) {
            throw new ShapeException("Storage offset " + storageOffset + " is outside storage of " + elements
                    + " elements");
        }
        if (sizes.numel() == 0) {
            return;
        }
        long maxOffset = storageOffset;
        for (int i = 0; i < sizes.rank(); i++) {
            long extent;
            try {
                extent = Math.multiplyExact(sizes.dim(i) - 1, strides.stride(i));
            } catch (ArithmeticException error) {
                throw new ShapeException("Tensor offset computation overflows a 64-bit index");
            }
            try {
                maxOffset = Math.addExact(maxOffset, extent);
            } catch (ArithmeticException error) {
                throw new ShapeException("Tensor offset computation overflows a 64-bit index");
            }
        }
        if (maxOffset >= elements) {
            throw new ShapeException("Tensor accesses element " + maxOffset + " but storage holds " + elements
                    + " elements");
        }
    }

    private void setRequiresGradInternal(boolean value) {
        if (value && !dtype.isFloatingOrComplex()) {
            throw new DTypeException("Only floating point or complex tensors can require gradients, got " + dtype);
        }
        this.requiresGrad = value;
    }

    public Storage storage() {
        return storage;
    }

    public MemorySegment segment() {
        if (storage == null) {
            throw new JTorchException("Meta tensors have no storage; use a CPU tensor for data access");
        }
        return storage.segment();
    }

    public long storageOffset() {
        return storageOffset;
    }

    public Shape sizes() {
        return sizes;
    }

    public Strides strides() {
        return strides;
    }

    public DType dtype() {
        return dtype;
    }

    public Device device() {
        return device;
    }

    public Layout layout() {
        return Layout.STRIDED;
    }

    public VersionCounter version() {
        return version;
    }

    public boolean isView() {
        return view;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public void setRequiresGrad(boolean value) {
        if (gradFn != null) {
            throw new AutogradException("Only leaf tensors can toggle requires_grad; this tensor has a grad_fn");
        }
        setRequiresGradInternal(value);
    }

    public boolean isLeaf() {
        return gradFn == null;
    }

    public Tensor grad() {
        return grad;
    }

    public void setGrad(Tensor value) {
        this.grad = value;
    }

    public Node gradFn() {
        return gradFn;
    }

    public int outputNr() {
        return outputNr;
    }

    public void setGradFn(Node node, int outputNumber) {
        this.gradFn = node;
        this.outputNr = outputNumber;
    }

    public void clearGradFn() {
        this.gradFn = null;
        this.outputNr = 0;
        this.graphFreed = true;
    }

    public boolean isGraphFreed() {
        return graphFreed;
    }

    public void bumpVersion() {
        version.bump();
    }

    public List<Consumer<Tensor>> postAccumulateHooks() {
        return postAccumulateHooks;
    }

    public void addPostAccumulateHook(Consumer<Tensor> hook) {
        postAccumulateHooks.add(hook);
    }

    /** True when both tensors are views of exactly the same storage object. */
    public boolean sharesStorage(TensorImpl other) {
        return storage != null && storage == other.storage;
    }

    public boolean isContiguous() {
        return strides.isContiguous(sizes);
    }

    public long numel() {
        return sizes.numel();
    }
}
