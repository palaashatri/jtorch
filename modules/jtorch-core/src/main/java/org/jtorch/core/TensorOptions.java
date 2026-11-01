package org.jtorch.core;

import java.util.Objects;

/** Immutable tensor construction options. */
public final class TensorOptions {
    private static final TensorOptions DEFAULT =
            new TensorOptions(DType.FLOAT32, Device.cpu(), Layout.STRIDED, false);

    private final DType dtype;
    private final Device device;
    private final Layout layout;
    private final boolean requiresGrad;

    private TensorOptions(DType dtype, Device device, Layout layout, boolean requiresGrad) {
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.device = Objects.requireNonNull(device, "device");
        this.layout = Objects.requireNonNull(layout, "layout");
        this.requiresGrad = requiresGrad;
    }

    public static TensorOptions defaults() {
        return DEFAULT;
    }

    public DType dtype() {
        return dtype;
    }

    public Device device() {
        return device;
    }

    public Layout layout() {
        return layout;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public TensorOptions dtype(DType value) {
        return new TensorOptions(value, device, layout, requiresGrad);
    }

    public TensorOptions device(Device value) {
        return new TensorOptions(dtype, value, layout, requiresGrad);
    }

    public TensorOptions layout(Layout value) {
        return new TensorOptions(dtype, device, value, requiresGrad);
    }

    public TensorOptions requiresGrad(boolean value) {
        return new TensorOptions(dtype, device, layout, value);
    }

    @Override
    public String toString() {
        return "TensorOptions[dtype=" + dtype + ", device=" + device + ", layout=" + layout
                + ", requiresGrad=" + requiresGrad + "]";
    }
}
