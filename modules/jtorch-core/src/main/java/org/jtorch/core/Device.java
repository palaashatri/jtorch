package org.jtorch.core;

import java.util.Locale;
import java.util.Objects;

/** A compute device. The reference build implements CPU and meta tensors. */
public final class Device {
    public enum Type {
        CPU,
        META,
        CUDA,
        ROCM,
        MPS,
        VULKAN
    }

    private static final Device CPU = new Device(Type.CPU, 0);
    private static final Device META = new Device(Type.META, 0);

    private final Type type;
    private final int index;

    private Device(Type type, int index) {
        if (index < 0) {
            throw new DeviceException("Device index must be non-negative");
        }
        this.type = type;
        this.index = index;
    }

    public static Device cpu() {
        return CPU;
    }

    public static Device meta() {
        return META;
    }

    public static Device cuda(int index) {
        return new Device(Type.CUDA, index);
    }

    public static Device of(Type type, int index) {
        return switch (type) {
            case CPU -> CPU;
            case META -> META;
            default -> new Device(type, index);
        };
    }

    public Type type() {
        return type;
    }

    public int index() {
        return index;
    }

    public boolean isCpu() {
        return type == Type.CPU;
    }

    public boolean isMeta() {
        return type == Type.META;
    }

    /** True when this build has a working backend for the device. */
    public boolean isAvailable() {
        return type == Type.CPU;
    }

    /** Meta is a shape-only device, not an execution backend. */
    public boolean isMetaDevice() {
        return type == Type.META;
    }

    public static Device parse(String text) {
        String[] parts = text.trim().toLowerCase(Locale.ROOT).split(":", 2);
        int index = parts.length == 2 ? Integer.parseInt(parts[1]) : 0;
        return switch (parts[0]) {
            case "cpu" -> CPU;
            case "meta" -> META;
            case "cuda" -> cuda(index);
            case "rocm" -> of(Type.ROCM, index);
            case "mps" -> of(Type.MPS, index);
            case "vulkan" -> of(Type.VULKAN, index);
            default -> throw new DeviceException("Unknown device '" + text + "'");
        };
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Device device && device.type == type && device.index == index;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, index);
    }

    @Override
    public String toString() {
        return switch (type) {
            case CPU -> "cpu";
            case META -> "meta";
            default -> type.name().toLowerCase(Locale.ROOT) + ":" + index;
        };
    }
}
