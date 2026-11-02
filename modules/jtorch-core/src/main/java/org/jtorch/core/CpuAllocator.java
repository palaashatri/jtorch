package org.jtorch.core;

import org.jtorch.core.memory.NativeStorage;

/** Entry points for host allocation with overflow and lifetime accounting. */
public final class CpuAllocator {
    private CpuAllocator() {}

    public static NativeStorage allocate(DType dtype, long elements) {
        return NativeStorage.allocate(dtype, elements);
    }

    public static long liveBytes() {
        return NativeStorage.liveBytes();
    }

    public static long liveAllocations() {
        return NativeStorage.liveAllocations();
    }

    public static long totalAllocatedBytes() {
        return NativeStorage.totalAllocatedBytes();
    }
}
