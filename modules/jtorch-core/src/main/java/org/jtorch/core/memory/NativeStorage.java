package org.jtorch.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicLong;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.JTorchException;

/**
 * Aligned FFM host allocation with debug accounting.
 *
 * <p>The arena is closed by a {@link Cleaner} once the storage object is unreachable, so views, autograd saved
 * tensors and queued work that still reference the storage keep it alive. The allocator never ties an arena to a
 * lexical scope that a tensor could outlive.
 */
public final class NativeStorage implements Storage {
    private static final Cleaner CLEANER = Cleaner.create();
    private static final AtomicLong LIVE_BYTES = new AtomicLong();
    private static final AtomicLong LIVE_ALLOCATIONS = new AtomicLong();
    private static final AtomicLong TOTAL_BYTES = new AtomicLong();

    private final MemorySegment segment;
    private final DType dtype;
    private final long byteSize;

    private NativeStorage(MemorySegment segment, DType dtype, long byteSize) {
        this.segment = segment;
        this.dtype = dtype;
        this.byteSize = byteSize;
    }

    public static NativeStorage allocate(DType dtype, long elements) {
        if (elements < 0) {
            throw new JTorchException("Cannot allocate a negative number of elements: " + elements);
        }
        long bytes;
        try {
            bytes = Math.multiplyExact(elements, (long) dtype.sizeBytes());
        } catch (ArithmeticException error) {
            throw new JTorchException(
                    "Allocation of " + elements + " " + dtype + " elements overflows a 64-bit byte count");
        }
        Arena arena = Arena.ofShared();
        MemorySegment segment;
        try {
            segment = bytes == 0 ? arena.allocate(1, 64) : arena.allocate(bytes, 64);
        } catch (OutOfMemoryError error) {
            arena.close();
            throw new JTorchException("Unable to allocate " + bytes + " bytes for a " + dtype + " tensor", error);
        }
        LIVE_BYTES.addAndGet(bytes);
        LIVE_ALLOCATIONS.incrementAndGet();
        TOTAL_BYTES.addAndGet(bytes);
        NativeStorage storage = new NativeStorage(segment, dtype, bytes);
        // Register the returned instance, not a temporary: the arena must stay alive while any tensor can reach it.
        CLEANER.register(storage, new Release(arena, bytes));
        return storage;
    }

    @Override
    public long byteSize() {
        return byteSize;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public Device device() {
        return Device.cpu();
    }

    @Override
    public StorageKind kind() {
        return StorageKind.NATIVE;
    }

    @Override
    public MemorySegment segment() {
        return segment;
    }

    public static long liveBytes() {
        return LIVE_BYTES.get();
    }

    public static long liveAllocations() {
        return LIVE_ALLOCATIONS.get();
    }

    public static long totalAllocatedBytes() {
        return TOTAL_BYTES.get();
    }

    private static final class Release implements Runnable {
        private final Arena arena;
        private final long bytes;

        Release(Arena arena, long bytes) {
            this.arena = arena;
            this.bytes = bytes;
        }

        @Override
        public void run() {
            arena.close();
            LIVE_BYTES.addAndGet(-bytes);
            LIVE_ALLOCATIONS.decrementAndGet();
        }
    }
}
