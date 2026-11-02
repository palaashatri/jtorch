package org.jtorch.core.memory;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.atomic.AtomicBoolean;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.JTorchException;

/**
 * A host tensor backed by a memory-mapped file. The file must already contain exactly {@code elements * dtype}
 * bytes; truncation is rejected instead of being allowed to fault later.
 *
 * <p>The mapping owns an FFM {@link Arena}. Callers that need to release the mapping deterministically (notably on
 * Windows, where an open mapping prevents file deletion) should use {@link #close()}. Closing invalidates every
 * tensor and view that still references this storage.
 */
public final class MappedStorage implements Storage, AutoCloseable {
    private final Arena arena;
    private final MemorySegment segment;
    private final DType dtype;
    private final long byteSize;
    private final Path path;
    private final AtomicBoolean closed = new AtomicBoolean();

    private MappedStorage(Arena arena, MemorySegment segment, DType dtype, long byteSize, Path path) {
        this.arena = arena;
        this.segment = segment;
        this.dtype = dtype;
        this.byteSize = byteSize;
        this.path = path;
    }

    public static MappedStorage map(Path path, DType dtype, long elements, boolean writable) {
        long expected;
        try {
            expected = Math.multiplyExact(elements, (long) dtype.sizeBytes());
        } catch (ArithmeticException error) {
            throw new JTorchException("Mapped tensor size overflows a 64-bit byte count");
        }
        if (expected == 0) {
            throw new JTorchException("Cannot map an empty file region for " + path);
        }
        Arena arena = Arena.ofShared();
        try {
            long actual = Files.size(path);
            if (actual != expected) {
                throw new JTorchException("Mapped file " + path + " has " + actual + " bytes but the tensor needs "
                        + expected);
            }
            FileChannel.MapMode mode =
                    writable ? FileChannel.MapMode.READ_WRITE : FileChannel.MapMode.READ_ONLY;
            MemorySegment segment;
            try (FileChannel channel = writable
                    ? FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE)
                    : FileChannel.open(path, StandardOpenOption.READ)) {
                segment = channel.map(mode, 0, expected, arena);
            }
            return new MappedStorage(arena, segment, dtype, expected, path);
        } catch (IOException error) {
            arena.close();
            throw new JTorchException("Unable to map " + path, error);
        } catch (RuntimeException | Error error) {
            arena.close();
            throw error;
        }
    }

    /** Flushes dirty pages to the backing file. */
    public void flush() {
        ensureOpen();
        segment.force();
    }

    /** Releases the mapping. Any later tensor access through this storage fails with {@link IllegalStateException}. */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            arena.close();
        }
    }

    public boolean isClosed() {
        return closed.get();
    }

    public Path path() {
        return path;
    }

    @Override
    public long byteSize() {
        ensureOpen();
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
        return StorageKind.MAPPED;
    }

    @Override
    public MemorySegment segment() {
        ensureOpen();
        return segment;
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("Mapped storage for " + path + " has been closed");
        }
    }
}
