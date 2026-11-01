package org.jtorch.core.memory;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.JTorchException;

/**
 * A host tensor backed by a memory-mapped file. The file must already contain exactly {@code elements * dtype}
 * bytes; truncation is rejected instead of being allowed to fault later.
 */
public final class MappedStorage implements Storage {
    private final MappedByteBuffer buffer;
    private final MemorySegment segment;
    private final DType dtype;
    private final long byteSize;
    private final Path path;

    private MappedStorage(MappedByteBuffer buffer, DType dtype, long byteSize, Path path) {
        this.buffer = buffer;
        this.segment = MemorySegment.ofBuffer(buffer);
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
        try {
            long actual = java.nio.file.Files.size(path);
            if (actual != expected) {
                throw new JTorchException("Mapped file " + path + " has " + actual + " bytes but the tensor needs "
                        + expected);
            }
            FileChannel channel = writable
                    ? FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE)
                    : FileChannel.open(path, StandardOpenOption.READ);
            MappedByteBuffer buffer = channel.map(
                    writable ? FileChannel.MapMode.READ_WRITE : FileChannel.MapMode.READ_ONLY, 0, expected);
            channel.close();
            return new MappedStorage(buffer, dtype, expected, path);
        } catch (IOException error) {
            throw new JTorchException("Unable to map " + path, error);
        }
    }

    /** Flushes dirty pages to the backing file. */
    public void flush() {
        buffer.force();
    }

    public Path path() {
        return path;
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
        return StorageKind.MAPPED;
    }

    @Override
    public MemorySegment segment() {
        return segment;
    }
}
