package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.jtorch.core.CpuAllocator;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.ShapeException;
import org.jtorch.core.Strides;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorImpl;
import org.jtorch.core.VersionCounter;
import org.jtorch.core.memory.MappedStorage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class StorageTest {
    @TempDir
    Path temporary;

    @Test
    void nativeAllocationsAreReleasedAfterCollection() throws InterruptedException {
        long baseline = CpuAllocator.liveBytes();
        Tensor large = Torch.zeros(Shape.of(1024, 1024));
        assertTrue(CpuAllocator.liveBytes() >= baseline + 1024 * 1024 * 4,
                "live bytes did not include the new tensor");
        large = null;
        long deadline = System.nanoTime() + 5_000_000_000L;
        while (CpuAllocator.liveBytes() > baseline && System.nanoTime() < deadline) {
            System.gc();
            Thread.sleep(20);
        }
        assertTrue(CpuAllocator.liveBytes() <= baseline,
                "native allocation leaked: baseline=" + baseline + " live=" + CpuAllocator.liveBytes());
    }

    @Test
    void zeroSizeTensorsAreValid() {
        Tensor empty = Torch.zeros(Shape.of(0, 3));
        assertEquals(0, empty.numel());
        assertEquals(0, empty.toDoubleArray().length);
        assertEquals(Shape.of(0, 3), empty.shape());
        assertEquals(Shape.of(1, 3), empty.sum(0, true).shape());
    }

    @Test
    void excessiveAllocationsAreRejectedBeforeOom() {
        assertThrows(org.jtorch.core.JTorchException.class,
                () -> CpuAllocator.allocate(DType.FLOAT64, Long.MAX_VALUE));
        assertThrows(org.jtorch.core.JTorchException.class,
                () -> CpuAllocator.allocate(DType.FLOAT32, -1));
        assertThrows(ShapeException.class, () -> Torch.zeros(Shape.of(Long.MAX_VALUE, 2)));
    }

    @Test
    void tensorBoundsAreValidatedAtConstruction() {
        Tensor base = Torch.zeros(Shape.of(4));
        assertThrows(ShapeException.class, () -> new TensorImpl(
                base.impl().storage(), 0, Shape.of(5), Strides.of(1), new VersionCounter(), false, false));
        assertThrows(ShapeException.class, () -> new TensorImpl(
                base.impl().storage(), 3, Shape.of(2), Strides.of(1), new VersionCounter(), false, true));
        assertThrows(ShapeException.class, () -> new TensorImpl(
                base.impl().storage(), -1, Shape.of(1), Strides.of(1), new VersionCounter(), false, true));
    }

    @Test
    void mappedStorageReadsAndFlushesFileBytes() throws IOException {
        Path file = temporary.resolve("mapped.bin");
        Files.write(file, new byte[2 * 3 * 4]);
        MappedStorage storage = MappedStorage.map(file, DType.FLOAT32, 6, true);
        Tensor tensor = Tensor.wrap(TensorImpl.fresh(storage, Shape.of(2, 3), Strides.contiguous(Shape.of(2, 3)),
                false));
        tensor.set(1.5, 0, 1);
        tensor.set(-2.0, 1, 2);
        storage.flush();
        byte[] bytes = Files.readAllBytes(file);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        assertEquals(0f, buffer.getFloat(0));
        assertEquals(1.5f, buffer.getFloat(4));
        assertEquals(-2.0f, buffer.getFloat(20));
    }

    @Test
    void mappedFileLengthMustMatchTheDeclaredShape() throws IOException {
        Path file = temporary.resolve("short.bin");
        Files.write(file, new byte[7]);
        assertThrows(org.jtorch.core.JTorchException.class,
                () -> MappedStorage.map(file, DType.FLOAT32, 2, true));
    }
}
