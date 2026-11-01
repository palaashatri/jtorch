package org.jtorch.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DataTest {
    @TempDir
    Path temporary;

    @Test
    void tensorDatasetSelectsRows() {
        Tensor features = Torch.arange(0, 12).reshape(4, 3);
        Tensor labels = Torch.tensor(new long[] {0, 1, 2, 3}, Shape.of(4));
        TensorDataset dataset = new TensorDataset(features, labels);
        assertEquals(4, dataset.size());
        Sample sample = dataset.get(2);
        assertArrayEquals(new long[] {6, 7, 8}, sample.tensor(0).toLongArray());
        assertEquals(2, sample.tensor(1).itemLong());
        assertThrows(IndexOutOfBoundsException.class, () -> dataset.get(4));
        assertThrows(IllegalArgumentException.class,
                () -> new TensorDataset(Torch.zeros(Shape.of(2, 3)), Torch.zeros(Shape.of(3))));
    }

    @Test
    void dataLoaderBatchesAndCollates() {
        Tensor features = Torch.arange(0, 20).reshape(10, 2);
        Tensor labels = Torch.arange(0, 10);
        DataLoader loader = new DataLoader(new TensorDataset(features, labels), 4);
        List<Sample> batches = new ArrayList<>();
        loader.forEach(batches::add);
        assertEquals(3, batches.size());
        assertEquals(Shape.of(4, 2), batches.get(0).tensor(0).shape());
        assertEquals(Shape.of(4), batches.get(0).tensor(1).shape());
        assertEquals(Shape.of(2, 2), batches.get(2).tensor(0).shape());

        DataLoader dropLast = new DataLoader(new TensorDataset(features, labels), 4, false, true, 0);
        assertEquals(2, dropLast.batchCount());
    }

    @Test
    void shufflingIsDeterministicPerSeedAndEpoch() {
        Tensor data = Torch.arange(0, 20).reshape(10, 2);
        DataLoader loader = new DataLoader(new TensorDataset(data), 2, true, false, 123);
        List<Long> first = collectOrder(loader);
        List<Long> second = collectOrder(loader);
        assertEquals(first, second);
        loader.setEpoch(1);
        List<Long> third = collectOrder(loader);
        assertNotEquals(first, third);
        assertEquals(10, Set.copyOf(first).size());

        DataLoader otherSeed = new DataLoader(new TensorDataset(data), 2, true, false, 999);
        assertNotEquals(first, collectOrder(otherSeed));
    }

    private static List<Long> collectOrder(DataLoader loader) {
        List<Long> order = new ArrayList<>();
        for (Sample batch : loader) {
            for (long value : batch.tensor(0).select(1, 0).toLongArray()) {
                order.add(value);
            }
        }
        return order;
    }

    @Test
    void idxReaderParsesMnistHeaders() throws IOException {
        Path images = temporary.resolve("images.idx");
        ByteBuffer imageBuffer = ByteBuffer.allocate(16 + 8).order(ByteOrder.BIG_ENDIAN);
        imageBuffer.putInt(0x00000803);
        imageBuffer.putInt(2);
        imageBuffer.putInt(2);
        imageBuffer.putInt(2);
        imageBuffer.put(new byte[] {0, (byte) 255, (byte) 128, 64, 1, 2, 3, 4});
        Files.write(images, imageBuffer.array());
        Tensor parsed = IdxReader.readImages(images);
        assertEquals(Shape.of(2, 4), parsed.shape());
        assertArrayEquals(new float[] {0, 1, 128f / 255f, 64f / 255f, 1f / 255f, 2f / 255f, 3f / 255f,
                4f / 255f}, parsed.toFloatArray(), 1e-7f);

        Path labels = temporary.resolve("labels.idx");
        ByteBuffer labelBuffer = ByteBuffer.allocate(8 + 2).order(ByteOrder.BIG_ENDIAN);
        labelBuffer.putInt(0x00000801);
        labelBuffer.putInt(2);
        labelBuffer.put(new byte[] {3, 7});
        Files.write(labels, labelBuffer.array());
        Tensor parsedLabels = IdxReader.readLabels(labels);
        assertEquals(DType.INT64, parsedLabels.dtype());
        assertArrayEquals(new long[] {3, 7}, parsedLabels.toLongArray());
    }

    @Test
    void idxReaderRejectsCorruptFiles() throws IOException {
        Path badMagic = temporary.resolve("bad-magic.idx");
        Files.write(badMagic, new byte[16]);
        assertThrows(IOException.class, () -> IdxReader.readImages(badMagic));

        Path truncated = temporary.resolve("truncated.idx");
        Files.write(truncated, new byte[4]);
        assertThrows(IOException.class, () -> IdxReader.readLabels(truncated));

        Path wrongSize = temporary.resolve("wrong-size.idx");
        ByteBuffer buffer = ByteBuffer.allocate(16 + 3).order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(0x00000803);
        buffer.putInt(1);
        buffer.putInt(2);
        buffer.putInt(2);
        buffer.put(new byte[] {1, 2, 3});
        Files.write(wrongSize, buffer.array());
        assertThrows(IOException.class, () -> IdxReader.readImages(wrongSize));
    }

    @Test
    void syntheticDigitsAreDeterministicAndWellFormed() {
        TensorDataset first = SyntheticDigits.create(100, 42);
        TensorDataset second = SyntheticDigits.create(100, 42);
        assertEquals(100, first.size());
        assertEquals(Shape.of(100, SyntheticDigits.FEATURES), first.columns().get(0).shape());
        assertEquals(DType.INT64, first.columns().get(1).dtype());
        assertArrayEquals(first.columns().get(0).toFloatArray(), second.columns().get(0).toFloatArray(), 0f);
        for (long label : first.columns().get(1).toLongArray()) {
            assertTrue(label >= 0 && label < SyntheticDigits.CLASSES);
        }
        TensorDataset different = SyntheticDigits.create(100, 43);
        assertNotEquals(first.columns().get(0).toFloatArray()[0], different.columns().get(0).toFloatArray()[0]);
    }

    @Test
    void sampleEqualityUsesTensorIdentity() {
        Tensor tensor = Torch.zeros(Shape.of(1));
        assertEquals(new Sample(new Tensor[] {tensor}), new Sample(new Tensor[] {tensor}));
        assertNotEquals(new Sample(new Tensor[] {tensor}), new Sample(new Tensor[] {Torch.zeros(Shape.of(1))}));
    }

    @Test
    void sampleAccessorsAreDefensive() {
        Tensor[] tensors = {Torch.zeros(Shape.of(1))};
        Sample sample = new Sample(tensors);
        tensors[0] = null;
        assertEquals(1, sample.arity());
        sample.tensors()[0] = null;
        assertEquals(1, sample.arity());
    }
}
