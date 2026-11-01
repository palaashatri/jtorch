package org.jtorch.data;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;

/** Reader for the MNIST/IDX binary format (no pickle, bounded allocation, strict header validation). */
public final class IdxReader {
    private static final int IMAGE_MAGIC = 0x00000803;
    private static final int LABEL_MAGIC = 0x00000801;

    private IdxReader() {}

    public static Tensor readImages(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        if (bytes.length < 16) {
            throw new IOException("IDX image file is truncated: " + path);
        }
        int magic = buffer.getInt();
        if (magic != IMAGE_MAGIC) {
            throw new IOException("Not an IDX image file (magic 0x" + Integer.toHexString(magic) + "): " + path);
        }
        int count = buffer.getInt();
        int rows = buffer.getInt();
        int columns = buffer.getInt();
        if (count < 0 || rows <= 0 || columns <= 0) {
            throw new IOException("Invalid IDX image dimensions: count=" + count + " rows=" + rows + " cols=" + columns);
        }
        long expected = 16L + (long) count * rows * columns;
        if (bytes.length != expected) {
            throw new IOException("IDX image file has " + bytes.length + " bytes but the header requires " + expected);
        }
        float[] values = new float[count * rows * columns];
        for (int i = 0; i < values.length; i++) {
            values[i] = (bytes[16 + i] & 0xFF) / 255.0f;
        }
        return Torch.tensor(values, Shape.of(count, (long) rows * columns));
    }

    public static Tensor readLabels(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        if (bytes.length < 8) {
            throw new IOException("IDX label file is truncated: " + path);
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        int magic = buffer.getInt();
        if (magic != LABEL_MAGIC) {
            throw new IOException("Not an IDX label file (magic 0x" + Integer.toHexString(magic) + "): " + path);
        }
        int count = buffer.getInt();
        if (count < 0 || bytes.length != 8 + count) {
            throw new IOException("IDX label file size does not match its header: " + path);
        }
        long[] labels = new long[count];
        for (int i = 0; i < count; i++) {
            labels[i] = bytes[8 + i] & 0xFF;
        }
        return Torch.tensor(labels, Shape.of(count), TensorOptions.defaults().dtype(DType.INT64));
    }
}
