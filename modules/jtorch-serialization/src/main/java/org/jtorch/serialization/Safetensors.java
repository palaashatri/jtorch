package org.jtorch.serialization;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.json.Json;

/**
 * Safetensors reader and writer. The parser validates header bounds, duplicate keys, contiguous offsets, shape
 * consistency and exact file length before allocating any tensor; it never deserialises code.
 */
public final class Safetensors {
    private static final int MAX_HEADER_BYTES = 100 * 1024 * 1024;

    private Safetensors() {}

    public static void save(Map<String, Tensor> tensors, Path path) {
        TreeMap<String, Tensor> ordered = new TreeMap<>(tensors);
        Map<String, Object> header = new LinkedHashMap<>();
        List<byte[]> payloads = new ArrayList<>();
        long offset = 0;
        for (Map.Entry<String, Tensor> entry : ordered.entrySet()) {
            Tensor tensor = entry.getValue();
            if (!tensor.device().isCpu()) {
                throw new SerializationException("Only CPU tensors can be saved, got " + tensor.device());
            }
            byte[] data = TensorBytes.encode(tensor);
            Map<String, Object> description = new LinkedHashMap<>();
            description.put("dtype", TensorBytes.safetensorsName(tensor.dtype()));
            description.put("shape", toList(tensor.shape()));
            description.put("data_offsets", List.of(offset, offset + data.length));
            header.put(entry.getKey(), description);
            payloads.add(data);
            offset += data.length;
            if (offset > Integer.MAX_VALUE - 8) {
                throw new SerializationException("Safetensors payload is too large for this build");
            }
        }
        byte[] headerBytes = Json.write(header).getBytes(StandardCharsets.UTF_8);
        int padding = (8 - headerBytes.length % 8) % 8;
        byte[] padded = new byte[headerBytes.length + padding];
        System.arraycopy(headerBytes, 0, padded, 0, headerBytes.length);
        for (int i = 0; i < padding; i++) {
            padded[headerBytes.length + i] = ' ';
        }
        try {
            ByteBuffer file = ByteBuffer.allocate(8 + padded.length + (int) offset).order(ByteOrder.LITTLE_ENDIAN);
            file.putLong(padded.length);
            file.put(padded);
            for (byte[] payload : payloads) {
                file.put(payload);
            }
            java.nio.file.Path parent = path.toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(path, file.array());
        } catch (IOException error) {
            throw new SerializationException("Unable to write Safetensors file " + path, error);
        }
    }

    public static Map<String, Tensor> load(Path path) {
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(path);
        } catch (IOException error) {
            throw new SerializationException("Unable to read Safetensors file " + path, error);
        }
        if (bytes.length < 8) {
            throw new SerializationException("Safetensors file is truncated: " + path);
        }
        long headerLength = ByteBuffer.wrap(bytes, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
        if (headerLength <= 0 || headerLength > MAX_HEADER_BYTES || headerLength > bytes.length - 8) {
            throw new SerializationException("Invalid Safetensors header length " + headerLength);
        }
        String headerText = new String(bytes, 8, (int) headerLength, StandardCharsets.UTF_8);
        Map<String, Object> header;
        try {
            header = Json.parseObject(headerText);
        } catch (RuntimeException error) {
            throw new SerializationException("Malformed Safetensors header", error);
        }
        int dataStart = 8 + (int) headerLength;
        long dataLength = bytes.length - (long) dataStart;
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        List<Entry> entries = new ArrayList<>();
        for (Map.Entry<String, Object> entry : header.entrySet()) {
            if (entry.getKey().equals("__metadata__")) {
                continue;
            }
            Map<String, Object> description = Json.object(entry.getValue());
            DType dtype = TensorBytes.fromSafetensorsName(Json.string(Json.require(description, "dtype")));
            List<Object> rawShape = Json.array(Json.require(description, "shape"));
            List<Long> dims = new ArrayList<>(rawShape.size());
            for (Object dimension : rawShape) {
                long value = Json.longValue(dimension);
                if (value < 0) {
                    throw new SerializationException("Negative dimension for tensor '" + entry.getKey() + "'");
                }
                dims.add(value);
            }
            Shape shape = Shape.of(dims);
            List<Object> offsets = Json.array(Json.require(description, "data_offsets"));
            if (offsets.size() != 2) {
                throw new SerializationException("data_offsets must contain two entries");
            }
            long begin = Json.longValue(offsets.get(0));
            long end = Json.longValue(offsets.get(1));
            long required = shape.numel() * dtype.sizeBytes();
            if (begin < 0 || end < begin || end - begin != required) {
                throw new SerializationException("Tensor '" + entry.getKey() + "' declares invalid byte offsets ["
                        + begin + ", " + end + ") for " + required + " bytes");
            }
            if (end > dataLength) {
                throw new SerializationException("Tensor '" + entry.getKey() + "' extends past the file end");
            }
            if (required > Integer.MAX_VALUE) {
                throw new SerializationException("Tensor '" + entry.getKey() + "' is too large for this build");
            }
            entries.add(new Entry(entry.getKey(), dtype, shape, begin, end));
        }
        entries.sort(java.util.Comparator.comparingLong(Entry::begin));
        long expectedOffset = 0;
        for (Entry entry : entries) {
            if (entry.begin() != expectedOffset) {
                throw new SerializationException("Safetensors offsets must be contiguous; '" + entry.name()
                        + "' starts at " + entry.begin() + " but " + expectedOffset + " was expected");
            }
            tensors.put(entry.name(), TensorBytes.decode(bytes, dataStart + (int) entry.begin(),
                    (int) (entry.end() - entry.begin()), entry.dtype(), entry.shape()));
            expectedOffset = entry.end();
        }
        if (expectedOffset != dataLength) {
            throw new SerializationException(
                    "Safetensors payload has " + dataLength + " bytes but the header covers " + expectedOffset);
        }
        return tensors;
    }

    private record Entry(String name, DType dtype, Shape shape, long begin, long end) {}

    private static List<Long> toList(Shape shape) {
        List<Long> dims = new ArrayList<>(shape.rank());
        for (long dim : shape.toArray()) {
            dims.add(dim);
        }
        return dims;
    }
}
