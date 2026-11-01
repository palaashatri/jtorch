package org.jtorch.serialization;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
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
 * JTorch's native checkpoint format: a versioned JSON manifest, little-endian raw tensor payloads and a trailing
 * SHA-256 digest. Loading validates every bound and checksum before constructing tensors and never executes code.
 */
public final class Checkpoint {
    private static final byte[] MAGIC = "JTORCH01".getBytes(StandardCharsets.US_ASCII);
    private static final long MAX_FILE_BYTES = 2L * 1024 * 1024 * 1024;

    private final Map<String, Tensor> tensors;
    private final Map<String, String> metadata;

    private Checkpoint(Map<String, Tensor> tensors, Map<String, String> metadata) {
        this.tensors = Map.copyOf(tensors);
        this.metadata = Map.copyOf(metadata);
    }

    public static void save(Path path, Map<String, Tensor> tensors, Map<String, String> metadata) {
        TreeMap<String, Tensor> ordered = new TreeMap<>(tensors);
        List<byte[]> payloads = new ArrayList<>();
        List<Object> entries = new ArrayList<>();
        long offset = 0;
        for (Map.Entry<String, Tensor> entry : ordered.entrySet()) {
            Tensor tensor = entry.getValue();
            if (!tensor.device().isCpu()) {
                throw new SerializationException("Only CPU tensors can be saved");
            }
            byte[] payload = TensorBytes.encode(tensor);
            Map<String, Object> description = new LinkedHashMap<>();
            description.put("name", entry.getKey());
            description.put("dtype", tensor.dtype().typeName());
            description.put("shape", toList(tensor.shape()));
            description.put("offset", offset);
            description.put("length", (long) payload.length);
            entries.add(description);
            payloads.add(payload);
            offset += payload.length;
            if (offset > Integer.MAX_VALUE - 8 - 32) {
                throw new SerializationException("Checkpoint payload is too large for this build");
            }
        }
        Map<String, Object> header = new LinkedHashMap<>();
        header.put("format", "jtorch-checkpoint");
        header.put("version", 1L);
        header.put("metadata", new TreeMap<>(metadata));
        header.put("tensors", entries);
        byte[] headerBytes = Json.write(header).getBytes(StandardCharsets.UTF_8);
        long payloadLength = offset;

        try {
            ByteBuffer file = ByteBuffer.allocate(
                            MAGIC.length + 8 + headerBytes.length + (int) payloadLength + 32)
                    .order(ByteOrder.LITTLE_ENDIAN);
            file.put(MAGIC);
            file.putLong(headerBytes.length);
            file.put(headerBytes);
            for (byte[] payload : payloads) {
                file.put(payload);
            }
            byte[] body = file.array();
            byte[] digest = sha256(body, 0, body.length - 32);
            System.arraycopy(digest, 0, body, body.length - 32, 32);
            Path parent = path.toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(path, body);
        } catch (IOException error) {
            throw new SerializationException("Unable to write checkpoint " + path, error);
        }
    }

    public static Checkpoint load(Path path) {
        byte[] bytes;
        try {
            long size = Files.size(path);
            if (size > MAX_FILE_BYTES) {
                throw new SerializationException("Checkpoint is too large: " + size + " bytes");
            }
            bytes = Files.readAllBytes(path);
        } catch (IOException error) {
            throw new SerializationException("Unable to read checkpoint " + path, error);
        }
        int minimum = MAGIC.length + 8 + 32;
        if (bytes.length < minimum) {
            throw new SerializationException("Checkpoint is truncated");
        }
        for (int i = 0; i < MAGIC.length; i++) {
            if (bytes[i] != MAGIC[i]) {
                throw new SerializationException("Not a JTorch checkpoint (bad magic)");
            }
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes, MAGIC.length, 8).order(ByteOrder.LITTLE_ENDIAN);
        long headerLength = buffer.getLong();
        if (headerLength <= 0 || headerLength > bytes.length - minimum) {
            throw new SerializationException("Invalid checkpoint header length " + headerLength);
        }
        int headerStart = MAGIC.length + 8;
        int payloadStart = headerStart + (int) headerLength;
        int payloadEnd = bytes.length - 32;
        byte[] expectedDigest = sha256(bytes, 0, payloadEnd);
        for (int i = 0; i < 32; i++) {
            if (expectedDigest[i] != bytes[payloadEnd + i]) {
                throw new SerializationException("Checkpoint checksum mismatch");
            }
        }
        Map<String, Object> header;
        try {
            header = Json.parseObject(new String(bytes, headerStart, (int) headerLength, StandardCharsets.UTF_8));
        } catch (RuntimeException error) {
            throw new SerializationException("Malformed checkpoint header", error);
        }
        if (!Json.string(Json.require(header, "format")).equals("jtorch-checkpoint")) {
            throw new SerializationException("Unsupported checkpoint format");
        }
        if (Json.intValue(Json.require(header, "version")) != 1) {
            throw new SerializationException("Unsupported checkpoint version");
        }
        Map<String, String> metadata = new LinkedHashMap<>();
        if (header.containsKey("metadata")) {
            for (Map.Entry<String, Object> entry : Json.object(header.get("metadata")).entrySet()) {
                metadata.put(entry.getKey(), Json.string(entry.getValue()));
            }
        }
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        int payloadLength = payloadEnd - payloadStart;
        for (Object raw : Json.array(Json.require(header, "tensors"))) {
            Map<String, Object> description = Json.object(raw);
            String name = Json.string(Json.require(description, "name"));
            if (tensors.containsKey(name)) {
                throw new SerializationException("Duplicate tensor name '" + name + "'");
            }
            DType dtype = DType.parse(Json.string(Json.require(description, "dtype")));
            List<Long> dims = new ArrayList<>();
            for (Object dimension : Json.array(Json.require(description, "shape"))) {
                long value = Json.longValue(dimension);
                if (value < 0) {
                    throw new SerializationException("Negative dimension for tensor '" + name + "'");
                }
                dims.add(value);
            }
            long offset = Json.longValue(Json.require(description, "offset"));
            long length = Json.longValue(Json.require(description, "length"));
            if (offset < 0 || length < 0 || offset > payloadLength || length > payloadLength - offset) {
                throw new SerializationException("Tensor '" + name + "' extends past the payload");
            }
            if (length > Integer.MAX_VALUE) {
                throw new SerializationException("Tensor '" + name + "' is too large for this build");
            }
            Shape shape = Shape.of(dims);
            tensors.put(name, TensorBytes.decode(bytes, payloadStart + (int) offset, (int) length, dtype, shape));
        }
        return new Checkpoint(tensors, metadata);
    }

    public Map<String, Tensor> tensors() {
        return Map.copyOf(tensors);
    }

    public Map<String, String> metadata() {
        return Map.copyOf(metadata);
    }

    private static List<Long> toList(Shape shape) {
        List<Long> dims = new ArrayList<>(shape.rank());
        for (long dim : shape.toArray()) {
            dims.add(dim);
        }
        return dims;
    }

    private static byte[] sha256(byte[] data, int offset, int length) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(data, offset, length);
            return digest.digest();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is required by the Java platform", error);
        }
    }
}
