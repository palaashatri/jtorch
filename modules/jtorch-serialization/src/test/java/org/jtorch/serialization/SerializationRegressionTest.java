package org.jtorch.serialization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.json.Json;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Regression tests for serialization defects found by hostile QA. */
class SerializationRegressionTest {
    @TempDir
    Path temporary;

    @Test
    void checkpointRejectsOverflowingOffset() throws Exception {
        Map<String, Object> tensor = new LinkedHashMap<>();
        tensor.put("name", "x");
        tensor.put("dtype", "float32");
        tensor.put("shape", List.of(2L));
        tensor.put("offset", Long.MAX_VALUE - 1);
        tensor.put("length", 8L);
        Map<String, Object> header = new LinkedHashMap<>();
        header.put("format", "jtorch-checkpoint");
        header.put("version", 1L);
        header.put("metadata", Map.of());
        header.put("tensors", List.of(tensor));
        byte[] headerBytes = Json.write(header).getBytes(StandardCharsets.UTF_8);
        ByteBuffer file = ByteBuffer.allocate(8 + 8 + headerBytes.length + 8 + 32)
                .order(ByteOrder.LITTLE_ENDIAN);
        file.put("JTORCH01".getBytes(StandardCharsets.US_ASCII));
        file.putLong(headerBytes.length);
        file.put(headerBytes);
        file.put(new byte[8]);
        byte[] body = file.array();
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        digest.update(body, 0, body.length - 32);
        System.arraycopy(digest.digest(), 0, body, body.length - 32, 32);
        Path path = temporary.resolve("overflow.jtch");
        Files.write(path, body);
        assertThrows(SerializationException.class, () -> Checkpoint.load(path));
    }

    @Test
    void safetensorsAcceptsAnyHeaderKeyOrder() throws Exception {
        Map<String, Object> a = new LinkedHashMap<>();
        a.put("dtype", "F32");
        a.put("shape", List.of(2L, 2L));
        a.put("data_offsets", List.of(0L, 16L));
        Map<String, Object> b = new LinkedHashMap<>();
        b.put("dtype", "I64");
        b.put("shape", List.of(2L));
        b.put("data_offsets", List.of(16L, 32L));
        Map<String, Object> header = new LinkedHashMap<>();
        header.put("b", b);
        header.put("a", a);
        byte[] headerBytes = Json.write(header).getBytes(StandardCharsets.UTF_8);
        byte[] padded = new byte[(headerBytes.length + 7) / 8 * 8];
        System.arraycopy(headerBytes, 0, padded, 0, headerBytes.length);
        for (int i = headerBytes.length; i < padded.length; i++) {
            padded[i] = ' ';
        }
        ByteBuffer data = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        data.putFloat(1).putFloat(2).putFloat(3).putFloat(4);
        data.putLong(5).putLong(6);
        ByteBuffer file = ByteBuffer.allocate(8 + padded.length + 32).order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(padded.length);
        file.put(padded);
        file.put(data.array());
        Path path = temporary.resolve("reordered.safetensors");
        Files.write(path, file.array());
        Map<String, Tensor> loaded = Safetensors.load(path);
        assertArrayEquals(new float[] {1, 2, 3, 4}, loaded.get("a").toFloatArray(), 0f);
        assertArrayEquals(new long[] {5, 6}, loaded.get("b").toLongArray());
        assertEquals(DType.INT64, loaded.get("b").dtype());
        assertEquals(Shape.of(2, 2), loaded.get("a").shape());
    }
}
