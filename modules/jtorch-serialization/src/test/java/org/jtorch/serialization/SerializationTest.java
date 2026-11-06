package org.jtorch.serialization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.jtorch.json.Json;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SerializationTest {
    @TempDir
    Path temporary;

    private static Map<String, Tensor> samples() {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        tensors.put("float32", Torch.tensor(new float[] {1.5f, -2.25f, 3f, 0f}, Shape.of(2, 2)));
        tensors.put("float64", Torch.tensor(new double[] {Math.PI, -Math.E}, Shape.of(2)));
        tensors.put("int64", Torch.tensor(new long[] {1L << 40, -7}, Shape.of(2)));
        tensors.put("int32", Torch.tensor(new int[] {1, -2, 3}, Shape.of(3)));
        tensors.put("bool", Torch.tensor(new boolean[] {true, false, true}, Shape.of(3)));
        tensors.put("scalar", Torch.tensor(new float[] {42f}, Shape.scalar()));
        tensors.put("empty", Torch.zeros(Shape.of(0, 3)));
        return tensors;
    }

    @Test
    void checkpointRoundTripsEverySupportedDtype() {
        Map<String, Tensor> original = samples();
        Path path = temporary.resolve("model.jtch");
        Checkpoint.save(path, original, Map.of("app", "test", "epoch", "3"));
        Checkpoint loaded = Checkpoint.load(path);
        assertEquals(original.keySet(), loaded.tensors().keySet());
        assertEquals("test", loaded.metadata().get("app"));
        assertEquals("3", loaded.metadata().get("epoch"));
        for (Map.Entry<String, Tensor> entry : original.entrySet()) {
            Tensor actual = loaded.tensors().get(entry.getKey());
            assertEquals(entry.getValue().dtype(), actual.dtype(), entry.getKey());
            assertEquals(entry.getValue().shape(), actual.shape(), entry.getKey());
            assertArrayEquals(asDoubles(entry.getValue()), asDoubles(actual), 0, entry.getKey());
        }
    }

    @Test
    void checkpointBytesAreDeterministic() throws IOException {
        Path first = temporary.resolve("a.jtch");
        Path second = temporary.resolve("b.jtch");
        Checkpoint.save(first, samples(), Map.of("b", "2", "a", "1"));
        Checkpoint.save(second, samples(), Map.of("a", "1", "b", "2"));
        assertArrayEquals(Files.readAllBytes(first), Files.readAllBytes(second));
    }

    @Test
    void checkpointDetectsCorruptionTruncationAndBadMagic() throws IOException {
        Path path = temporary.resolve("good.jtch");
        Checkpoint.save(path, samples(), Map.of());
        byte[] bytes = Files.readAllBytes(path);

        byte[] corrupted = bytes.clone();
        corrupted[corrupted.length - 1] ^= 0x01;
        Path corruptPath = temporary.resolve("corrupt.jtch");
        Files.write(corruptPath, corrupted);
        SerializationException checksum = assertThrows(SerializationException.class, () -> Checkpoint.load(corruptPath));
        assertTrue(checksum.getMessage().contains("checksum"), checksum.getMessage());

        Path truncated = temporary.resolve("truncated.jtch");
        Files.write(truncated, java.util.Arrays.copyOf(bytes, bytes.length - 10));
        assertThrows(SerializationException.class, () -> Checkpoint.load(truncated));

        byte[] badMagic = bytes.clone();
        badMagic[0] = 'X';
        Path badMagicPath = temporary.resolve("bad-magic.jtch");
        Files.write(badMagicPath, badMagic);
        assertThrows(SerializationException.class, () -> Checkpoint.load(badMagicPath));
    }

    @Test
    void safetensorsRoundTripsEverySupportedDtype() {
        Map<String, Tensor> original = samples();
        Path path = temporary.resolve("model.safetensors");
        Safetensors.save(original, path);
        Map<String, Tensor> loaded = Safetensors.load(path);
        assertEquals(original.keySet(), loaded.keySet());
        for (Map.Entry<String, Tensor> entry : original.entrySet()) {
            Tensor actual = loaded.get(entry.getKey());
            assertEquals(entry.getValue().dtype(), actual.dtype(), entry.getKey());
            assertEquals(entry.getValue().shape(), actual.shape(), entry.getKey());
            assertArrayEquals(asDoubles(entry.getValue()), asDoubles(actual), 0, entry.getKey());
        }
    }

    @Test
    void safetensorsReadsTheReferencePythonFile() throws IOException {
        Path path = temporary.resolve("reference.safetensors");
        try (InputStream stream = getClass().getResourceAsStream("/golden/reference.safetensors")) {
            if (stream == null) {
                throw new IOException("missing reference fixture");
            }
            Files.write(path, stream.readAllBytes());
        }
        Map<String, Tensor> loaded = Safetensors.load(path);
        Map<String, Object> expected = Json.parseObject(read("/golden/reference.json"));
        for (Map.Entry<String, Object> entry : Json.object(expected.get("tensors")).entrySet()) {
            Map<String, Object> description = Json.object(entry.getValue());
            Tensor tensor = loaded.get(entry.getKey());
            assertEquals(Json.string(description.get("dtype")), tensor.dtype().typeName());
            long[] shape = Json.array(description.get("shape")).stream()
                    .mapToLong(Json::longValue).toArray();
            assertArrayEquals(shape, tensor.shape().toArray());
            if (tensor.dtype() == DType.BOOL) {
                boolean[] values = new boolean[Json.array(description.get("data")).size()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = Json.booleanValue(Json.array(description.get("data")).get(i));
                }
                assertArrayEquals(values, tensor.toBooleanArray(), entry.getKey());
            } else {
                double[] values = Json.array(description.get("data")).stream()
                        .mapToDouble(Json::doubleValue).toArray();
                assertArrayEquals(values, asDoubles(tensor), 0, entry.getKey());
            }
        }
    }

    @Test
    void safetensorsRejectsLengthMismatches() throws IOException {
        Path path = temporary.resolve("good.safetensors");
        Safetensors.save(samples(), path);
        byte[] bytes = Files.readAllBytes(path);

        Path extra = temporary.resolve("extra.safetensors");
        byte[] extended = java.util.Arrays.copyOf(bytes, bytes.length + 1);
        Files.write(extra, extended);
        assertThrows(SerializationException.class, () -> Safetensors.load(extra));

        Path shortFile = temporary.resolve("short.safetensors");
        Files.write(shortFile, java.util.Arrays.copyOf(bytes, bytes.length - 1));
        assertThrows(SerializationException.class, () -> Safetensors.load(shortFile));
    }

    @Test
    void nonContiguousTensorsAreSavedInLogicalOrder() {
        Tensor base = Torch.tensor(new float[] {1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
        Tensor transposed = base.t();
        Path path = temporary.resolve("transposed.safetensors");
        Safetensors.save(Map.of("t", transposed), path);
        Tensor loaded = Safetensors.load(path).get("t");
        assertArrayEquals(new double[] {1, 4, 2, 5, 3, 6}, loaded.toDoubleArray(), 0);
        assertTrue(loaded.isContiguous());
    }

    @Test
    void emptyCheckpointIsValid() {
        Path path = temporary.resolve("empty.jtch");
        Checkpoint.save(path, Map.of(), Map.of());
        Checkpoint loaded = Checkpoint.load(path);
        assertTrue(loaded.tensors().isEmpty());
        assertTrue(loaded.metadata().isEmpty());
    }

    @Test
    void dtypeMetadataIsStable() {
        assertEquals(DType.FLOAT32, TensorBytes.fromSafetensorsName("F32"));
        assertEquals(DType.FLOAT64, TensorBytes.fromSafetensorsName("F64"));
        assertEquals(DType.INT64, TensorBytes.fromSafetensorsName("I64"));
        assertEquals(DType.INT32, TensorBytes.fromSafetensorsName("I32"));
        assertEquals(DType.BOOL, TensorBytes.fromSafetensorsName("BOOL"));
        assertThrows(SerializationException.class, () -> TensorBytes.fromSafetensorsName("BF16"));
        assertNotEquals("F32", TensorBytes.safetensorsName(DType.FLOAT64));
    }

    private static double[] asDoubles(Tensor tensor) {
        return switch (tensor.dtype()) {
            case BOOL -> {
                boolean[] values = tensor.toBooleanArray();
                double[] result = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    result[i] = values[i] ? 1 : 0;
                }
                yield result;
            }
            case INT32, INT64 -> {
                long[] values = tensor.toLongArray();
                double[] result = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    result[i] = values[i];
                }
                yield result;
            }
            case FLOAT32 -> {
                float[] values = tensor.toFloatArray();
                double[] result = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    result[i] = values[i];
                }
                yield result;
            }
            case FLOAT64 -> tensor.toDoubleArray();
            case COMPLEX64, COMPLEX128 -> throw new IllegalArgumentException(
                    "Serialization test helper does not accept complex tensors");
        };
    }

    private static String read(String resource) throws IOException {
        try (InputStream stream = SerializationTest.class.getResourceAsStream(resource)) {
            if (stream == null) {
                throw new IOException("Missing resource " + resource);
            }
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
