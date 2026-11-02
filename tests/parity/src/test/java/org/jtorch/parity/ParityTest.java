package org.jtorch.parity;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.Op;
import org.jtorch.core.Scalar;
import org.jtorch.core.Shape;
import org.jtorch.core.Strides;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorImpl;
import org.jtorch.core.TensorOptions;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.generated.Schemas;
import org.jtorch.core.memory.TensorAccess;
import org.jtorch.core.value.Value;
import org.jtorch.json.Json;
import org.junit.jupiter.api.Test;

/**
 * Differential parity runner.
 *
 * <p>Every fixture replays the same schema-ordered arguments through the dispatcher, compares forward values,
 * shapes, strides and dtypes against the pinned PyTorch oracle, then compares gradients of
 * {@code sum(floating outputs)}. The same call is replayed on meta tensors to require meta and CPU metadata to
 * agree.
 */
class ParityTest {

    @Test
    void everyFixtureMatchesThePinnedOracle() throws IOException {
        Map<String, Object> index = Json.parseObject(read("/golden/index.json"));
        List<Object> cases = Json.array(index.get("cases"));
        List<String> failures = new ArrayList<>();
        for (Object raw : cases) {
            Map<String, Object> entry = Json.object(raw);
            String file = Json.string(entry.get("file"));
            try {
                checkFixture(file);
            } catch (Throwable error) {
                if (failures.isEmpty()) {
                    error.printStackTrace();
                }
                failures.add(file + ": " + error);
            }
        }
        if (!failures.isEmpty()) {
            fail("Parity failures (" + failures.size() + "):\n  " + String.join("\n  ", failures));
        }
    }

    @Test
    void everySchemaOperatorHasAtLeastOneFixture() throws IOException {
        Map<String, Object> index = Json.parseObject(read("/golden/index.json"));
        Set<String> covered = new HashSet<>();
        for (Object raw : Json.array(index.get("cases"))) {
            covered.add(Json.string(Json.object(raw).get("operator")));
        }
        List<String> missing = new ArrayList<>();
        for (Schemas.Schema schema : Schemas.ALL) {
            if (schema.internal()) {
                continue;
            }
            if (!covered.contains(schema.fullName())) {
                missing.add(schema.fullName());
            }
        }
        assertTrue(missing.isEmpty(), "Operators without a golden fixture: " + missing);
    }

    private void checkFixture(String file) throws IOException {
        Map<String, Object> fixture = Json.parseObject(read("/golden/" + file));
        String operator = Json.string(fixture.get("operator"));
        Op op = Op.of(operator);
        Schemas.Schema schema = op.schema();
        boolean nondeterministic = Json.booleanValue(fixture.get("nondeterministic"));
        boolean inplace = Json.booleanValue(fixture.get("inplace"));
        List<Object> rawArgs = Json.array(fixture.get("args"));
        List<Object> rawOutputs = Json.array(fixture.get("outputs"));

        Value[] args = buildArgs(schema, rawArgs, false);
        long versionBefore = inplace ? args[0].asTensor().versionCounter().current() : 0;
        Tensor[] outputs = call(op, args);
        assertEquals(rawOutputs.size(), outputs.length, operator + " output count");

        for (int i = 0; i < outputs.length; i++) {
            Map<String, Object> expected = Json.object(rawOutputs.get(i));
            Tensor actual = outputs[i];
            assertEquals(Json.array(expected.get("shape")).size(), actual.dim(), operator + " rank");
            assertArrayEquals(ints(expected.get("shape")), actual.shape().toArray(), operator + " shape");
            assertEquals(Json.string(expected.get("dtype")), actual.dtype().typeName(), operator + " dtype");
            if (!nondeterministic) {
                assertArrayEquals(ints(expected.get("strides")), actual.strides().toArray(),
                        operator + " strides");
                assertEquals(Json.longValue(expected.get("storageOffset")), actual.storageOffset(),
                        operator + " storage offset");
                compareValues(Json.string(expected.get("dtype")), expected, actual, operator);
            }
        }
        if (inplace) {
            assertTrue(args[0].asTensor().versionCounter().current() > versionBefore,
                    operator + " must bump the version counter");
        }

        checkMetaAgreement(op, schema, rawArgs, outputs);

        if (!nondeterministic) {
            checkGradients(op, fixture, args, outputs);
        }
    }

    private static void checkMetaAgreement(
            Op op, Schemas.Schema schema, List<Object> rawArgs, Tensor[] cpuOutputs) {
        Value[] metaArgs = buildArgs(schema, rawArgs, true);
        Tensor[] metaOutputs = call(op, metaArgs);
        assertEquals(cpuOutputs.length, metaOutputs.length, op.name() + " meta output count");
        for (int i = 0; i < cpuOutputs.length; i++) {
            assertEquals(cpuOutputs[i].shape(), metaOutputs[i].shape(), op.name() + " meta shape");
            assertEquals(cpuOutputs[i].dtype(), metaOutputs[i].dtype(), op.name() + " meta dtype");
            assertEquals(cpuOutputs[i].strides(), metaOutputs[i].strides(), op.name() + " meta strides");
            assertEquals(Device.meta(), metaOutputs[i].device(), op.name() + " meta device");
        }
    }

    private static void checkGradients(
            Op op, Map<String, Object> fixture, Value[] args, Tensor[] outputs) {
        List<Object> gradientEntries = Json.array(fixture.get("grads"));
        if (gradientEntries.isEmpty()) {
            return;
        }
        Tensor loss = null;
        for (Tensor output : outputs) {
            if (output.dtype().isFloatingPoint()) {
                Tensor contribution = output.sum();
                loss = loss == null ? contribution : loss.add(contribution);
            }
        }
        assertNotNull(loss, op.name() + " has gradients but no floating output");
        assertTrue(loss.requiresGrad(), op.name() + " loss must require grad");
        loss.backward();
        for (Object raw : gradientEntries) {
            Map<String, Object> entry = Json.object(raw);
            int argIndex = Json.intValue(entry.get("argIndex"));
            Map<String, Object> expected = Json.object(entry.get("grad"));
            Tensor actual = args[argIndex].asTensor().grad();
            assertNotNull(actual, op.name() + " gradient for argument " + argIndex);
            assertArrayEquals(ints(expected.get("shape")), actual.shape().toArray(),
                    op.name() + " gradient shape");
            compareValues(Json.string(expected.get("dtype")), expected, actual,
                    op.name() + " gradient[" + argIndex + "]");
        }
    }

    private static Value[] buildArgs(Schemas.Schema schema, List<Object> rawArgs, boolean meta) {
        Value[] values = new Value[rawArgs.size()];
        for (int i = 0; i < rawArgs.size(); i++) {
            Schemas.Arg arg = schema.args().get(i);
            Map<String, Object> raw = Json.object(rawArgs.get(i));
            values[i] = buildValue(arg.type(), raw, meta);
        }
        return values;
    }

    private static Value buildValue(String type, Map<String, Object> raw, boolean meta) {
        boolean optional = type.endsWith("?");
        String base = optional ? type.substring(0, type.length() - 1) : type;
        String kind = Json.string(raw.get("kind"));
        if (kind.equals("none")) {
            return Value.none();
        }
        return switch (base) {
            case "Tensor" -> {
                if (kind.equals("tensorlist")) {
                    throw new IllegalArgumentException("Tensor[] value in a Tensor slot");
                }
                yield Value.tensor(tensor(raw, meta));
            }
            case "Tensor[]" -> {
                List<Object> entries = Json.array(raw.get("value"));
                Tensor[] tensors = new Tensor[entries.size()];
                for (int i = 0; i < entries.size(); i++) {
                    tensors[i] = tensor(Json.object(entries.get(i)), meta);
                }
                yield Value.tensors(tensors);
            }
            case "Scalar" -> Value.scalar(switch (kind) {
                case "int" -> Scalar.of(Json.longValue(raw.get("value")));
                case "float" -> Scalar.of(Json.doubleValue(raw.get("value")));
                case "bool" -> Scalar.of(Json.booleanValue(raw.get("value")));
                default -> throw new IllegalArgumentException("bad scalar kind " + kind);
            });
            case "int" -> Value.integer(Json.longValue(raw.get("value")));
            case "int[]" -> Value.integers(longs(raw.get("value")));
            case "bool" -> Value.bool(Json.booleanValue(raw.get("value")));
            case "str" -> Value.string(Json.string(raw.get("value")));
            case "DType" -> Value.dtype(DType.parse(Json.string(raw.get("value"))));
            case "Device" -> Value.device(meta ? Device.meta() : Device.parse(Json.string(raw.get("value"))));
            default -> throw new IllegalArgumentException("unknown schema argument type " + type);
        };
    }

    private static Tensor tensor(Map<String, Object> raw, boolean meta) {
        String dtypeText = Json.string(raw.get("dtype"));
        DType dtype = DType.parse(dtypeText);
        long[] shape = longs(raw.get("shape"));
        long[] strides = longs(raw.get("strides"));
        long storageOffset = Json.longValue(raw.get("storageOffset"));
        boolean requiresGrad = Json.booleanValue(raw.get("requiresGrad"));
        List<Object> data = Json.array(raw.get("data"));
        Shape shapeValue = Shape.of(shape);
        Strides stridesValue = Strides.of(strides);
        if (meta) {
            return Tensor.wrap(TensorImpl.meta(shapeValue, stridesValue, dtype,
                    requiresGrad && dtype.isFloatingPoint()));
        }
        long storageElements = requiredElements(shapeValue, stridesValue, storageOffset);
        Tensor base = Torch.empty(Shape.of(storageElements), TensorOptions.defaults().dtype(dtype));
        for (int i = 0; i < data.size(); i++) {
            long[] index = new long[shape.length];
            long remaining = i;
            for (int d = shape.length - 1; d >= 0; d--) {
                long size = shape[d];
                index[d] = size == 0 ? 0 : remaining % size;
                remaining = size == 0 ? 0 : remaining / size;
            }
            long offset = storageOffset;
            for (int d = 0; d < shape.length; d++) {
                offset += index[d] * strides[d];
            }
            setValue(base, offset, dtype, data.get(i));
        }
        if (storageOffset == 0 && stridesValue.isContiguous(shapeValue)) {
            TensorImpl impl = new TensorImpl(base.impl().storage(), 0, shapeValue, stridesValue,
                    base.versionCounter(), requiresGrad, false);
            return Tensor.wrap(impl);
        }
        TensorImpl impl = new TensorImpl(base.impl().storage(), storageOffset, shapeValue, stridesValue,
                base.versionCounter(), requiresGrad, true);
        return Tensor.wrap(impl);
    }

    private static long requiredElements(Shape shape, Strides strides, long storageOffset) {
        long max = storageOffset;
        for (int i = 0; i < shape.rank(); i++) {
            if (shape.dim(i) == 0) {
                return Math.max(storageOffset, 0);
            }
            max += (shape.dim(i) - 1) * strides.stride(i);
        }
        return max + 1;
    }

    private static void setValue(Tensor tensor, long offset, DType dtype, Object value) {
        switch (dtype) {
            case BOOL -> TensorAccess.setBoolean(tensor.impl(), offset, Json.booleanValue(value));
            case INT32, INT64 -> TensorAccess.setLong(tensor.impl(), offset, Json.longValue(value));
            case FLOAT32, FLOAT64 -> TensorAccess.setDouble(tensor.impl(), offset, Json.doubleValue(value));
        }
    }

    private static void compareValues(
            String dtypeText, Map<String, Object> expected, Tensor actual, String context) {
        DType dtype = DType.parse(dtypeText);
        List<Object> data = Json.array(expected.get("data"));
        assertEquals(data.size(), actual.numel(), context + " element count");
        for (int i = 0; i < data.size(); i++) {
            long offset = actual.elementOffset(i);
            switch (dtype) {
                case BOOL -> assertEquals(Json.booleanValue(data.get(i)),
                        TensorAccess.getBoolean(actual.impl(), offset), context + " value[" + i + "]");
                case INT32, INT64 -> assertEquals(Json.longValue(data.get(i)),
                        TensorAccess.getLong(actual.impl(), offset), context + " value[" + i + "]");
                case FLOAT32 -> {
                    double expectedValue = Json.doubleValue(data.get(i));
                    double actualValue = TensorAccess.getFloat(actual.impl(), offset);
                    double tolerance = 1e-5 * Math.max(1.0, Math.abs(expectedValue));
                    assertTrue(Math.abs(expectedValue - actualValue) <= tolerance,
                            context + " value[" + i + "] expected " + expectedValue + " but was " + actualValue);
                }
                case FLOAT64 -> {
                    double expectedValue = Json.doubleValue(data.get(i));
                    double actualValue = TensorAccess.getDouble(actual.impl(), offset);
                    double tolerance = 1e-11 * Math.max(1.0, Math.abs(expectedValue));
                    assertTrue(Math.abs(expectedValue - actualValue) <= tolerance,
                            context + " value[" + i + "] expected " + expectedValue + " but was " + actualValue);
                }
            }
        }
    }

    private static Tensor[] call(Op op, Value[] args) {
        Value[] result = Dispatcher.get().call(op, args);
        Tensor[] tensors = new Tensor[result.length];
        for (int i = 0; i < result.length; i++) {
            tensors[i] = result[i].asTensor();
        }
        return tensors;
    }

    private static long[] ints(Object value) {
        List<Object> raw = Json.array(value);
        long[] out = new long[raw.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = Json.longValue(raw.get(i));
        }
        return out;
    }

    private static long[] longs(Object value) {
        return ints(value);
    }

    private static String read(String resource) throws IOException {
        try (InputStream stream = ParityTest.class.getResourceAsStream(resource)) {
            if (stream == null) {
                throw new IOException("Missing test resource " + resource);
            }
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

}
