package org.jtorch.core.value;

import java.util.Arrays;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Scalar;
import org.jtorch.core.Tensor;

/**
 * The runtime value algebra for operator arguments and results. Operator kernels never exchange raw {@code Object}
 * arrays: every argument is one of these typed values.
 */
public sealed interface Value {

    record TensorValue(Tensor value) implements Value {}

    record TensorListValue(Tensor[] values) implements Value {
        public TensorListValue {
            values = values.clone();
        }
    }

    record IntValue(long value) implements Value {}

    record IntListValue(long[] values) implements Value {
        public IntListValue {
            values = values.clone();
        }
    }

    record FloatValue(double value) implements Value {}

    record BoolValue(boolean value) implements Value {}

    record StringValue(String value) implements Value {}

    record ScalarValue(Scalar value) implements Value {}

    record DTypeValue(DType value) implements Value {}

    record DeviceValue(Device value) implements Value {}

    /** Optional argument that is absent, matching {@code None} in the operator schema. */
    enum None implements Value {
        INSTANCE
    }

    static Value tensor(Tensor tensor) {
        return new TensorValue(tensor);
    }

    static Value tensors(Tensor... tensors) {
        return new TensorListValue(tensors);
    }

    static Value integer(long value) {
        return new IntValue(value);
    }

    static Value integers(long... values) {
        return new IntListValue(values);
    }

    static Value floating(double value) {
        return new FloatValue(value);
    }

    static Value bool(boolean value) {
        return new BoolValue(value);
    }

    static Value string(String value) {
        return new StringValue(value);
    }

    static Value scalar(Scalar value) {
        return new ScalarValue(value);
    }

    static Value dtype(DType value) {
        return new DTypeValue(value);
    }

    static Value device(Device value) {
        return new DeviceValue(value);
    }

    static Value none() {
        return None.INSTANCE;
    }

    default Tensor asTensor() {
        if (this instanceof TensorValue tensor) {
            return tensor.value();
        }
        throw mismatch("Tensor");
    }

    default Tensor[] asTensorList() {
        if (this instanceof TensorListValue list) {
            return list.values().clone();
        }
        throw mismatch("Tensor[]");
    }

    default long asLong() {
        if (this instanceof IntValue value) {
            return value.value();
        }
        throw mismatch("int");
    }

    default int asInt() {
        return Math.toIntExact(asLong());
    }

    default long[] asLongArray() {
        if (this instanceof IntListValue list) {
            return list.values();
        }
        throw mismatch("int[]");
    }

    default double asDouble() {
        if (this instanceof FloatValue value) {
            return value.value();
        }
        throw mismatch("float");
    }

    default boolean asBoolean() {
        if (this instanceof BoolValue value) {
            return value.value();
        }
        throw mismatch("bool");
    }

    default String asString() {
        if (this instanceof StringValue value) {
            return value.value();
        }
        throw mismatch("str");
    }

    default Scalar asScalar() {
        if (this instanceof ScalarValue scalar) {
            return scalar.value();
        }
        throw mismatch("Scalar");
    }

    default DType asDType() {
        if (this instanceof DTypeValue value) {
            return value.value();
        }
        throw mismatch("DType");
    }

    default Device asDevice() {
        if (this instanceof DeviceValue value) {
            return value.value();
        }
        throw mismatch("Device");
    }

    default boolean isNone() {
        return this == None.INSTANCE;
    }

    default Tensor asOptionalTensor() {
        return isNone() ? null : asTensor();
    }

    default Scalar asOptionalScalar() {
        return isNone() ? null : asScalar();
    }

    default Integer asOptionalInt() {
        return isNone() ? null : asInt();
    }

    private JTorchException mismatch(String expected) {
        String actual = this == None.INSTANCE ? "None" : getClass().getSimpleName();
        return new JTorchException("Operator argument type mismatch: expected " + expected + " but got " + actual);
    }

    static String describe(Value value) {
        if (value instanceof IntListValue list) {
            return Arrays.toString(list.values());
        }
        return String.valueOf(value);
    }
}
