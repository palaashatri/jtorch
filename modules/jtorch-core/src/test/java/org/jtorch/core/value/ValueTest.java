package org.jtorch.core.value;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Scalar;
import org.junit.jupiter.api.Test;

class ValueTest {
    @Test
    void optionalValuesRoundTrip() {
        assertTrue(Value.none().isNone());
        assertNull(Value.none().asOptionalTensor());
        assertNull(Value.none().asOptionalScalar());
        assertNull(Value.none().asOptionalInt());
    }

    @Test
    void typedAccessorsRejectWrongKinds() {
        assertThrows(JTorchException.class, () -> Value.integer(1).asTensor());
        assertThrows(JTorchException.class, () -> Value.bool(true).asLong());
        assertThrows(JTorchException.class, () -> Value.none().asDType());
    }

    @Test
    void scalarKindsSurviveTheValueBoundary() {
        Value value = Value.scalar(Scalar.of(3));
        assertTrue(value.asScalar().isIntegral());
        assertEquals(3, value.asScalar().toLong());
        assertEquals(DType.FLOAT32, Value.dtype(DType.FLOAT32).asDType());
        assertEquals(Device.meta(), Value.device(Device.meta()).asDevice());
    }
}
