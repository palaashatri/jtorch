package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.jtorch.core.AutogradException;
import org.jtorch.core.Device;
import org.jtorch.core.GradMode;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Op;
import org.jtorch.core.OpRegistry;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.dispatch.DispatchKey;
import org.jtorch.core.dispatch.DispatchKeySet;
import org.jtorch.core.generated.Ops;
import org.jtorch.core.generated.Schemas;
import org.jtorch.core.value.Value;
import org.junit.jupiter.api.Test;

class DispatchTest {
    @Test
    void generatedIdentifiersAgreeWithTheSchema() {
        assertEquals(Schemas.ALL.size(), Ops.class.getDeclaredFields().length);
        for (Schemas.Schema schema : Schemas.ALL) {
            assertTrue(schema.differentiable() || schema.status().equals("forward"));
            if (schema.differentiable()) {
                assertTrue(schema.reverse() != null, "missing reverse for " + schema.fullName());
            }
        }
    }

    @Test
    void everyOperatorHasAKernelForEveryRequiredKey() {
        Dispatcher dispatcher = Dispatcher.get();
        java.util.List<String> missing = new java.util.ArrayList<>();
        for (Schemas.Schema schema : Schemas.ALL) {
            Op op = Op.of(schema.fullName());
            boolean composite = op.composite();
            if (!dispatcher.hasKernel(op, DispatchKey.CPU) && !composite) {
                missing.add(schema.fullName() + " cpu");
            }
            if (!dispatcher.hasKernel(op, DispatchKey.META) && !composite) {
                missing.add(schema.fullName() + " meta");
            }
            if (schema.differentiable() && !composite
                    && !dispatcher.hasKernel(op, DispatchKey.AUTOGRAD)) {
                missing.add(schema.fullName() + " autograd");
            }
            if (!schema.differentiable() && dispatcher.hasKernel(op, DispatchKey.AUTOGRAD)) {
                missing.add(schema.fullName() + " unexpected autograd kernel");
            }
        }
        assertTrue(missing.isEmpty(), "Kernel coverage gaps: " + missing);
    }

    @Test
    void duplicateRegistrationFailsLoudly() {
        Dispatcher dispatcher = Dispatcher.get();
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> dispatcher.register(OpRegistry.ABS, DispatchKey.CPU,
                        (op, args, keys) -> new Value[] {Value.tensor(args[0].asTensor())}));
        assertTrue(error.getMessage().contains("Duplicate"), error.getMessage());
    }

    @Test
    void dispatchKeysAreDerivedFromArguments() {
        Tensor cpu = Torch.zeros(Shape.of(2));
        DispatchKeySet cpuKeys = Dispatcher.computeKeys(OpRegistry.ABS, new Value[] {Value.tensor(cpu)});
        assertTrue(cpuKeys.has(DispatchKey.CPU));
        assertFalse(cpuKeys.has(DispatchKey.AUTOGRAD));

        Tensor grad = Torch.zeros(Shape.of(2)).requiresGrad(true);
        DispatchKeySet gradKeys = Dispatcher.computeKeys(OpRegistry.ABS, new Value[] {Value.tensor(grad)});
        assertTrue(gradKeys.has(DispatchKey.AUTOGRAD));

        DispatchKeySet noGradKeys = GradMode.noGrad(
                () -> Dispatcher.computeKeys(OpRegistry.ABS, new Value[] {Value.tensor(grad)}));
        assertFalse(noGradKeys.has(DispatchKey.AUTOGRAD));

        Tensor meta = Torch.zeros(Shape.of(2), TensorOptions.defaults().device(Device.meta()));
        DispatchKeySet metaKeys = Dispatcher.computeKeys(OpRegistry.ABS, new Value[] {Value.tensor(meta)});
        assertTrue(metaKeys.has(DispatchKey.META));
    }

    @Test
    void arityAndTypesAreValidatedBeforeKernelsRun() {
        Tensor value = Torch.zeros(Shape.of(2));
        assertThrows(JTorchException.class,
                () -> Dispatcher.get().call(OpRegistry.ADD_TENSOR, Value.tensor(value)));
        assertThrows(JTorchException.class,
                () -> Dispatcher.get().call(OpRegistry.ABS, Value.integer(2)));
        assertThrows(IllegalArgumentException.class, () -> Op.of("aten::does_not_exist.default"));
    }

    @Test
    void autogradKeyWithoutAFormulaFailsAtForward() {
        Tensor grad = Torch.zeros(Shape.of(2)).requiresGrad(true);
        AutogradException error = assertThrows(AutogradException.class,
                () -> Dispatcher.get().callWithKeys(
                        OpRegistry.PROD, new Value[] {Value.tensor(grad)},
                        DispatchKeySet.of(DispatchKey.CPU, DispatchKey.AUTOGRAD)));
        assertTrue(error.getMessage().contains("No autograd kernel"), error.getMessage());
    }

    @Test
    void dispatcherRejectsUnsupportedDevices() {
        // Factory dispatch itself must reject an unavailable device rather than silently using the CPU.
        assertThrows(org.jtorch.core.DeviceException.class,
                () -> Torch.zeros(Shape.of(2), TensorOptions.defaults().device(Device.cuda(0))));
    }
}
