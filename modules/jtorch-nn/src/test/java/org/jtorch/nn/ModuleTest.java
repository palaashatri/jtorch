package org.jtorch.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class ModuleTest {
    private static final double TOLERANCE = 1e-6;

    private static final class Toy extends Module {
        private final Parameter weight;
        private final Tensor runningMean;
        private final Tensor calls;

        Toy() {
            weight = registerParameter("weight", new Parameter(Torch.ones(Shape.of(2, 2))));
            runningMean = registerBuffer("running_mean", Torch.zeros(Shape.of(2)), true);
            calls = registerBuffer("calls", Torch.zeros(Shape.of(1)), false);
        }

        @Override
        public Tensor forward(Tensor input) {
            return input.matmul(weight);
        }
    }

    @Test
    void parametersAndBuffersAreRegisteredExplicitly() {
        Toy toy = new Toy();
        assertEquals(List.of("weight"), List.copyOf(toy.namedParameters().keySet()));
        assertEquals(List.of("running_mean", "calls"), List.copyOf(toy.namedBuffers().keySet()));
        assertEquals(List.of("weight", "running_mean"), List.copyOf(toy.stateDict().keySet()));
        assertTrue(toy.parameters().get(0).requiresGrad());
    }

    @Test
    void duplicateRegistrationFailsLoudly() {
        assertThrows(IllegalArgumentException.class, () -> new Module() {
            {
                registerParameter("p", new Parameter(Torch.zeros(Shape.of(1))));
                registerParameter("p", new Parameter(Torch.zeros(Shape.of(1))));
            }

            @Override
            public Tensor forward(Tensor input) {
                return input;
            }
        });
    }

    @Test
    void hooksWrapTheForwardCall() {
        Toy toy = new Toy();
        toy.registerForwardPreHook(input -> input.add(Torch.ones(input.shape())));
        toy.registerForwardHook(output -> output.mul(Torch.full(Shape.scalar(),
                org.jtorch.core.Scalar.of(2))));
        Tensor input = Torch.ones(Shape.of(1, 2));
        Tensor output = toy.call(input);
        assertArrayEquals(new double[] {8, 8}, output.toDoubleArray(), TOLERANCE);
    }

    @Test
    void trainModePropagatesToChildren() {
        Sequential model = new Sequential(new Linear(2, 2), new ReLU());
        assertTrue(model.isTraining());
        model.eval();
        assertFalse(model.isTraining());
        assertFalse(model.modules().get(0).isTraining());
        model.train();
        assertTrue(model.modules().get(1).isTraining());
    }

    @Test
    void stateDictRoundTripsThroughStrictLoad() {
        Linear source = new Linear(3, 2);
        Tensor before = source.stateDict().get("weight").clone();
        StateDict saved = source.stateDict();
        Linear target = new Linear(3, 2);
        LoadStateResult result = target.loadStateDict(new StateDict(saved), true);
        assertTrue(result.isClean());
        assertArrayEquals(before.toFloatArray(), target.stateDict().get("weight").toFloatArray(), 1e-6f);
    }

    @Test
    void strictLoadReportsMissingAndUnexpectedKeys() {
        Linear model = new Linear(3, 2);
        StateDict empty = new StateDict();
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> model.loadStateDict(empty, true));
        assertTrue(error.getMessage().contains("missing keys"), error.getMessage());

        StateDict tooMany = new StateDict(model.stateDict());
        tooMany.put("extra", Torch.zeros(Shape.of(1)));
        assertThrows(IllegalArgumentException.class, () -> model.loadStateDict(tooMany, true));
        LoadStateResult nonStrict = model.loadStateDict(tooMany, false);
        assertEquals(List.of("extra"), nonStrict.unexpectedKeys());
    }

    @Test
    void shapeMismatchIsRejected() {
        Linear model = new Linear(3, 2);
        StateDict broken = new StateDict(model.stateDict());
        broken.put("weight", Torch.zeros(Shape.of(4, 4)));
        assertThrows(IllegalArgumentException.class, () -> model.loadStateDict(new StateDict(broken), false));
    }

    @Test
    void positionalAndNestedNamesMatchPyTorch() {
        Sequential sequential = new Sequential(
                new Linear(4, 3),
                new ReLU(),
                new Linear(3, 2));
        assertEquals(List.of("0.weight", "0.bias", "2.weight", "2.bias"),
                new ArrayList<>(sequential.stateDict().keySet()));
        sequential.zeroGrad();
        for (Parameter parameter : sequential.parameters()) {
            assertNull(parameter.grad());
        }
        assertNotNull(sequential.modules().get(1));
    }

    @Test
    void parameterSharesStorageAndVersionWithSource() {
        Tensor source = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2));
        Parameter parameter = new Parameter(source);
        assertTrue(parameter.sharesStorageWith(source));
        assertTrue(parameter.requiresGrad());
        Torch.noGrad(() -> parameter.fill_(0));
        assertArrayEquals(new double[] {0, 0, 0, 0}, source.toDoubleArray(), TOLERANCE);
    }

    @Test
    void linearBiasIsOptional() {
        Linear withBias = new Linear(2, 3, true);
        Linear withoutBias = new Linear(2, 3, false);
        assertTrue(withBias.namedParameters().containsKey("bias"));
        assertFalse(withoutBias.namedParameters().containsKey("bias"));
        Tensor input = Torch.ones(Shape.of(1, 2));
        assertEquals(Shape.of(1, 3), withoutBias.forward(input).shape());
    }
}
