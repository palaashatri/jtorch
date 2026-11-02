package org.jtorch.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.junit.jupiter.api.Test;

/** Regression tests for module-registration defects found by hostile QA. */
class ModuleRegistrationRegressionTest {
    private static final class Toy extends Module {
        private final Parameter weight;
        private final Tensor persistent;
        private final Tensor scratch;

        Toy() {
            weight = registerParameter("weight", new Parameter(Torch.ones(Shape.of(2))));
            persistent = registerBuffer("running", Torch.zeros(Shape.of(1)), true);
            scratch = registerBuffer("scratch", Torch.zeros(Shape.of(1)), false);
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    @Test
    void nonPersistentBuffersDoNotBreakStrictSelfLoad() {
        Toy toy = new Toy();
        LoadStateResult result = toy.loadStateDict(new StateDict(toy.stateDict()), true);
        assertEquals(List.of(), result.missingKeys());
        assertEquals(List.of(), result.unexpectedKeys());
        assertEquals(List.of("weight", "running"), List.copyOf(toy.stateDict().keySet()));
        assertEquals(List.of("running", "scratch"), List.copyOf(toy.namedBuffers().keySet()));
    }

    @Test
    void bufferCannotShadowParameterName() {
        assertThrows(IllegalArgumentException.class, () -> new Module() {
            {
                registerParameter("x", new Parameter(Torch.ones(Shape.of(1))));
                registerBuffer("x", Torch.zeros(Shape.of(1)), true);
            }

            @Override
            public Tensor forward(Tensor input) {
                return input;
            }
        });
    }

    @Test
    void sharedChildParametersAreDeduplicated() {
        Module model = new Module() {
            private final Linear shared = registerModule("a", new Linear(2, 2, false));
            private final Linear alias = registerModule("b", shared);

            @Override
            public Tensor forward(Tensor input) {
                return shared.call(input).add(alias.call(input));
            }
        };
        assertEquals(1, model.parameters().size());
        assertEquals(List.of("a.weight"), List.copyOf(model.namedParameters().keySet()));
        assertEquals(List.of("a.weight", "b.weight"), List.copyOf(model.stateDict().keySet()));
    }
}
