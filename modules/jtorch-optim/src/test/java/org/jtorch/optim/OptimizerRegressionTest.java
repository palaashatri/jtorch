package org.jtorch.optim;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.nn.Linear;
import org.jtorch.nn.Module;
import org.jtorch.nn.Parameter;
import org.junit.jupiter.api.Test;

/** Regression tests for optimizer API and shared-parameter defects found by hostile QA. */
class OptimizerRegressionTest {
    @Test
    void adamWBuilderIsFluentWithTheConcreteType() {
        Parameter parameter = new Parameter(Torch.ones(Shape.of(2)));
        AdamW optimizer = AdamW.builder()
                .parameters(List.of(parameter))
                .learningRate(0.1)
                .betas(0.9, 0.99)
                .epsilon(0.0)
                .weightDecay(0.01)
                .build();
        parameter.impl().setGrad(Torch.ones(Shape.of(2)));
        optimizer.step();
        assertEquals(0.1, optimizer.learningRate(), 0);
    }

    @Test
    void hyperparametersAreValidated() {
        Parameter parameter = new Parameter(Torch.ones(Shape.of(1)));
        assertThrows(IllegalArgumentException.class, () -> SGD.builder()
                .parameters(List.of(parameter)).momentum(-0.5).build());
        assertThrows(IllegalArgumentException.class, () -> SGD.builder()
                .parameters(List.of(parameter)).weightDecay(-1).build());
        assertThrows(IllegalArgumentException.class, () -> SGD.builder()
                .parameters(List.of(parameter)).dampening(-0.1).build());
        assertThrows(IllegalArgumentException.class, () -> SGD.builder()
                .parameters(List.of(parameter)).learningRate(0).build());
    }

    @Test
    void documentedVerticalSliceUpdatesTheSourceTensor() {
        Tensor x = Torch.tensor(new float[] {1, 2, 3, 4}, org.jtorch.core.Shape.of(2, 2))
                .requiresGrad(true);
        Tensor w = Torch.randn(org.jtorch.core.Shape.of(2, 1),
                org.jtorch.core.TensorOptions.defaults().requiresGrad(true));
        Tensor loss = x.matmul(w).square().mean();
        loss.backward();
        double[] before = w.toDoubleArray();
        SGD optimizer = SGD.builder()
                .parameters(List.of(new Parameter(w)))
                .learningRate(1e-2)
                .build();
        optimizer.step();
        assertArrayEquals(new double[] {
            before[0] - 0.01 * w.grad().toDoubleArray()[0],
            before[1] - 0.01 * w.grad().toDoubleArray()[1]}, w.toDoubleArray(), 1e-6);
    }

    @Test
    void sharedParametersReceiveOneUpdate() {
        Module model = new Module() {
            private final Linear shared = registerModule("a", new Linear(2, 2, false));
            private final Linear alias = registerModule("b", shared);

            @Override
            public Tensor forward(Tensor input) {
                return shared.call(input).add(alias.call(input));
            }
        };
        Torch.noGrad(() -> model.namedParameters().get("a.weight").fill_(1));
        Tensor input = Torch.ones(Shape.of(1, 2));
        model.call(input).sum().backward();
        SGD optimizer = SGD.builder().parameters(model.parameters()).learningRate(0.1).build();
        optimizer.step();
        assertArrayEquals(new float[] {0.8f, 0.8f, 0.8f, 0.8f},
                model.namedParameters().get("a.weight").toFloatArray(), 1e-6f);
    }
}
