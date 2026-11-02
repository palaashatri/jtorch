package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.jtorch.core.AutogradException;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class AutogradFormulasTest {
    private static final double TOLERANCE = 1e-10;

    private static Tensor vector(double... values) {
        return Torch.tensor(values, Shape.of(values.length));
    }

    private static Tensor matrix(double[] values, long rows, long cols) {
        return Torch.tensor(values, Shape.of(rows, cols));
    }

    @Test
    void unaryFormulasMatchAnalyticDerivatives() {
        Tensor x = vector(0.5, 1.5).requiresGrad(true);
        Tensor exp = x.exp();
        exp.sum().backward();
        assertArrayEquals(new double[] {Math.exp(0.5), Math.exp(1.5)}, x.grad().toDoubleArray(), TOLERANCE);

        Tensor log = vector(0.5, 1.5).requiresGrad(true);
        log.log().sum().backward();
        assertArrayEquals(new double[] {2.0, 1.0 / 1.5}, log.grad().toDoubleArray(), TOLERANCE);

        Tensor sqrt = vector(0.5, 1.5).requiresGrad(true);
        sqrt.sqrt().sum().backward();
        assertArrayEquals(new double[] {1 / (2 * Math.sqrt(0.5)), 1 / (2 * Math.sqrt(1.5))},
                sqrt.grad().toDoubleArray(), TOLERANCE);

        Tensor tanh = vector(0.5, 1.5).requiresGrad(true);
        tanh.tanh().sum().backward();
        assertArrayEquals(new double[] {1 - Math.tanh(0.5) * Math.tanh(0.5),
                1 - Math.tanh(1.5) * Math.tanh(1.5)}, tanh.grad().toDoubleArray(), TOLERANCE);

        Tensor sigmoid = vector(0.5, 1.5).requiresGrad(true);
        sigmoid.sigmoid().sum().backward();
        double s0 = 1 / (1 + Math.exp(-0.5));
        double s1 = 1 / (1 + Math.exp(-1.5));
        assertArrayEquals(new double[] {s0 * (1 - s0), s1 * (1 - s1)}, sigmoid.grad().toDoubleArray(), TOLERANCE);

        Tensor relu = vector(-1, 0, 2).requiresGrad(true);
        relu.relu().sum().backward();
        assertArrayEquals(new double[] {0, 0, 1}, relu.grad().toDoubleArray(), TOLERANCE);

        Tensor abs = vector(-2, 3).requiresGrad(true);
        abs.abs().sum().backward();
        assertArrayEquals(new double[] {-1, 1}, abs.grad().toDoubleArray(), TOLERANCE);

        Tensor reciprocal = vector(2, 4).requiresGrad(true);
        reciprocal.reciprocal().sum().backward();
        assertArrayEquals(new double[] {-0.25, -0.0625}, reciprocal.grad().toDoubleArray(), TOLERANCE);

        Tensor trig = vector(0.5, 1.0).requiresGrad(true);
        trig.sin().sum().backward();
        assertArrayEquals(new double[] {Math.cos(0.5), Math.cos(1.0)}, trig.grad().toDoubleArray(), TOLERANCE);

        Tensor cos = vector(0.5, 1.0).requiresGrad(true);
        cos.cos().sum().backward();
        assertArrayEquals(new double[] {-Math.sin(0.5), -Math.sin(1.0)}, cos.grad().toDoubleArray(), TOLERANCE);

        Tensor log1p = vector(1, 2).requiresGrad(true);
        log1p.log1p().sum().backward();
        assertArrayEquals(new double[] {0.5, 1.0 / 3.0}, log1p.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void clampGradientIsMaskedAtTheBoundaries() {
        Tensor x = vector(-2, -1, 0, 1, 2).requiresGrad(true);
        x.clamp(-1, 1).sum().backward();
        assertArrayEquals(new double[] {0, 1, 1, 1, 0}, x.grad().toDoubleArray(), TOLERANCE);

        Tensor lower = vector(-2, -1, 0, 1, 2).requiresGrad(true);
        lower.clampMin(0).sum().backward();
        assertArrayEquals(new double[] {0, 0, 1, 1, 1}, lower.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void sumAndMeanDimGradientsExpandCorrectly() {
        Tensor x = Torch.ones(Shape.of(2, 3, 4)).requiresGrad(true);
        x.sum(new int[] {0, 2}, false).sum().backward();
        double[] ones = new double[24];
        java.util.Arrays.fill(ones, 1.0);
        assertArrayEquals(ones, x.grad().toDoubleArray(), TOLERANCE);
        assertEquals(Shape.of(2, 3, 4), x.grad().shape());

        Tensor meanInput = Torch.ones(Shape.of(2, 3, 4)).requiresGrad(true);
        meanInput.mean(new int[] {0, 2}, true).sum().backward();
        double[] expected = new double[24];
        java.util.Arrays.fill(expected, 1.0 / 8.0);
        assertArrayEquals(expected, meanInput.grad().toDoubleArray(), TOLERANCE);

        Tensor allMean = Torch.ones(Shape.of(4)).requiresGrad(true);
        allMean.mean().backward();
        assertArrayEquals(new double[] {0.25, 0.25, 0.25, 0.25}, allMean.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void maxAndMinSelectTheExtremum() {
        Tensor x = matrix(new double[] {1, 4, 3, 2}, 2, 2).requiresGrad(true);
        x.max().backward();
        assertArrayEquals(new double[] {0, 1, 0, 0}, x.grad().toDoubleArray(), TOLERANCE);

        Tensor y = matrix(new double[] {1, 4, 3, 2}, 2, 2).requiresGrad(true);
        y.min().backward();
        assertArrayEquals(new double[] {1, 0, 0, 0}, y.grad().toDoubleArray(), TOLERANCE);

        Tensor z = matrix(new double[] {1, 4, 3, 2}, 2, 2).requiresGrad(true);
        z.maxWithIndices(1, false).values().sum().backward();
        assertArrayEquals(new double[] {0, 1, 1, 0}, z.grad().toDoubleArray(), TOLERANCE);

        Tensor w = matrix(new double[] {1, 4, 3, 2}, 2, 2).requiresGrad(true);
        w.minWithIndices(0, false).values().sum().backward();
        assertArrayEquals(new double[] {1, 0, 0, 1}, w.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void maximumSplitsGradientsAtTies() {
        Tensor a = vector(2, 1).requiresGrad(true);
        Tensor b = vector(2, 3).requiresGrad(true);
        a.maximum(b).sum().backward();
        assertArrayEquals(new double[] {0.5, 0}, a.grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(new double[] {0.5, 1}, b.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void powGradientsForScalarAndTensorExponent() {
        Tensor base = vector(2, 3).requiresGrad(true);
        base.pow(3).sum().backward();
        assertArrayEquals(new double[] {12, 27}, base.grad().toDoubleArray(), TOLERANCE);

        Tensor tensorBase = vector(2, 3).requiresGrad(true);
        Tensor exponent = vector(2, 2).requiresGrad(true);
        tensorBase.pow(exponent).sum().backward();
        assertArrayEquals(new double[] {4, 6}, tensorBase.grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(new double[] {4 * Math.log(2), 9 * Math.log(3)},
                exponent.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void divisionAndMultiplicationBroadcasting() {
        Tensor a = matrix(new double[] {1, 2, 3, 4, 5, 6}, 2, 3).requiresGrad(true);
        Tensor b = vector(2, 4, 8).requiresGrad(true);
        a.div(b).sum().backward();
        assertArrayEquals(new double[] {0.5, 0.25, 0.125, 0.5, 0.25, 0.125}, a.grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(new double[] {-1.25, -7.0 / 16.0, -9.0 / 64.0},
                b.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void softmaxAndLogSoftmaxGradients() {
        Tensor logits = vector(1, 2, 3).requiresGrad(true);
        Tensor weights = vector(0.5, -1, 2);
        Tensor probabilities = logits.softmax(0);
        Tensor loss = probabilities.mul(weights).sum();
        loss.backward();
        double[] p = probabilities.toDoubleArray();
        double dot = p[0] * 0.5 + p[1] * -1 + p[2] * 2;
        assertArrayEquals(new double[] {p[0] * (0.5 - dot), p[1] * (-1 - dot), p[2] * (2 - dot)},
                logits.grad().toDoubleArray(), TOLERANCE);

        Tensor logs = vector(1, 2, 3).requiresGrad(true);
        logs.logSoftmax(0).mul(weights).sum().backward();
        double[] softmax = new double[] {p[0], p[1], p[2]};
        // d/dx sum(w * log_softmax(x)) = w - softmax(x) * sum(w)
        assertArrayEquals(new double[] {0.5 - softmax[0] * 1.5, -1 - softmax[1] * 1.5, 2 - softmax[2] * 1.5},
                logs.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void viewGradientsReturnToTheBaseShape() {
        Tensor base = matrix(new double[] {1, 2, 3, 4, 5, 6}, 2, 3).requiresGrad(true);
        base.transpose(0, 1).sum().backward();
        double[] ones = new double[6];
        java.util.Arrays.fill(ones, 1.0);
        assertArrayEquals(ones, base.grad().toDoubleArray(), TOLERANCE);

        Tensor expanded = matrix(new double[] {1, 2}, 2, 1).requiresGrad(true);
        expanded.expand(2, 3).sum().backward();
        assertArrayEquals(new double[] {3, 3}, expanded.grad().toDoubleArray(), TOLERANCE);

        Tensor narrowed = matrix(new double[] {1, 2, 3, 4, 5, 6}, 2, 3).requiresGrad(true);
        narrowed.narrow(1, 1, 2).sum().backward();
        assertArrayEquals(new double[] {0, 1, 1, 0, 1, 1}, narrowed.grad().toDoubleArray(), TOLERANCE);

        Tensor selected = matrix(new double[] {1, 2, 3, 4, 5, 6}, 2, 3).requiresGrad(true);
        selected.select(0, 1).sum().backward();
        assertArrayEquals(new double[] {0, 0, 0, 1, 1, 1}, selected.grad().toDoubleArray(), TOLERANCE);

        Tensor reshaped = matrix(new double[] {1, 2, 3, 4, 5, 6}, 2, 3).requiresGrad(true);
        reshaped.reshape(3, 2).sum().backward();
        assertArrayEquals(ones, reshaped.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void catAndStackGradientsRouteToEachInput() {
        Tensor a = Torch.ones(Shape.of(2, 3)).requiresGrad(true);
        Tensor b = Torch.ones(Shape.of(2, 3)).requiresGrad(true);
        Tensor.cat(List.of(a, b), 1).sum().backward();
        double[] ones = new double[6];
        java.util.Arrays.fill(ones, 1.0);
        assertArrayEquals(ones, a.grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(ones, b.grad().toDoubleArray(), TOLERANCE);

        Tensor c = Torch.ones(Shape.of(2, 3)).requiresGrad(true);
        Tensor d = Torch.ones(Shape.of(2, 3)).requiresGrad(true);
        Tensor.stack(List.of(c, d), 0).sum().backward();
        assertArrayEquals(ones, c.grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(ones, d.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void detachStopsGradientFlow() {
        Tensor x = vector(1, 2).requiresGrad(true);
        Tensor detached = x.detach();
        assertFalse(detached.requiresGrad());
        assertThrows(AutogradException.class, () -> detached.sum().backward());

        Tensor y = x.mul(2).mul(detached);
        y.sum().backward();
        assertArrayEquals(new double[] {2, 4}, x.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void gradientsAccumulateAcrossIndependentGraphs() {
        Tensor x = vector(1, 2).requiresGrad(true);
        x.mul(x).sum().backward();
        x.mul(3).sum().backward();
        assertArrayEquals(new double[] {5, 7}, x.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void postAccumulateHooksSeeTheAccumulatedGradient() {
        Tensor x = vector(1, 2).requiresGrad(true);
        List<double[]> observed = new ArrayList<>();
        x.registerPostAccumulateHook(tensor -> observed.add(tensor.toDoubleArray()));
        x.mul(2).sum().backward();
        assertEquals(1, observed.size());
        assertArrayEquals(new double[] {2, 2}, observed.get(0), TOLERANCE);
    }

    @Test
    void backwardRejectsNonScalarImplicitSeedAndHigherOrderRequests() {
        Tensor x = Torch.ones(Shape.of(2)).requiresGrad(true);
        assertThrows(AutogradException.class, x::backward);
        assertThrows(AutogradException.class,
                () -> x.sum().backward(Torch.ones(Shape.of(3)), false));
        assertThrows(AutogradException.class,
                () -> org.jtorch.core.JTorch.gradientEngine().backward(x.sum(), null, false, true));
    }

    @Test
    void explicitSeedShapesAreValidated() {
        Tensor x = Torch.tensor(new double[] {1, 2}, Shape.of(2)).requiresGrad(true);
        Tensor loss = x.mul(x).sum();
        loss.backward(Torch.ones(Shape.scalar()), true);
        assertArrayEquals(new double[] {2, 4}, x.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void graphFreeingIsControlledByRetainGraph() {
        Tensor x = vector(1, 2).requiresGrad(true);
        Tensor loss = x.mul(2).sum();
        loss.backward();
        assertTrue(x.grad() != null);
        AtomicReference<String> secondError = new AtomicReference<>();
        try {
            loss.backward();
        } catch (AutogradException error) {
            secondError.set(error.getMessage());
        }
        assertTrue(secondError.get().contains("second time"));

        Tensor y = vector(1, 2).requiresGrad(true);
        Tensor retained = y.mul(3).sum();
        retained.backward(true);
        retained.backward(true);
        assertArrayEquals(new double[] {6, 6}, y.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void inPlaceMutationThroughAViewInvalidatesSavedTensors() {
        Tensor base = matrix(new double[] {1, 2, 3, 4}, 2, 2).requiresGrad(true);
        Tensor product = base.mul(base);
        Tensor view = base.transpose(0, 1);
        org.jtorch.core.GradMode.noGrad(() -> view.fill_(0));
        AutogradException error = assertThrows(AutogradException.class, () -> product.sum().backward());
        assertTrue(error.getMessage().contains("modified by an inplace operation"));
    }

    @Test
    void floatInputsOnlyCanRequireGrad() {
        Tensor ints = Torch.tensor(new long[] {1, 2}, Shape.of(2));
        assertThrows(org.jtorch.core.DTypeException.class, () -> ints.requiresGrad(true));
        Tensor clone = ints.clone();
        assertFalse(clone.requiresGrad());
    }

    @Test
    void tensorConstructionOptionsAreHonoured() {
        Tensor tensor = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(2, 2),
                TensorOptions.defaults().dtype(DType.FLOAT64).requiresGrad(true));
        assertEquals(DType.FLOAT64, tensor.dtype());
        assertTrue(tensor.requiresGrad());
        assertTrue(tensor.isLeaf());
    }
}
