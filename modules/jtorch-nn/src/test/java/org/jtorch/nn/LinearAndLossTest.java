package org.jtorch.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.jtorch.nn.functional.F;
import org.junit.jupiter.api.Test;

class LinearAndLossTest {
    private static final double TOLERANCE = 1e-5;

    @Test
    void linearMatchesHandComputedForwardAndGradients() {
        Linear linear = new Linear(3, 2);
        Tensor weight = Torch.tensor(new float[] {1, 0, -1, 0.5f, 1, 0}, Shape.of(2, 3));
        Tensor bias = Torch.tensor(new float[] {0.1f, -0.2f}, Shape.of(2));
        Torch.noGrad(() -> {
            linear.weight().copy_(weight);
            linear.bias().copy_(bias);
        });
        Tensor input = Torch.tensor(new float[] {1, 2, 3, 4, 5, 6}, Shape.of(2, 3)).requiresGrad(true);
        Tensor output = linear.call(input);
        assertArrayEquals(new double[] {-1.9, 2.3, -1.9, 6.8}, output.toDoubleArray(), TOLERANCE);
        output.sum().backward();
        assertArrayEquals(new double[] {1.5, 1.0, -1.0, 1.5, 1.0, -1.0}, input.grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(new double[] {5, 7, 9, 5, 7, 9}, linear.weight().grad().toDoubleArray(), TOLERANCE);
        assertArrayEquals(new double[] {2, 2}, linear.bias().grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void mseLossReductions() {
        Tensor input = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
        Tensor target = Torch.zeros(Shape.of(2, 2));
        Tensor mean = new MSELoss().forward(input, target);
        assertEquals(7.5, mean.item(), TOLERANCE);
        Tensor sum = new MSELoss(F.Reduction.SUM).forward(input, target);
        assertEquals(30, sum.item(), TOLERANCE);
        Tensor none = new MSELoss(F.Reduction.NONE).forward(input, target);
        assertEquals(Shape.of(2, 2), none.shape());
        mean.backward();
        assertArrayEquals(new double[] {0.5, 1.0, 1.5, 2.0}, input.grad().toDoubleArray(), TOLERANCE);
    }

    @Test
    void crossEntropyValuesAndGradient() {
        Tensor logits = Torch.tensor(new float[] {0, 0, 1, 1}, Shape.of(2, 2)).requiresGrad(true);
        Tensor targets = Torch.tensor(new long[] {0, 1}, Shape.of(2),
                TensorOptions.defaults().dtype(DType.INT64));
        Tensor loss = new CrossEntropyLoss().forward(logits, targets);
        assertEquals(Math.log(2), loss.item(), TOLERANCE);
        loss.backward();
        double[] expected = {0.25, 0.25, 0.25, 0.25};
        // Row 1: softmax(0,0) - onehot(0) = [-0.5, 0.5]; row 2: [0.5, -0.5], scaled by 1/2.
        assertArrayEquals(new double[] {-0.25, 0.25, 0.25, -0.25}, logits.grad().toDoubleArray(), TOLERANCE);
        assertEquals(4, expected.length);
    }

    @Test
    void crossEntropyRejectsBadTargets() {
        Tensor logits = Torch.zeros(Shape.of(2, 3));
        Tensor wrongDtype = Torch.tensor(new float[] {0, 1}, Shape.of(2));
        assertThrows(org.jtorch.core.DTypeException.class,
                () -> new CrossEntropyLoss().forward(logits, wrongDtype));
        Tensor outOfRange = Torch.tensor(new long[] {0, 5}, Shape.of(2),
                TensorOptions.defaults().dtype(DType.INT64));
        assertThrows(IndexOutOfBoundsException.class,
                () -> new CrossEntropyLoss().forward(logits, outOfRange));
    }

    @Test
    void invalidReductionIsRejected() {
        Tensor input = Torch.zeros(Shape.of(2));
        assertThrows(IllegalArgumentException.class, () -> input.mseLoss(input, "median"));
    }
}
