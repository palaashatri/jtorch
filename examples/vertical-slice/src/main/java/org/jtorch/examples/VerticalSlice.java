package org.jtorch.examples;

import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.jtorch.nn.Parameter;
import org.jtorch.optim.Optimizer;
import org.jtorch.optim.SGD;

/** The design's §0.2 vertical slice, runnable from a clean checkout. */
public final class VerticalSlice {
    private VerticalSlice() {}

    public static void main(String[] args) {
        Tensor x = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
        Tensor w = Torch.randn(Shape.of(2, 1), TensorOptions.defaults()
                .dtype(DType.FLOAT32)
                .device(Device.cpu())
                .requiresGrad(true));

        Tensor loss = x.matmul(w).square().mean();
        loss.backward();

        Optimizer optimizer = SGD.builder()
                .parameters(List.of(new Parameter(w)))
                .learningRate(1e-2)
                .build();
        double[] beforeStep = w.toDoubleArray();
        optimizer.step();
        double[] afterStep = w.toDoubleArray();
        if (java.util.Arrays.equals(beforeStep, afterStep)) {
            throw new IllegalStateException("SGD.step() did not update the parameter");
        }

        System.out.printf("loss=%.6f%n", loss.item());
        System.out.println("x.grad=" + java.util.Arrays.toString(x.grad().toDoubleArray()));
        System.out.println("w.grad=" + java.util.Arrays.toString(w.grad().toDoubleArray()));
        System.out.println("w.after_step=" + java.util.Arrays.toString(w.toDoubleArray()));
        System.out.println("VERTICAL_SLICE_OK");
    }
}
