package org.jtorch.nn;

import org.jtorch.api.Torch;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/**
 * Inference-mode batch normalization over NCHW tensors. Training-mode batch statistics are not implemented; use
 * {@code eval()} before calling the module and fail loudly otherwise.
 */
public final class BatchNorm2d extends Module {
    private final Parameter weight;
    private final Parameter bias;
    private final Tensor runningMean;
    private final Tensor runningVar;
    private final int channels;
    private final double eps;

    public BatchNorm2d(int channels) {
        this(channels, 1e-5);
    }

    public BatchNorm2d(int channels, double eps) {
        if (channels <= 0 || eps < 0) {
            throw new IllegalArgumentException("Invalid BatchNorm2d channels/eps");
        }
        this.channels = channels;
        this.eps = eps;
        this.weight = registerParameter("weight", new Parameter(Torch.ones(Shape.of(channels))));
        this.bias = registerParameter("bias", new Parameter(Torch.zeros(Shape.of(channels))));
        this.runningMean = registerBuffer("running_mean", Torch.zeros(Shape.of(channels)), true);
        this.runningVar = registerBuffer("running_var", Torch.ones(Shape.of(channels)), true);
    }

    @Override
    public Tensor forward(Tensor input) {
        if (isTraining()) {
            throw new JTorchException(
                    "BatchNorm2d training mode is not implemented; call eval() or training(false)");
        }
        return input.batchNorm(weight, bias, runningMean, runningVar, false, 0.0, eps);
    }

    public Parameter weight() {
        return weight;
    }

    public Parameter bias() {
        return bias;
    }

    public Tensor runningMean() {
        return runningMean;
    }

    public Tensor runningVar() {
        return runningVar;
    }
}
