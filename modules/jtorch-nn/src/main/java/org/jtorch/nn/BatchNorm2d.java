package org.jtorch.nn;

import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/** Batch normalization over NCHW tensors with per-channel running statistics and standard momentum. */
public final class BatchNorm2d extends Module {
    private final Parameter weight;
    private final Parameter bias;
    private final Tensor runningMean;
    private final Tensor runningVar;
    private final Tensor numBatchesTracked;
    private final int channels;
    private final double eps;
    private final double momentum;

    public BatchNorm2d(int channels) {
        this(channels, 1e-5, 0.1);
    }

    public BatchNorm2d(int channels, double eps) {
        this(channels, eps, 0.1);
    }

    public BatchNorm2d(int channels, double eps, double momentum) {
        if (channels < 0) {
            throw new IllegalArgumentException("BatchNorm2d channels must be non-negative");
        }
        this.channels = channels;
        this.eps = eps;
        this.momentum = momentum;
        this.weight = registerParameter("weight", new Parameter(Torch.ones(Shape.of(channels))));
        this.bias = registerParameter("bias", new Parameter(Torch.zeros(Shape.of(channels))));
        this.runningMean = registerBuffer("running_mean", Torch.zeros(Shape.of(channels)), true);
        this.runningVar = registerBuffer("running_var", Torch.ones(Shape.of(channels)), true);
        this.numBatchesTracked = registerBuffer("num_batches_tracked",
                Torch.zeros(Shape.scalar(), org.jtorch.core.TensorOptions.defaults()
                        .dtype(org.jtorch.core.DType.INT64)), true);
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.dim() != 4 || input.size(1) != channels) {
            throw new org.jtorch.core.ShapeException(
                    "BatchNorm2d expects NCHW input with " + channels + " channels, got " + input.shape());
        }
        if (isTraining()) {
            Tensor output = input.batchNorm(weight, bias, runningMean, runningVar, true, momentum, eps);
            org.jtorch.api.Torch.noGrad(() -> numBatchesTracked.add_(1));
            return output;
        }
        return input.batchNorm(weight, bias, runningMean, runningVar, false, momentum, eps);
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

    public Tensor numBatchesTracked() {
        return numBatchesTracked;
    }
}
