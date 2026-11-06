package org.jtorch.vision.models;

import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.nn.AdaptiveAvgPool2d;
import org.jtorch.nn.BatchNorm2d;
import org.jtorch.nn.Conv2d;
import org.jtorch.nn.Flatten;
import org.jtorch.nn.Linear;
import org.jtorch.nn.Module;
import org.jtorch.nn.ReLU;
import org.jtorch.nn.Sequential;

/**
 * A compact ResNet-class fixture: a stem plus two residual stages (four basic blocks), global average pooling and a
 * linear classifier. The layer names mirror the equivalent PyTorch module tree so Safetensors state dictionaries
 * transfer without renaming.
 */
public final class ResNet9 extends Module {
    private final Sequential stem;
    private final Sequential layer1;
    private final Sequential layer2;
    private final AdaptiveAvgPool2d pool;
    private final Flatten flatten;
    private final Linear fc;

    public ResNet9() {
        this(16, 10);
    }

    public ResNet9(int width, int classes) {
        if (width <= 0 || classes <= 0) {
            throw new IllegalArgumentException("ResNet9 width and classes must be positive");
        }
        this.stem = registerModule("stem", new Sequential(
                new Conv2d(3, width, new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1},
                        new int[] {1, 1}, 1, false),
                new BatchNorm2d(width),
                new ReLU()));
        this.layer1 = registerModule("layer1", new Sequential(
                new BasicBlock(width, width, 1),
                new BasicBlock(width, width, 1)));
        this.layer2 = registerModule("layer2", new Sequential(
                new BasicBlock(width, 2 * width, 2),
                new BasicBlock(2 * width, 2 * width, 1)));
        this.pool = registerModule("pool", new AdaptiveAvgPool2d(1));
        this.flatten = registerModule("flatten", new Flatten());
        this.fc = registerModule("fc", new Linear(2 * width, classes));
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor value = stem.call(input);
        value = layer1.call(value);
        value = layer2.call(value);
        value = pool.call(value);
        value = flatten.call(value);
        return fc.call(value);
    }

    /** One residual basic block: two 3x3 convolutions plus an optional 1x1 projection shortcut. */
    public static final class BasicBlock extends Module {
        private final Conv2d conv1;
        private final BatchNorm2d bn1;
        private final Conv2d conv2;
        private final BatchNorm2d bn2;
        private final Module downsample;

        public BasicBlock(int inChannels, int outChannels, int stride) {
            this.conv1 = registerModule("conv1", new Conv2d(inChannels, outChannels,
                    new int[] {3, 3}, new int[] {stride, stride}, new int[] {1, 1},
                    new int[] {1, 1}, 1, false));
            this.bn1 = registerModule("bn1", new BatchNorm2d(outChannels));
            this.conv2 = registerModule("conv2", new Conv2d(outChannels, outChannels,
                    new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1},
                    new int[] {1, 1}, 1, false));
            this.bn2 = registerModule("bn2", new BatchNorm2d(outChannels));
            this.downsample = inChannels == outChannels && stride == 1
                    ? null
                    : registerModule("downsample", new Sequential(
                            new Conv2d(inChannels, outChannels, new int[] {1, 1},
                                    new int[] {stride, stride}, new int[] {0, 0},
                                    new int[] {1, 1}, 1, false),
                            new BatchNorm2d(outChannels)));
        }

        @Override
        public Tensor forward(Tensor input) {
            Tensor identity = downsample == null ? input : downsample.call(input);
            Tensor value = conv1.call(input);
            value = bn1.call(value).relu();
            value = conv2.call(value);
            value = bn2.call(value);
            return value.add(identity).relu();
        }
    }
}
