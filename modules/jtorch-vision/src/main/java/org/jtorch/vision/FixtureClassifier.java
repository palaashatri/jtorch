package org.jtorch.vision;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.Tensor;
import org.jtorch.nn.StateDict;
import org.jtorch.serialization.Safetensors;
import org.jtorch.vision.models.ResNet9;

/** Loads the pinned ResNet-class vision fixture and exposes classification and saliency. */
public final class FixtureClassifier {
    public static final int IMAGE_SIZE = 32;
    public static final List<String> LABELS = List.of("red", "green", "blue", "checker");

    private final ResNet9 model;

    public FixtureClassifier() {
        this.model = new ResNet9(8, 4);
        byte[] weights = resource("/fixtures/resnet9-weights.safetensors");
        model.loadStateDict(new StateDict(Safetensors.load(weights)), true);
        model.eval();
    }

    public List<Classification> classify(Tensor image) {
        requireImage(image);
        Tensor logits = Torch.noGrad(() -> model.call(image).select(0, 0));
        return Classification.topK(logits, LABELS, LABELS.size());
    }

    public Tensor saliency(Tensor image, int classIndex) {
        requireImage(image);
        return Saliency.heatmap(Saliency.inputGradient(model, image, classIndex));
    }

    public ResNet9 model() {
        return model;
    }

    private static void requireImage(Tensor image) {
        if (image.dim() != 4 || image.size(0) != 1 || image.size(1) != 3
                || image.size(2) != IMAGE_SIZE || image.size(3) != IMAGE_SIZE) {
            throw new IllegalArgumentException("Expected a [1, 3, " + IMAGE_SIZE + ", " + IMAGE_SIZE + "] image");
        }
    }

    private static byte[] resource(String name) {
        try (InputStream stream = FixtureClassifier.class.getResourceAsStream(name)) {
            if (stream == null) {
                throw new IllegalStateException("Missing fixture resource " + name);
            }
            return stream.readAllBytes();
        } catch (IOException error) {
            throw new IllegalStateException("Unable to read fixture resource " + name, error);
        }
    }

    public static void main(String[] args) {
        FixtureClassifier classifier = new FixtureClassifier();
        System.out.println("Loaded fixture classifier with labels " + FixtureClassifier.LABELS
                + " and " + classifier.model.stateDict().size() + " state entries");
    }
}
