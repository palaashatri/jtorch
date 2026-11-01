package org.jtorch.data;

import java.util.SplittableRandom;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;

/**
 * Deterministic ten-class synthetic digit-family dataset used by tests and demos when no MNIST IDX files are
 * available. It is explicitly synthetic: use {@link IdxReader} for real MNIST data.
 */
public final class SyntheticDigits {
    public static final int FEATURES = 64;
    public static final int CLASSES = 10;

    private SyntheticDigits() {}

    public static TensorDataset create(int samples, long seed) {
        if (samples <= 0) {
            throw new IllegalArgumentException("samples must be positive");
        }
        SplittableRandom random = new SplittableRandom(seed);
        double[][] prototypes = new double[CLASSES][FEATURES];
        for (int label = 0; label < CLASSES; label++) {
            for (int feature = 0; feature < FEATURES; feature++) {
                prototypes[label][feature] = random.nextDouble();
            }
        }
        double[][] images = new double[samples][FEATURES];
        long[] labels = new long[samples];
        for (int sample = 0; sample < samples; sample++) {
            int label = random.nextInt(CLASSES);
            labels[sample] = label;
            for (int feature = 0; feature < FEATURES; feature++) {
                double value = prototypes[label][feature] + normal(random) * 0.12;
                images[sample][feature] = Math.max(0, Math.min(1, value));
            }
        }
        float[] imageData = new float[samples * FEATURES];
        for (int sample = 0; sample < samples; sample++) {
            for (int feature = 0; feature < FEATURES; feature++) {
                imageData[sample * FEATURES + feature] = (float) images[sample][feature];
            }
        }
        return new TensorDataset(
                Torch.tensor(imageData, Shape.of(samples, FEATURES)),
                Torch.tensor(labels, Shape.of(samples)));
    }

    private static double normal(SplittableRandom random) {
        double u = random.nextDouble();
        double v = random.nextDouble();
        while (u == 0) {
            u = random.nextDouble();
        }
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
}
