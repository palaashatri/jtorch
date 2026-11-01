package org.jtorch.data;

import java.util.SplittableRandom;

/** Visits every dataset index once in a seed- and epoch-deterministic shuffled order. */
public final class RandomSampler implements Sampler {
    private final long seed;

    public RandomSampler(long seed) {
        this.seed = seed;
    }

    @Override
    public int[] indices(int datasetSize, int epoch) {
        if (epoch < 0) {
            throw new IllegalArgumentException("epoch must be non-negative");
        }
        int[] order = new int[datasetSize];
        for (int i = 0; i < datasetSize; i++) {
            order[i] = i;
        }
        SplittableRandom random = new SplittableRandom(seed + 0x9E3779B97F4A7C15L * (epoch + 1));
        for (int i = order.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int swap = order[i];
            order[i] = order[j];
            order[j] = swap;
        }
        return order;
    }
}
