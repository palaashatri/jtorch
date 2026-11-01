package org.jtorch.data;

/** Visits every dataset index once, in order. */
public final class SequentialSampler implements Sampler {
    public SequentialSampler() {}

    @Override
    public int[] indices(int datasetSize, int epoch) {
        int[] order = new int[datasetSize];
        for (int i = 0; i < datasetSize; i++) {
            order[i] = i;
        }
        return order;
    }
}
