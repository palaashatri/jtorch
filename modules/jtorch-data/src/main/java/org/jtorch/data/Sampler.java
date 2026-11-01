package org.jtorch.data;

/** Deterministic index sampler used by {@link DataLoader}. */
public interface Sampler {
    int[] indices(int datasetSize, int epoch);
}
