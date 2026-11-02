package org.jtorch.data;

import java.util.ArrayList;
import java.util.List;

/** Splits a sampler's index order into batches. */
public final class BatchSampler {
    private final Sampler sampler;
    private final int batchSize;
    private final boolean dropLast;

    public BatchSampler(Sampler sampler, int batchSize, boolean dropLast) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        this.sampler = sampler;
        this.batchSize = batchSize;
        this.dropLast = dropLast;
    }

    public List<int[]> batches(int datasetSize, int epoch) {
        int[] order = sampler.indices(datasetSize, epoch);
        List<int[]> batches = new ArrayList<>();
        for (int start = 0; start < order.length; start += batchSize) {
            int length = Math.min(batchSize, order.length - start);
            if (dropLast && length < batchSize) {
                break;
            }
            int[] batch = new int[length];
            System.arraycopy(order, start, batch, 0, length);
            batches.add(batch);
        }
        return batches;
    }
}
