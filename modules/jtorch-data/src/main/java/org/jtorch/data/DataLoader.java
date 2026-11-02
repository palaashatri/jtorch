package org.jtorch.data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.SplittableRandom;
import org.jtorch.core.Tensor;

/**
 * Iterable batch loader with deterministic shuffling. Each batch stacks the row tensors of its samples along a new
 * leading dimension.
 */
public final class DataLoader implements Iterable<Sample> {
    private final Dataset<Sample> dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final long seed;
    private final Sampler sampler;
    private int epoch;

    public DataLoader(Dataset<Sample> dataset, int batchSize) {
        this(dataset, batchSize, false, false, 0);
    }

    public DataLoader(Dataset<Sample> dataset, int batchSize, boolean shuffle, boolean dropLast, long seed) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.seed = seed;
        this.sampler = null;
    }

    /** Uses an explicit sampler instead of the shuffle flag. */
    public DataLoader(Dataset<Sample> dataset, int batchSize, Sampler sampler, boolean dropLast) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.shuffle = false;
        this.dropLast = dropLast;
        this.seed = 0;
        this.sampler = sampler;
    }

    public int batchCount() {
        int total = dataset.size();
        int full = total / batchSize;
        return dropLast || total % batchSize == 0 ? full : full + 1;
    }

    public void setEpoch(int value) {
        if (value < 0) {
            throw new IllegalArgumentException("epoch must be non-negative");
        }
        this.epoch = value;
    }

    public int batchSize() {
        return batchSize;
    }

    /** Returns a bounded asynchronous prefetch wrapper around this loader. */
    public PrefetchingDataLoader prefetch(int depth) {
        return new PrefetchingDataLoader(this, depth);
    }

    @Override
    public Iterator<Sample> iterator() {
        int[] order;
        if (sampler != null) {
            order = sampler.indices(dataset.size(), epoch);
            if (order.length != dataset.size()) {
                throw new IllegalArgumentException("Sampler returned " + order.length + " indices for a dataset of "
                        + dataset.size());
            }
            boolean[] seen = new boolean[dataset.size()];
            for (int index : order) {
                if (index < 0 || index >= dataset.size() || seen[index]) {
                    throw new IllegalArgumentException("Sampler produced an invalid or duplicate index " + index);
                }
                seen[index] = true;
            }
        } else {
            order = new int[dataset.size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            if (shuffle) {
                SplittableRandom random = new SplittableRandom(seed + 0x9E3779B97F4A7C15L * (epoch + 1));
                for (int i = order.length - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int swap = order[i];
                    order[i] = order[j];
                    order[j] = swap;
                }
            }
        }
        return new Iterator<>() {
            private int cursor;

            @Override
            public boolean hasNext() {
                int remaining = order.length - cursor;
                return dropLast ? remaining >= batchSize : remaining > 0;
            }

            @Override
            public Sample next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                int count = Math.min(batchSize, order.length - cursor);
                Sample[] samples = new Sample[count];
                for (int i = 0; i < count; i++) {
                    samples[i] = dataset.get(order[cursor + i]);
                }
                cursor += count;
                return collate(samples);
            }
        };
    }

    private static Sample collate(Sample[] samples) {
        int arity = samples[0].arity();
        Tensor[] batch = new Tensor[arity];
        for (int column = 0; column < arity; column++) {
            List<Tensor> rows = new ArrayList<>(samples.length);
            for (Sample sample : samples) {
                if (sample.arity() != arity) {
                    throw new IllegalArgumentException("Inconsistent sample arity in dataset");
                }
                rows.add(sample.tensor(column));
            }
            batch[column] = Tensor.stack(rows, 0);
        }
        return new Sample(batch);
    }
}
