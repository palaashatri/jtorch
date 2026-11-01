package org.jtorch.data;

import java.util.ArrayList;
import java.util.List;
import org.jtorch.core.Tensor;

/** A dataset backed by tensor columns that share their leading dimension. */
public final class TensorDataset implements Dataset<Sample> {
    private final List<Tensor> columns;
    private final int size;

    public TensorDataset(Tensor... columns) {
        if (columns.length == 0) {
            throw new IllegalArgumentException("TensorDataset requires at least one column");
        }
        for (Tensor column : columns) {
            if (column.dim() == 0) {
                throw new IllegalArgumentException("TensorDataset columns must have at least one dimension");
            }
        }
        long firstSize = columns[0].size(0);
        for (Tensor column : columns) {
            if (column.size(0) != firstSize) {
                throw new IllegalArgumentException(
                        "All TensorDataset columns must share their first dimension");
            }
        }
        if (firstSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("TensorDataset is too large for a Java index");
        }
        this.columns = List.of(columns.clone());
        this.size = (int) firstSize;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Sample get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Dataset index " + index + " out of range [0, " + size + ")");
        }
        Tensor[] row = new Tensor[columns.size()];
        for (int i = 0; i < row.length; i++) {
            row[i] = columns.get(i).select(0, index);
        }
        return new Sample(row);
    }

    public List<Tensor> columns() {
        return columns;
    }
}
