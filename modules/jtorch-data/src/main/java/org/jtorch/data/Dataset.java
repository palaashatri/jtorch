package org.jtorch.data;

/** An indexable dataset. */
public interface Dataset<T> {
    int size();

    T get(int index);
}
