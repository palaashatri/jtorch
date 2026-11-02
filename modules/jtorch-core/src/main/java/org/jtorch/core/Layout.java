package org.jtorch.core;

/**
 * Tensor layout. Only dense strided tensors are implemented; sparse, nested, quantized and other layouts are
 * listed as unsupported in the compatibility ledger rather than exposed as broken enum values.
 */
public enum Layout {
    STRIDED;

    @Override
    public String toString() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
