package org.jtorch.core.memory;

/** Provenance of a storage allocation. */
public enum StorageKind {
    /** FFM native memory owned by an {@code Arena}. */
    NATIVE,
    /** A file-backed mapped region. */
    MAPPED
}
