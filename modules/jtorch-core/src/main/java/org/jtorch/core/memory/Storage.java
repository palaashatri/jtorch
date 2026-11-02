package org.jtorch.core.memory;

import java.lang.foreign.MemorySegment;
import org.jtorch.core.DType;
import org.jtorch.core.Device;

/** A typed region of storage. Views share one {@code Storage} instance and therefore one lifetime. */
public interface Storage {
    /** Number of addressable bytes in this allocation. */
    long byteSize();

    DType dtype();

    Device device();

    StorageKind kind();

    /** Host-addressable memory. Meta storage is absent; CPU storage is always host addressable. */
    MemorySegment segment();
}
