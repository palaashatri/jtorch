package org.jtorch.core;

import java.util.concurrent.atomic.AtomicLong;

/** A version counter shared by every tensor and view that aliases the same storage. */
public final class VersionCounter {
    private final AtomicLong version = new AtomicLong();

    public VersionCounter() {}

    public long current() {
        return version.get();
    }

    void bump() {
        version.incrementAndGet();
    }

    @Override
    public String toString() {
        return "VersionCounter(" + current() + ")";
    }
}
