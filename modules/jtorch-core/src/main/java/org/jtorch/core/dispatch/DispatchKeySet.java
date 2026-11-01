package org.jtorch.core.dispatch;

import java.util.EnumSet;

/** An immutable set of dispatch keys. */
public final class DispatchKeySet {
    private static final DispatchKey[] VALUES = DispatchKey.values();
    private static final DispatchKeySet EMPTY = new DispatchKeySet(0L);

    private final long bits;

    private DispatchKeySet(long bits) {
        this.bits = bits;
    }

    public static DispatchKeySet empty() {
        return EMPTY;
    }

    public static DispatchKeySet of(DispatchKey... keys) {
        long bits = 0;
        for (DispatchKey key : keys) {
            bits |= 1L << key.ordinal();
        }
        return bits == 0 ? EMPTY : new DispatchKeySet(bits);
    }

    public static DispatchKeySet of(EnumSet<DispatchKey> keys) {
        long bits = 0;
        for (DispatchKey key : keys) {
            bits |= 1L << key.ordinal();
        }
        return bits == 0 ? EMPTY : new DispatchKeySet(bits);
    }

    public DispatchKeySet union(DispatchKey key) {
        return new DispatchKeySet(bits | (1L << key.ordinal()));
    }

    public DispatchKeySet union(DispatchKeySet other) {
        return new DispatchKeySet(bits | other.bits);
    }

    public DispatchKeySet without(DispatchKey key) {
        return new DispatchKeySet(bits & ~(1L << key.ordinal()));
    }

    public boolean has(DispatchKey key) {
        return (bits & (1L << key.ordinal())) != 0;
    }

    public boolean isEmpty() {
        return bits == 0;
    }

    /** The highest-priority key present, or {@code null} for an empty set. */
    public DispatchKey highest() {
        DispatchKey best = null;
        for (DispatchKey key : VALUES) {
            if ((bits & (1L << key.ordinal())) != 0 && (best == null || key.priority() > best.priority())) {
                best = key;
            }
        }
        return best;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof DispatchKeySet set && set.bits == bits;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(bits);
    }

    @Override
    public String toString() {
        StringBuilder text = new StringBuilder("DispatchKeySet[");
        boolean first = true;
        DispatchKey[] ordered = VALUES.clone();
        java.util.Arrays.sort(ordered, (left, right) -> Integer.compare(right.priority(), left.priority()));
        for (DispatchKey key : ordered) {
            if ((bits & (1L << key.ordinal())) != 0) {
                if (!first) {
                    text.append(", ");
                }
                text.append(key);
                first = false;
            }
        }
        return text.append(']').toString();
    }
}
