package org.jtorch.core;

import java.util.Locale;
import java.util.Objects;

/** A scalar argument with PyTorch's integer/float/boolean distinction preserved. */
public final class Scalar {
    public enum Kind {
        BOOL,
        INT,
        FLOAT
    }

    private final Kind kind;
    private final long longValue;
    private final double doubleValue;
    private final boolean boolValue;

    private Scalar(Kind kind, long longValue, double doubleValue, boolean boolValue) {
        this.kind = kind;
        this.longValue = longValue;
        this.doubleValue = doubleValue;
        this.boolValue = boolValue;
    }

    public static Scalar of(long value) {
        return new Scalar(Kind.INT, value, value, false);
    }

    public static Scalar of(int value) {
        return of((long) value);
    }

    public static Scalar of(double value) {
        return new Scalar(Kind.FLOAT, (long) value, value, false);
    }

    public static Scalar of(float value) {
        return of((double) value);
    }

    public static Scalar of(boolean value) {
        return new Scalar(Kind.BOOL, value ? 1 : 0, value ? 1 : 0, value);
    }

    public Kind kind() {
        return kind;
    }

    public boolean isIntegral() {
        return kind == Kind.INT;
    }

    public boolean isFloatingPoint() {
        return kind == Kind.FLOAT;
    }

    public boolean isBoolean() {
        return kind == Kind.BOOL;
    }

    public long toLong() {
        return switch (kind) {
            case BOOL -> boolValue ? 1 : 0;
            case INT -> longValue;
            case FLOAT -> (long) doubleValue;
        };
    }

    public int toInt() {
        return Math.toIntExact(toLong());
    }

    public double toDouble() {
        return switch (kind) {
            case BOOL -> boolValue ? 1 : 0;
            case INT -> (double) longValue;
            case FLOAT -> doubleValue;
        };
    }

    public float toFloat() {
        return (float) toDouble();
    }

    public boolean toBoolean() {
        return switch (kind) {
            case BOOL -> boolValue;
            case INT -> longValue != 0;
            case FLOAT -> doubleValue != 0;
        };
    }

    int kindRank() {
        return switch (kind) {
            case BOOL -> 0;
            case INT -> 1;
            case FLOAT -> 2;
        };
    }

    /** Converts this scalar to the given dtype using PyTorch's truncating casts. */
    public long toLong(DType dtype) {
        if (dtype.isFloatingPoint() || dtype.isComplex()) {
            return (long) toDouble();
        }
        if (dtype == DType.BOOL) {
            return toBoolean() ? 1 : 0;
        }
        return toLong();
    }

    /** Converts this scalar to the given dtype using PyTorch's truncating casts. */
    public double toDouble(DType dtype) {
        if (dtype.isFloatingPoint() || dtype.isComplex()) {
            // Wrapped scalars promote to the real part of a complex target (imaginary 0).
            return toDouble();
        }
        if (dtype == DType.BOOL) {
            return toBoolean() ? 1 : 0;
        }
        return toLong();
    }

    public static Scalar parse(String text) {
        String trimmed = text.trim().toLowerCase(Locale.ROOT);
        if (trimmed.equals("true")) {
            return of(true);
        }
        if (trimmed.equals("false")) {
            return of(false);
        }
        try {
            if (trimmed.contains(".") || trimmed.contains("e")) {
                return of(Double.parseDouble(trimmed));
            }
            return of(Long.parseLong(trimmed));
        } catch (NumberFormatException error) {
            throw new DTypeException("Invalid scalar '" + text + "'");
        }
    }

    @Override
    public String toString() {
        return switch (kind) {
            case BOOL -> Boolean.toString(boolValue);
            case INT -> Long.toString(longValue);
            case FLOAT -> Double.toString(doubleValue);
        };
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof Scalar scalar)) {
            return false;
        }
        if (kind != scalar.kind) {
            return false;
        }
        return switch (kind) {
            case BOOL -> boolValue == scalar.boolValue;
            case INT -> longValue == scalar.longValue;
            case FLOAT -> Double.compare(doubleValue, scalar.doubleValue) == 0;
        };
    }

    @Override
    public int hashCode() {
        return switch (kind) {
            case BOOL -> Objects.hash(kind, boolValue);
            case INT -> Objects.hash(kind, longValue);
            case FLOAT -> Objects.hash(kind, doubleValue);
        };
    }
}
