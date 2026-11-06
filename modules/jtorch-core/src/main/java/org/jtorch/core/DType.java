package org.jtorch.core;

import java.util.Locale;

/**
 * Element types supported by the JTorch CPU reference backend.
 *
 * <p>Real types cover float32/float64/int32/int64/bool. Complex types (complex64 as float32 pairs,
 * complex128 as float64 pairs, interleaved real/imag per element) are supported for FFT foundations; real
 * arithmetic kernels reject complex explicitly rather than silently coercing. Types that the long-term design
 * mentions but this build does not implement (float16, bfloat16, float8, quantized) are deliberately absent:
 * they are recorded as {@code unsupported} in the compatibility ledger instead of being exposed as silently
 * broken options.
 */
public enum DType {
    COMPLEX128("complex128", 16, Kind.COMPLEX),
    COMPLEX64("complex64", 8, Kind.COMPLEX),
    FLOAT64("float64", 8, Kind.FLOAT),
    FLOAT32("float32", 4, Kind.FLOAT),
    INT64("int64", 8, Kind.INT),
    INT32("int32", 4, Kind.INT),
    BOOL("bool", 1, Kind.BOOL);

    private enum Kind {
        BOOL(0),
        INT(1),
        FLOAT(2),
        COMPLEX(3);

        final int rank;

        Kind(int rank) {
            this.rank = rank;
        }
    }

    private final String name;
    private final int sizeBytes;
    private final Kind kind;

    DType(String name, int sizeBytes, Kind kind) {
        this.name = name;
        this.sizeBytes = sizeBytes;
        this.kind = kind;
    }

    /** Lower-case PyTorch-style name, for example {@code float32}. */
    public String typeName() {
        return name;
    }

    public int sizeBytes() {
        return sizeBytes;
    }

    public boolean isFloatingPoint() {
        return kind == Kind.FLOAT;
    }

    public boolean isComplex() {
        return kind == Kind.COMPLEX;
    }

    /** Floating or complex; the leaf types that may require gradients. */
    public boolean isFloatingOrComplex() {
        return kind == Kind.FLOAT || kind == Kind.COMPLEX;
    }

    /** Component float dtype for complex types (complex64→float32, complex128→float64). */
    public DType componentDType() {
        if (this == COMPLEX64) {
            return FLOAT32;
        }
        if (this == COMPLEX128) {
            return FLOAT64;
        }
        throw new DTypeException("Dtype " + this + " has no float component");
    }

    public boolean isIntegral() {
        return kind == Kind.INT;
    }

    public boolean isBoolean() {
        return kind == Kind.BOOL;
    }

    public boolean isSignedInteger() {
        return kind == Kind.INT;
    }

    /** The default floating dtype, matching PyTorch's {@code torch.get_default_dtype()}. */
    public static DType defaultFloat() {
        return FLOAT32;
    }

    /** The default integer dtype for Python-int-like scalars. */
    public static DType defaultInt() {
        return INT64;
    }

    public static DType parse(String text) {
        String normalized = text.trim().toLowerCase(Locale.ROOT);
        if (normalized.startsWith("torch.")) {
            normalized = normalized.substring("torch.".length());
        }
        return switch (normalized) {
            case "float", "float32" -> FLOAT32;
            case "double", "float64" -> FLOAT64;
            case "int", "int32", "int64" -> normalized.equals("int32") ? INT32 : INT64;
            case "long" -> INT64;
            case "bool" -> BOOL;
            case "complex", "complex64", "cfloat" -> COMPLEX64;
            case "complex128", "cdouble" -> COMPLEX128;
            default -> throw new DTypeException("Unknown or unsupported dtype '" + text + "'");
        };
    }

    /**
     * PyTorch type promotion for two tensor dtypes. The supported lattice is
     * bool &lt; int32 &lt; int64 &lt; float32 &lt; float64 &lt; complex64 &lt; complex128
     * (complex64 lifts float32/int/bool; adding float64 to complex64 widens to complex128).
     */
    public static DType resultType(DType a, DType b) {
        if (a == b) {
            return a;
        }
        if (a.isComplex() || b.isComplex()) {
            if (a == COMPLEX128 || b == COMPLEX128 || a == FLOAT64 || b == FLOAT64) {
                return COMPLEX128;
            }
            return COMPLEX64;
        }
        if (a.isFloatingPoint() && b.isFloatingPoint()) {
            return a == FLOAT64 || b == FLOAT64 ? FLOAT64 : FLOAT32;
        }
        if (a.isFloatingPoint()) {
            return a;
        }
        if (b.isFloatingPoint()) {
            return b;
        }
        if (a.isIntegral() && b.isIntegral()) {
            return a == INT64 || b == INT64 ? INT64 : INT32;
        }
        // One side is bool.
        return a.isBoolean() ? b : a;
    }

    /**
     * PyTorch's "wrapped number" promotion: a Java/Python scalar participates as a weak value, so it never widens
     * a tensor to a smaller category but does lift a categorical tensor to the scalar's default dtype.
     */
    public static DType resultTypeWithScalar(DType tensor, Scalar scalar) {
        int scalarRank = scalar.kindRank();
        int tensorRank = categoryRank(tensor);
        if (scalarRank <= tensorRank) {
            return tensor;
        }
        return switch (scalar.kind()) {
            case BOOL -> BOOL;
            case INT -> defaultInt();
            case FLOAT -> defaultFloat();
        };
    }

    private static int categoryRank(DType dtype) {
        if (dtype.isBoolean()) {
            return 0;
        }
        if (dtype.isIntegral()) {
            return 1;
        }
        if (dtype.isFloatingPoint()) {
            return 2;
        }
        return 3;
    }

    @Override
    public String toString() {
        return name;
    }
}
