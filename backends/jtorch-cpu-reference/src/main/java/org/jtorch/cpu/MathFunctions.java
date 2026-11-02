package org.jtorch.cpu;

/**
 * Scalar math helpers that are not present in {@link java.lang.Math}. The error function uses a Maclaurin series on
 * {@code |x| < 2} and a continued fraction for the complementary error function above that threshold; the maximum
 * observed absolute error against the C library {@code erf} is about one double-precision epsilon.
 */
final class MathFunctions {
    private static final double SQRT_PI = Math.sqrt(Math.PI);
    private static final double INV_SQRT_2 = 1.0 / Math.sqrt(2.0);
    private static final double INV_SQRT_2PI = 1.0 / Math.sqrt(2.0 * Math.PI);
    private static final double TANH_SCALE = Math.sqrt(2.0 / Math.PI);

    private MathFunctions() {}

    static double erf(double x) {
        if (x == 0.0 || Double.isNaN(x)) {
            return x;
        }
        if (Double.isInfinite(x)) {
            return x > 0 ? 1.0 : -1.0;
        }
        double absolute = Math.abs(x);
        double result;
        if (absolute < 2.0) {
            double term = absolute;
            double sum = absolute;
            for (int n = 1; n < 400; n++) {
                term *= -absolute * absolute / n;
                double addition = term / (2 * n + 1);
                sum += addition;
                if (Math.abs(addition) < 1e-18 * Math.abs(sum)) {
                    break;
                }
            }
            result = 2.0 / SQRT_PI * sum;
        } else {
            result = 1.0 - erfcContinuedFraction(absolute);
        }
        return x < 0 ? -result : result;
    }

    private static double erfcContinuedFraction(double x) {
        double tiny = 1e-300;
        double f = x;
        double c = x;
        double d = 0.0;
        for (int n = 1; n < 400; n++) {
            double a = n / 2.0;
            d = x + a * d;
            if (d == 0.0) {
                d = tiny;
            }
            d = 1.0 / d;
            c = x + a / c;
            if (c == 0.0) {
                c = tiny;
            }
            double delta = c * d;
            f *= delta;
            if (Math.abs(delta - 1.0) < 1e-16) {
                break;
            }
        }
        return Math.exp(-x * x) / (SQRT_PI * f);
    }

    static double gelu(double x, boolean tanhApproximation) {
        if (tanhApproximation) {
            double inner = TANH_SCALE * (x + 0.044715 * x * x * x);
            return 0.5 * x * (1.0 + Math.tanh(inner));
        }
        return 0.5 * x * (1.0 + erf(x * INV_SQRT_2));
    }

    static double geluDerivative(double x, boolean tanhApproximation) {
        if (tanhApproximation) {
            double inner = TANH_SCALE * (x + 0.044715 * x * x * x);
            double tanh = Math.tanh(inner);
            double innerDerivative = TANH_SCALE * (1.0 + 3.0 * 0.044715 * x * x);
            return 0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * innerDerivative;
        }
        return 0.5 * (1.0 + erf(x * INV_SQRT_2)) + x * Math.exp(-0.5 * x * x) * INV_SQRT_2PI;
    }
}
