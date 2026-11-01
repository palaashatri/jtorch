package org.jtorch.cpu;

import java.util.SplittableRandom;

/**
 * Default CPU random generator. PyTorch does not promise bit-identical streams across implementations, so JTorch
 * promises reproducibility for a given seed and release, verified by statistical and seed-reproducibility tests.
 */
public final class CpuRandom {
    private static final Object LOCK = new Object();
    private static SplittableRandom random = new SplittableRandom(0x4A544F524348L);
    private static double cachedNormal;
    private static boolean hasCachedNormal;

    private CpuRandom() {}

    public static void manualSeed(long seed) {
        synchronized (LOCK) {
            random = new SplittableRandom(seed);
            cachedNormal = 0;
            hasCachedNormal = false;
        }
    }

    public static double nextDouble() {
        synchronized (LOCK) {
            return random.nextDouble();
        }
    }

    public static long nextLong(long bound) {
        synchronized (LOCK) {
            return random.nextLong(bound);
        }
    }

    /** Standard normal sample using the polar Box-Muller method. */
    public static double nextNormal() {
        synchronized (LOCK) {
            if (hasCachedNormal) {
                hasCachedNormal = false;
                return cachedNormal;
            }
            double u;
            double v;
            double squared;
            do {
                u = 2.0 * random.nextDouble() - 1.0;
                v = 2.0 * random.nextDouble() - 1.0;
                squared = u * u + v * v;
            } while (squared >= 1.0 || squared == 0.0);
            double factor = Math.sqrt(-2.0 * Math.log(squared) / squared);
            cachedNormal = v * factor;
            hasCachedNormal = true;
            return u * factor;
        }
    }
}
