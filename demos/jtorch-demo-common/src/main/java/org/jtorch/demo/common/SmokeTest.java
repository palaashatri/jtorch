package org.jtorch.demo.common;

/** Shared headless smoke-test protocol for demo applications. */
public final class SmokeTest {
    private SmokeTest() {}

    public static boolean hasFlag(String[] args, String flag) {
        for (String arg : args) {
            if (arg.equals(flag)) {
                return true;
            }
        }
        return false;
    }

    public static String option(String[] args, String name, String fallback) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                return args[i + 1];
            }
        }
        return fallback;
    }

    public static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException("Smoke test failed: " + message);
        }
    }

    public static void pass(String application, String details) {
        System.out.println("SMOKE " + application + " PASS " + details);
        System.out.flush();
    }
}
