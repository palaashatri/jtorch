package org.jtorch.demo.vision;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;

/** Build-time helper that builds the Vision Inspector app image and runs it headlessly. */
public final class PackageTool {
    private PackageTool() {}

    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length != 3) {
            throw new IllegalArgumentException("usage: PackageTool <inputDir> <destDir> <report>");
        }
        Path input = Path.of(args[0]).toAbsolutePath();
        Path destination = Path.of(args[1]).toAbsolutePath();
        Path report = Path.of(args[2]).toAbsolutePath();
        boolean windows = System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT).contains("win");
        Path jpackage = Path.of(System.getProperty("java.home"), "bin",
                windows ? "jpackage.exe" : "jpackage");
        if (!Files.isExecutable(jpackage)) {
            throw new IllegalStateException("jpackage not found at " + jpackage);
        }
        deleteRecursively(destination.resolve("VisionInspector"));
        deleteRecursively(destination.resolve("VisionInspector.app"));
        Files.createDirectories(destination);
        Files.createDirectories(report.getParent());
        run(
                jpackage.toString(),
                "--type", "app-image",
                "--name", "VisionInspector",
                "--input", input.toString(),
                "--main-jar", "vision-inspector.jar",
                "--main-class", "org.jtorch.demo.vision.VisionInspectorApp",
                "--dest", destination.toString());
        String os = System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT);
        Path launcher = os.contains("mac")
                ? destination.resolve("VisionInspector.app/Contents/MacOS/VisionInspector")
                : os.contains("win")
                        ? destination.resolve("VisionInspector/VisionInspector.exe")
                        : destination.resolve("VisionInspector/bin/VisionInspector");
        if (!Files.exists(launcher)) {
            throw new IllegalStateException("jpackage did not produce " + launcher);
        }
        run(launcher.toString(), "--smoke-test", "--headless", "--output", report.toString());
        System.out.println("Packaged Vision Inspector smoke test passed via " + launcher);
    }

    private static void run(String... command) throws IOException, InterruptedException {
        Process process = new ProcessBuilder(List.of(command)).inheritIO().start();
        int code = process.waitFor();
        if (code != 0) {
            throw new IllegalStateException("Command failed with exit code " + code + ": "
                    + String.join(" ", command));
        }
    }

    private static void deleteRecursively(Path path) throws IOException {
        if (!Files.exists(path)) {
            return;
        }
        try (var walk = Files.walk(path)) {
            for (Path entry : walk.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(entry);
            }
        }
    }
}
