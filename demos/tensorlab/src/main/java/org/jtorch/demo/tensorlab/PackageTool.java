package org.jtorch.demo.tensorlab;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;

/**
 * Build-time packaging helper invoked by the {@code demoPackage} Gradle task. It runs jpackage and then executes the
 * packaged launcher's headless smoke test, so packaging success is measured by running the real app image.
 */
public final class PackageTool {
    private PackageTool() {}

    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length != 3) {
            throw new IllegalArgumentException("usage: PackageTool <inputDir> <destDir> <checkpoint>");
        }
        Path input = Path.of(args[0]).toAbsolutePath();
        Path destination = Path.of(args[1]).toAbsolutePath();
        Path checkpoint = Path.of(args[2]).toAbsolutePath();
        Path jpackage = Path.of(System.getProperty("java.home"), "bin", "jpackage");
        if (!Files.isExecutable(jpackage)) {
            throw new IllegalStateException("jpackage not found at " + jpackage);
        }
        deleteRecursively(destination.resolve("TensorLab"));
        deleteRecursively(destination.resolve("TensorLab.app"));
        Files.createDirectories(destination);
        Files.createDirectories(checkpoint.getParent());
        run(
                jpackage.toString(),
                "--type", "app-image",
                "--name", "TensorLab",
                "--input", input.toString(),
                "--main-jar", "tensorlab.jar",
                "--main-class", "org.jtorch.demo.tensorlab.TensorLabApp",
                "--dest", destination.toString());
        String os = System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT);
        Path launcher = os.contains("mac")
                ? destination.resolve("TensorLab.app/Contents/MacOS/TensorLab")
                : os.contains("win")
                        ? destination.resolve("TensorLab/TensorLab.exe")
                        : destination.resolve("TensorLab/bin/TensorLab");
        if (!Files.exists(launcher)) {
            throw new IllegalStateException("jpackage did not produce " + launcher);
        }
        run(
                launcher.toString(),
                "--smoke-test",
                "--headless",
                "--checkpoint",
                checkpoint.toString());
        System.out.println("Packaged TensorLab smoke test passed via " + launcher);
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
