package org.jtorch.demo.vision;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import javax.imageio.ImageIO;
import javax.swing.SwingUtilities;
import org.jtorch.core.Tensor;
import org.jtorch.json.Json;
import org.jtorch.vision.Classification;
import org.jtorch.vision.FixtureClassifier;
import org.jtorch.vision.ImageTensors;

/** Vision Inspector: local image classification, batch classification and gradient saliency. */
public final class VisionInspectorApp {
    private VisionInspectorApp() {}

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        if (options.smokeTest) {
            runSmoke(options.output);
            return;
        }
        FixtureClassifier classifier = new FixtureClassifier();
        if (options.headless) {
            if (options.image != null) {
                classifyImage(classifier, options.image, options);
            } else if (options.folder != null) {
                classifyFolder(classifier, options.folder, options);
            } else {
                throw new IllegalArgumentException("Headless mode needs --image or --folder");
            }
            return;
        }
        SwingUtilities.invokeLater(() -> new VisionInspectorWindow(classifier).show());
    }

    private static void runSmoke(Path output) throws IOException {
        FixtureClassifier classifier = new FixtureClassifier();
        List<String> expected = List.of("red", "green", "blue", "checker");
        List<Map<String, Object>> results = new ArrayList<>();
        int correct = 0;
        for (int label = 0; label < expected.size(); label++) {
            Tensor input = ImageTensors.fromImage(SyntheticImages.pattern(label), FixtureClassifier.IMAGE_SIZE);
            List<Classification> top = classifier.classify(input);
            if (top.get(0).label().equals(expected.get(label))) {
                correct++;
            }
            Tensor heatmap = classifier.saliency(input, top.get(0).index());
            requireFiniteSaliency(heatmap);
            results.add(Map.of(
                    "expected", expected.get(label),
                    "top1", top.get(0).label(),
                    "confidence", top.get(0).confidence(),
                    "saliencyShape", List.of(heatmap.size(0), heatmap.size(1), heatmap.size(2))));
        }
        Path corrupt = Files.createTempFile("vision-inspector-corrupt", ".png");
        Files.write(corrupt, new byte[] {1, 2, 3, 4});
        boolean corruptRejected = false;
        try {
            ImageTensors.fromPath(corrupt, FixtureClassifier.IMAGE_SIZE);
        } catch (IllegalArgumentException expectedError) {
            corruptRejected = true;
        } finally {
            Files.deleteIfExists(corrupt);
        }
        if (!corruptRejected) {
            throw new IllegalStateException("Corrupt image was not rejected");
        }
        if (correct != expected.size()) {
            throw new IllegalStateException("Fixture accuracy " + correct + "/" + expected.size());
        }
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("model", "resnet9-fixture");
        report.put("labels", expected);
        report.put("images", results);
        report.put("corruptRejected", true);
        report.put("top1Accuracy", (double) correct / expected.size());
        writeJson(output, report);
        System.out.println("SMOKE vision-inspector PASS images=" + expected.size() + " top1=" + correct + "/"
                + expected.size() + " saliency=finite corrupt=rejected");
    }

    private static void classifyImage(FixtureClassifier classifier, Path image, Options options) throws IOException {
        Tensor input = ImageTensors.fromPath(image, FixtureClassifier.IMAGE_SIZE);
        List<Classification> top = classifier.classify(input);
        for (Classification result : top.subList(0, Math.min(options.topK, top.size()))) {
            System.out.printf(Locale.ROOT, "%s %.6f%n", result.label(), result.confidence());
        }
        if (options.saliency != null) {
            Tensor heatmap = classifier.saliency(input, top.get(0).index());
            requireFiniteSaliency(heatmap);
            BufferedImage base = readImage(image);
            BufferedImage overlay = VisionOverlay.blend(base, heatmap);
            ImageIO.write(overlay, "png", options.saliency.toFile());
        }
        if (options.output != null) {
            Map<String, Object> report = new LinkedHashMap<>();
            report.put("image", image.toString());
            report.put("topK", top.stream().limit(options.topK)
                    .map(result -> Map.of("label", result.label(), "confidence", result.confidence()))
                    .toList());
            writeJson(options.output, report);
        }
    }

    private static void classifyFolder(FixtureClassifier classifier, Path folder, Options options) throws IOException {
        List<Map<String, Object>> rows = new ArrayList<>();
        List<Map<String, Object>> errors = new ArrayList<>();
        try (var files = Files.list(folder)) {
            for (Path path : files.filter(VisionInspectorApp::isImage).sorted().toList()) {
                try {
                    Tensor input = ImageTensors.fromPath(path, FixtureClassifier.IMAGE_SIZE);
                    List<Classification> top = classifier.classify(input);
                    rows.add(Map.of(
                            "path", path.toString(),
                            "top1", top.get(0).label(),
                            "confidence", top.get(0).confidence()));
                } catch (RuntimeException error) {
                    errors.add(Map.of("path", path.toString(), "error", error.getMessage()));
                }
            }
        }
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("folder", folder.toString());
        report.put("images", rows);
        report.put("errors", errors);
        writeJson(options.output, report);
        if (options.output != null) {
            writeCsv(options.output.resolveSibling(options.output.getFileName() + ".csv"), rows);
        }
        System.out.println("BATCH vision-inspector images=" + rows.size() + " errors=" + errors.size());
    }

    private static boolean isImage(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")
                || name.endsWith(".bmp") || name.endsWith(".gif");
    }

    private static BufferedImage readImage(Path path) throws IOException {
        BufferedImage image = ImageIO.read(path.toFile());
        if (image == null) {
            throw new IllegalArgumentException("Unsupported or corrupt image " + path);
        }
        return image;
    }

    private static void requireFiniteSaliency(Tensor heatmap) {
        for (float value : heatmap.toFloatArray()) {
            if (!Float.isFinite(value) || value < 0.0f || value > 1.0f) {
                throw new IllegalStateException("Saliency contains an invalid value: " + value);
            }
        }
    }

    private static void writeCsv(Path output, List<Map<String, Object>> rows) throws IOException {
        StringBuilder csv = new StringBuilder("path,top1,confidence\n");
        for (Map<String, Object> row : rows) {
            csv.append('"').append(String.valueOf(row.get("path")).replace("\"", "\"\"")).append('"')
                    .append(',').append(row.get("top1")).append(',')
                    .append(row.get("confidence")).append('\n');
        }
        Files.writeString(output, csv.toString());
    }

    private static void writeJson(Path output, Object report) throws IOException {
        if (output == null) {
            return;
        }
        if (output.getParent() != null) {
            Files.createDirectories(output.getParent());
        }
        Files.writeString(output, Json.writePretty(report) + "\n");
    }

    private record Options(
            boolean smokeTest,
            boolean headless,
            Path image,
            Path folder,
            Path output,
            Path saliency,
            int topK) {
        static Options parse(String[] args) {
            boolean smokeTest = false;
            boolean headless = false;
            Path image = null;
            Path folder = null;
            Path output = null;
            Path saliency = null;
            int topK = 5;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--smoke-test" -> smokeTest = true;
                    case "--headless" -> headless = true;
                    case "--image" -> image = Path.of(args[++i]);
                    case "--folder" -> folder = Path.of(args[++i]);
                    case "--output" -> output = Path.of(args[++i]);
                    case "--saliency" -> saliency = Path.of(args[++i]);
                    case "--top-k" -> topK = Integer.parseInt(args[++i]);
                    default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
                }
            }
            return new Options(smokeTest, headless, image, folder, output, saliency, topK);
        }
    }
}
