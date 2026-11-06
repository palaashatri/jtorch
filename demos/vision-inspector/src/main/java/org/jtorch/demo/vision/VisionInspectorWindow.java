package org.jtorch.demo.vision;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import javax.swing.AbstractAction;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.KeyStroke;
import javax.swing.SwingConstants;
import javax.swing.SwingWorker;
import org.jtorch.core.Tensor;
import org.jtorch.json.Json;
import org.jtorch.vision.Classification;
import org.jtorch.vision.FixtureClassifier;
import org.jtorch.vision.ImageTensors;

/** Swing workbench for local image classification, batch runs and saliency overlays. */
final class VisionInspectorWindow {
    private static final int PREVIEW_SIZE = 320;
    private final FixtureClassifier classifier;
    private final JFrame frame = new JFrame("Vision Inspector — JTorch");
    private final JLabel preview = new JLabel("Open an image or folder", SwingConstants.CENTER);
    private final JTextArea results = new JTextArea();
    private final JLabel status = new JLabel("Ready");
    private Path currentPath;
    private BufferedImage currentImage;
    private Tensor currentTensor;
    private List<Classification> currentTopK = List.of();
    private boolean showingSaliency;
    private boolean running;

    VisionInspectorWindow(FixtureClassifier classifier) {
        this.classifier = classifier;
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getAccessibleContext().setAccessibleName("Vision Inspector");
        frame.getAccessibleContext().setAccessibleDescription(
                "Classifies local images, runs batch folders and overlays gradient saliency.");
        preview.setPreferredSize(new Dimension(PREVIEW_SIZE, PREVIEW_SIZE));
        preview.getAccessibleContext().setAccessibleName("Image preview");
        preview.getAccessibleContext().setAccessibleDescription("Selected image or saliency overlay.");
        results.setEditable(false);
        results.getAccessibleContext().setAccessibleName("Classification results");
        results.getAccessibleContext().setAccessibleDescription("Top-k labels and confidences.");
        status.getAccessibleContext().setAccessibleName("Status");
        JButton open = button("Open image", "Open a single image for classification.", KeyEvent.VK_O,
                this::openImage, "alt O", "open-image");
        JButton folder = button("Batch folder", "Classify every supported image in a folder.", KeyEvent.VK_B,
                this::openFolder, "alt B", "batch-folder");
        JButton saliency = button("Saliency", "Toggle the gradient saliency overlay.", KeyEvent.VK_H,
                this::toggleSaliency, "alt H", "saliency");
        JButton save = button("Export JSON", "Write the current classification report.", KeyEvent.VK_S,
                this::saveReport, "alt S", "save-report");
        JPanel toolbar = new JPanel(new FlowLayout(FlowLayout.LEFT));
        toolbar.add(open);
        toolbar.add(folder);
        toolbar.add(saliency);
        toolbar.add(save);
        frame.add(toolbar, BorderLayout.NORTH);
        frame.add(new JScrollPane(preview), BorderLayout.CENTER);
        frame.add(new JScrollPane(results), BorderLayout.EAST);
        results.setPreferredSize(new Dimension(280, 320));
        frame.add(status, BorderLayout.SOUTH);
        frame.pack();
        frame.setLocationRelativeTo(null);
    }

    void show() {
        frame.setVisible(true);
    }

    private JButton button(String text, String description, int mnemonic, Runnable action,
            String key, String actionName) {
        JButton button = new JButton(text);
        button.getAccessibleContext().setAccessibleName(text);
        button.getAccessibleContext().setAccessibleDescription(description);
        button.setMnemonic(mnemonic);
        button.setToolTipText(description);
        button.addActionListener(event -> action.run());
        frame.getRootPane().getInputMap(JPanel.WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke(key), actionName);
        frame.getRootPane().getActionMap().put(actionName, new AbstractAction() {
            private static final long serialVersionUID = 1L;

            @Override
            public void actionPerformed(ActionEvent event) {
                if (button.isEnabled()) {
                    button.doClick();
                }
            }
        });
        return button;
    }

    private void openImage() {
        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(frame) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        load(chooser.getSelectedFile().toPath());
    }

    /** Launches decode, conversion and inference off the event-dispatch thread. */
    private void load(Path path) {
        if (running) {
            status.setText("Busy — wait for the current task to finish");
            return;
        }
        running = true;
        status.setText("Loading " + path.getFileName() + " …");
        SwingWorker<LoadResult, Void> worker = new SwingWorker<>() {
            @Override
            protected LoadResult doInBackground() {
                BufferedImage image = ImageTensors.readDecodedImage(path);
                Tensor tensor = ImageTensors.fromImage(image, FixtureClassifier.IMAGE_SIZE);
                List<Classification> topK = classifier.classify(tensor);
                return new LoadResult(path, image, tensor, topK);
            }

            @Override
            protected void done() {
                running = false;
                try {
                    LoadResult result = get();
                    currentPath = result.path();
                    currentImage = result.image();
                    currentTensor = result.tensor();
                    currentTopK = result.topK();
                    showingSaliency = false;
                    preview.setIcon(new ImageIcon(result.image().getScaledInstance(PREVIEW_SIZE, PREVIEW_SIZE,
                            java.awt.Image.SCALE_SMOOTH)));
                    StringBuilder text = new StringBuilder("Top predictions\n\n");
                    for (Classification entry : result.topK()) {
                        text.append(String.format(Locale.ROOT, "%-12s %.4f%n", entry.label(),
                                entry.confidence()));
                    }
                    results.setText(text.toString());
                    status.setText(result.path().getFileName() + " — " + result.topK().get(0).label());
                } catch (ExecutionException error) {
                    showLoadError(error.getCause());
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    showLoadError(error);
                }
            }
        };
        worker.execute();
    }

    private void toggleSaliency() {
        if (currentTensor == null) {
            status.setText("Open an image first");
            return;
        }
        if (showingSaliency) {
            showingSaliency = false;
            preview.setIcon(new ImageIcon(currentImage.getScaledInstance(PREVIEW_SIZE, PREVIEW_SIZE,
                    java.awt.Image.SCALE_SMOOTH)));
            status.setText(currentPath.getFileName() + " — image");
            return;
        }
        if (running) {
            status.setText("Busy — wait for the current task to finish");
            return;
        }
        running = true;
        Tensor tensor = currentTensor;
        BufferedImage image = currentImage;
        Path path = currentPath;
        int classIndex = currentTopK.get(0).index();
        status.setText("Computing saliency …");
        SwingWorker<BufferedImage, Void> worker = new SwingWorker<>() {
            @Override
            protected BufferedImage doInBackground() {
                Tensor heatmap = classifier.saliency(tensor, classIndex);
                return VisionOverlay.blend(image, heatmap);
            }

            @Override
            protected void done() {
                running = false;
                try {
                    BufferedImage overlay = get();
                    showingSaliency = true;
                    preview.setIcon(new ImageIcon(overlay.getScaledInstance(PREVIEW_SIZE, PREVIEW_SIZE,
                            java.awt.Image.SCALE_SMOOTH)));
                    status.setText(path.getFileName() + " — saliency");
                } catch (ExecutionException error) {
                    showLoadError(error.getCause());
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    showLoadError(error);
                }
            }
        };
        worker.execute();
    }

    private void openFolder() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        if (chooser.showOpenDialog(frame) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        if (running) {
            status.setText("Busy — wait for the current task to finish");
            return;
        }
        Path folder = chooser.getSelectedFile().toPath();
        running = true;
        status.setText("Classifying folder …");
        SwingWorker<BatchResult, Void> worker = new SwingWorker<>() {
            @Override
            protected BatchResult doInBackground() throws IOException {
                List<Map<String, Object>> rows = new ArrayList<>();
                List<Map<String, Object>> errors = new ArrayList<>();
                try (var files = Files.list(folder)) {
                    for (Path entry : files.filter(VisionInspectorWindow::isImage).sorted().toList()) {
                        try {
                            Tensor input = ImageTensors.fromPath(entry, FixtureClassifier.IMAGE_SIZE);
                            List<Classification> top = classifier.classify(input);
                            rows.add(Map.of("path", entry.toString(), "top1", top.get(0).label(),
                                    "confidence", top.get(0).confidence()));
                        } catch (RuntimeException failure) {
                            errors.add(Map.of("path", entry.toString(),
                                    "error", String.valueOf(failure.getMessage())));
                        }
                    }
                }
                return new BatchResult(rows, errors);
            }

            @Override
            protected void done() {
                running = false;
                try {
                    BatchResult result = get();
                    Map<String, Object> report = new LinkedHashMap<>();
                    report.put("folder", folder.toString());
                    report.put("images", result.rows());
                    report.put("errors", result.errors());
                    results.setText("Batch results\n\n" + result.rows().size() + " images, "
                            + result.errors().size() + " errors\n\n" + Json.writePretty(report));
                    status.setText("Batch complete — " + result.rows().size() + " images, "
                            + result.errors().size() + " errors");
                } catch (ExecutionException error) {
                    status.setText("Folder failed: " + failureMessage(error.getCause()));
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    status.setText("Folder failed: " + failureMessage(error));
                }
            }
        };
        worker.execute();
    }

    private void saveReport() {
        if (currentPath == null) {
            status.setText("Classify an image first");
            return;
        }
        JFileChooser chooser = new JFileChooser();
        if (chooser.showSaveDialog(frame) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("image", currentPath.toString());
        report.put("topK", currentTopK.stream()
                .map(result -> Map.of("label", result.label(), "confidence", result.confidence()))
                .toList());
        try {
            Files.writeString(chooser.getSelectedFile().toPath(), Json.writePretty(report) + "\n");
            Path csv = chooser.getSelectedFile().toPath().resolveSibling(
                    chooser.getSelectedFile().getName() + ".csv");
            StringBuilder rows = new StringBuilder("label,confidence\n");
            for (Classification result : currentTopK) {
                rows.append(result.label()).append(',').append(result.confidence()).append('\n');
            }
            Files.writeString(csv, rows.toString());
            status.setText("Report written to " + chooser.getSelectedFile() + " and " + csv);
        } catch (IOException error) {
            status.setText("Write failed: " + error.getMessage());
        }
    }

    private void showLoadError(Throwable failure) {
        String message = failureMessage(failure);
        status.setText("Failed: " + message);
        JOptionPane.showMessageDialog(frame, message, "Unable to load image", JOptionPane.ERROR_MESSAGE);
    }

    private static String failureMessage(Throwable failure) {
        if (failure == null) {
            return "Unknown error";
        }
        return failure.getMessage() != null ? failure.getMessage() : failure.toString();
    }

    private static boolean isImage(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")
                || name.endsWith(".bmp") || name.endsWith(".gif");
    }

    private record LoadResult(Path path, BufferedImage image, Tensor tensor, List<Classification> topK) {}

    private record BatchResult(List<Map<String, Object>> rows, List<Map<String, Object>> errors) {}
}
