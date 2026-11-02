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
import javax.imageio.ImageIO;
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

    private void load(Path path) {
        try {
            BufferedImage image = ImageIO.read(path.toFile());
            if (image == null) {
                throw new IllegalArgumentException("Unsupported or corrupt image " + path);
            }
            currentPath = path;
            currentImage = image;
            currentTensor = ImageTensors.fromImage(image, FixtureClassifier.IMAGE_SIZE);
            currentTopK = classifier.classify(currentTensor);
            showingSaliency = false;
            preview.setIcon(new ImageIcon(image.getScaledInstance(PREVIEW_SIZE, PREVIEW_SIZE,
                    java.awt.Image.SCALE_SMOOTH)));
            StringBuilder text = new StringBuilder("Top predictions\n\n");
            for (Classification result : currentTopK) {
                text.append(String.format(Locale.ROOT, "%-12s %.4f%n", result.label(), result.confidence()));
            }
            results.setText(text.toString());
            status.setText(path.getFileName() + " — " + currentTopK.get(0).label());
        } catch (IOException | RuntimeException error) {
            status.setText("Failed: " + error.getMessage());
            JOptionPane.showMessageDialog(frame, error.getMessage(), "Unable to load image",
                    JOptionPane.ERROR_MESSAGE);
        }
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
        Tensor heatmap = classifier.saliency(currentTensor, currentTopK.get(0).index());
        BufferedImage overlay = VisionOverlay.blend(currentImage, heatmap);
        showingSaliency = true;
        preview.setIcon(new ImageIcon(overlay.getScaledInstance(PREVIEW_SIZE, PREVIEW_SIZE,
                java.awt.Image.SCALE_SMOOTH)));
        status.setText(currentPath.getFileName() + " — saliency");
    }

    private void openFolder() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        if (chooser.showOpenDialog(frame) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        Path folder = chooser.getSelectedFile().toPath();
        List<Map<String, Object>> rows = new ArrayList<>();
        List<Map<String, Object>> errors = new ArrayList<>();
        try (var files = Files.list(folder)) {
            for (Path path : files.filter(VisionInspectorWindow::isImage).sorted().toList()) {
                try {
                    Tensor input = ImageTensors.fromPath(path, FixtureClassifier.IMAGE_SIZE);
                    List<Classification> top = classifier.classify(input);
                    rows.add(Map.of("path", path.toString(), "top1", top.get(0).label(),
                            "confidence", top.get(0).confidence()));
                } catch (RuntimeException error) {
                    errors.add(Map.of("path", path.toString(), "error", error.getMessage()));
                }
            }
        } catch (IOException error) {
            status.setText("Folder failed: " + error.getMessage());
            return;
        }
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("folder", folder.toString());
        report.put("images", rows);
        report.put("errors", errors);
        results.setText("Batch results\n\n" + rows.size() + " images, " + errors.size() + " errors\n\n"
                + Json.writePretty(report));
        status.setText("Batch complete — " + rows.size() + " images, " + errors.size() + " errors");
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

    private static boolean isImage(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")
                || name.endsWith(".bmp") || name.endsWith(".gif");
    }
}
