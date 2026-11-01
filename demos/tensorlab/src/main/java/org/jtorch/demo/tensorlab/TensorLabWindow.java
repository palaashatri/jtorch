package org.jtorch.demo.tensorlab;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import org.jtorch.serialization.Checkpoint;

/** TensorLab's Swing window. All numbers come from {@link TensorLabTraining}. */
final class TensorLabWindow {
    private final TensorLabTraining training;
    private final LossChart chart = new LossChart();
    private final JLabel status = new JLabel("ready");
    private final JLabel accuracy = new JLabel("accuracy: -");
    private final JButton trainButton = new JButton("Train 1 epoch");
    private final JButton saveButton = new JButton("Save checkpoint");
    private final java.nio.file.Path checkpointPath;

    TensorLabWindow(TensorLabTraining training, java.nio.file.Path checkpointPath) {
        this.training = training;
        this.checkpointPath = checkpointPath;
    }

    void show() {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("TensorLab — JTorch training workbench");
            frame.getAccessibleContext().setAccessibleName("TensorLab training workbench");
            frame.getAccessibleContext().setAccessibleDescription(
                    "Trains a small CNN or MLP on a ten-class digit dataset and shows live loss and accuracy.");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setLayout(new BorderLayout(12, 12));
            JPanel header = new JPanel(new GridLayout(2, 1));
            JLabel title = new JLabel("Ten-class synthetic digit classifier");
            title.setFont(title.getFont().deriveFont(Font.BOLD, 18f));
            header.add(title);
            header.add(new JLabel("Linear(64→64) · ReLU · Linear(64→10) · cross entropy · SGD"));
            header.setBorder(BorderFactory.createEmptyBorder(12, 12, 0, 12));
            frame.add(header, BorderLayout.NORTH);

            chart.setPreferredSize(new Dimension(820, 420));
            TensorLabAccessibility.configureChart(chart, "Training loss chart",
                    "Line chart of mean training loss for each completed epoch.");
            TensorLabAccessibility.configureButton(trainButton, "Train one epoch",
                    "Run one training epoch and evaluate the test split.", TensorLabAccessibility.trainMnemonic());
            TensorLabAccessibility.configureButton(saveButton, "Save checkpoint",
                    "Write the current model state dictionary to a checkpoint file.",
                    TensorLabAccessibility.saveMnemonic());
            TensorLabAccessibility.configureLabel(accuracy, "Test accuracy", "Accuracy on the held-out test split.");
            TensorLabAccessibility.configureLabel(status, "Training status", "Current epoch, loss and save state.");
            TensorLabAccessibility.installShortcuts(frame.getRootPane(), trainButton, saveButton);
            frame.getRootPane().setDefaultButton(trainButton);
            chart.setBorder(BorderFactory.createEmptyBorder(0, 12, 0, 12));
            frame.add(chart, BorderLayout.CENTER);

            JPanel footer = new JPanel(new BorderLayout());
            JPanel buttons = new JPanel();
            buttons.add(trainButton);
            buttons.add(saveButton);
            footer.add(buttons, BorderLayout.WEST);
            JPanel metrics = new JPanel(new GridLayout(2, 1));
            metrics.add(accuracy);
            metrics.add(status);
            footer.add(metrics, BorderLayout.EAST);
            footer.setBorder(BorderFactory.createEmptyBorder(0, 12, 12, 12));
            frame.add(footer, BorderLayout.SOUTH);

            trainButton.addActionListener(event -> trainOneEpoch());
            saveButton.addActionListener(event -> save());
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }

    private void trainOneEpoch() {
        trainButton.setEnabled(false);
        status.setText("training epoch " + (training.epoch() + 1) + "…");
        new SwingWorker<double[], Void>() {
            @Override
            protected double[] doInBackground() {
                double loss = training.trainEpoch();
                double testAccuracy = training.evaluate();
                return new double[] {loss, testAccuracy};
            }

            @Override
            protected void done() {
                try {
                    double[] metrics = get();
                    List<Double> history = training.lossHistory();
                    chart.setPoints(history);
                    accuracy.setText(String.format("accuracy: %.2f%%", metrics[1] * 100));
                    status.setText(String.format("epoch %d · loss %.5f", training.epoch(), metrics[0]));
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    status.setText("interrupted");
                } catch (ExecutionException error) {
                    status.setText("failed: " + error.getCause().getMessage());
                } finally {
                    trainButton.setEnabled(true);
                }
            }
        }.execute();
    }

    private void save() {
        Map<String, org.jtorch.core.Tensor> tensors = training.stateDict();
        Checkpoint.save(checkpointPath, tensors,
                Map.of("app", "tensorlab", "epoch", Integer.toString(training.epoch())));
        status.setText("saved " + checkpointPath.getFileName());
    }
}
