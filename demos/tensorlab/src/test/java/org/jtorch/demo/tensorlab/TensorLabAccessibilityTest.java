package org.jtorch.demo.tensorlab;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.event.KeyEvent;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import org.junit.jupiter.api.Test;

class TensorLabAccessibilityTest {
    @Test
    void controlsExposeAccessibleNamesDescriptionsAndMnemonics() {
        JButton train = new JButton("Train 1 epoch");
        JButton save = new JButton("Save checkpoint");
        JLabel status = new JLabel("ready");
        JPanel root = new JPanel();
        TensorLabAccessibility.configureButton(train, "Train one epoch", "Run one epoch",
                TensorLabAccessibility.trainMnemonic());
        TensorLabAccessibility.configureButton(save, "Save checkpoint", "Save model",
                TensorLabAccessibility.saveMnemonic());
        TensorLabAccessibility.configureLabel(status, "Training status", "Current status");
        TensorLabAccessibility.installShortcuts(root, train, save);
        assertEquals("Train one epoch", train.getAccessibleContext().getAccessibleName());
        assertEquals("Run one epoch", train.getAccessibleContext().getAccessibleDescription());
        assertEquals("Save checkpoint", save.getAccessibleContext().getAccessibleName());
        assertEquals("Training status", status.getAccessibleContext().getAccessibleName());
        assertTrue(train.getMnemonic() == KeyEvent.VK_T);
        assertTrue(save.getMnemonic() == KeyEvent.VK_S);
        assertNotNull(root.getInputMap(JPanel.WHEN_IN_FOCUSED_WINDOW)
                .get(javax.swing.KeyStroke.getKeyStroke("alt T")));
        java.util.concurrent.atomic.AtomicInteger clicks = new java.util.concurrent.atomic.AtomicInteger();
        train.addActionListener(event -> clicks.incrementAndGet());
        root.getActionMap().get("train-one-epoch")
                .actionPerformed(new java.awt.event.ActionEvent(root, 0, "test"));
        assertEquals(1, clicks.get());
        train.setEnabled(false);
        root.getActionMap().get("train-one-epoch")
                .actionPerformed(new java.awt.event.ActionEvent(root, 0, "test"));
        assertEquals(1, clicks.get());
    }
}
