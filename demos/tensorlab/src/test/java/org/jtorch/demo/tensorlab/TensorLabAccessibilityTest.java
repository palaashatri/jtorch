package org.jtorch.demo.tensorlab;

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
        assertNotNull(train.getAccessibleContext().getAccessibleName());
        assertNotNull(train.getAccessibleContext().getAccessibleDescription());
        assertNotNull(save.getAccessibleContext().getAccessibleName());
        assertNotNull(status.getAccessibleContext().getAccessibleName());
        assertTrue(train.getMnemonic() == KeyEvent.VK_T);
        assertTrue(save.getMnemonic() == KeyEvent.VK_S);
        assertNotNull(root.getInputMap(JPanel.WHEN_IN_FOCUSED_WINDOW)
                .get(javax.swing.KeyStroke.getKeyStroke("alt T")));
        assertNotNull(root.getActionMap().get("train-one-epoch"));
    }
}
