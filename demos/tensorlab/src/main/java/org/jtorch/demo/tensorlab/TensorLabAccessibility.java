package org.jtorch.demo.tensorlab;

import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.KeyStroke;

/** Shared accessibility metadata and keyboard shortcuts for TensorLab controls. */
final class TensorLabAccessibility {
    private TensorLabAccessibility() {}

    static void configureButton(JButton button, String name, String description, int mnemonic) {
        button.getAccessibleContext().setAccessibleName(name);
        button.getAccessibleContext().setAccessibleDescription(description);
        button.setMnemonic(mnemonic);
        button.setToolTipText(description);
    }

    static void configureLabel(JLabel label, String name, String description) {
        label.getAccessibleContext().setAccessibleName(name);
        label.getAccessibleContext().setAccessibleDescription(description);
    }

    static void configureChart(JComponent chart, String name, String description) {
        chart.getAccessibleContext().setAccessibleName(name);
        chart.getAccessibleContext().setAccessibleDescription(description);
    }

    static void installShortcuts(JComponent root, JButton trainButton, JButton saveButton) {
        bind(root, "alt T", "train-one-epoch", trainButton);
        bind(root, "alt S", "save-checkpoint", saveButton);
    }

    private static void bind(JComponent root, String key, String actionName, JButton button) {
        root.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW)
                .put(KeyStroke.getKeyStroke(key), actionName);
        root.getActionMap().put(actionName, new AbstractAction() {
            private static final long serialVersionUID = 1L;

            @Override
            public void actionPerformed(ActionEvent event) {
                if (button.isEnabled()) {
                    button.doClick();
                }
            }
        });
    }

    static int trainMnemonic() {
        return KeyEvent.VK_T;
    }

    static int saveMnemonic() {
        return KeyEvent.VK_S;
    }
}
