package org.jtorch.demo.tensorlab;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.util.List;
import javax.swing.JPanel;

/** Minimal loss curve renderer using only JDK graphics. */
final class LossChart extends JPanel {
    private static final long serialVersionUID = 1L;
    private transient List<Double> points = List.of();

    LossChart() {
        setBackground(new Color(250, 250, 252));
        setOpaque(true);
    }

    void setPoints(List<Double> values) {
        this.points = List.copyOf(values);
        repaint();
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);
        Graphics2D g = (Graphics2D) graphics.create();
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            int width = getWidth();
            int height = getHeight();
            int left = 56;
            int right = 24;
            int top = 24;
            int bottom = 40;
            int plotWidth = Math.max(1, width - left - right);
            int plotHeight = Math.max(1, height - top - bottom);
            g.setColor(new Color(210, 214, 222));
            g.drawRect(left, top, plotWidth, plotHeight);
            g.setColor(new Color(90, 96, 108));
            g.drawString("training loss", left, top - 8);
            g.drawString("epoch", left + plotWidth - 36, top + plotHeight + 24);
            if (points.size() < 2) {
                g.drawString(points.isEmpty() ? "no epochs yet" : "one epoch", left + 12, top + 24);
                return;
            }
            double max = Double.NEGATIVE_INFINITY;
            double min = Double.POSITIVE_INFINITY;
            for (double value : points) {
                max = Math.max(max, value);
                min = Math.min(min, value);
            }
            if (max == min) {
                max = min + 1;
            }
            g.setColor(new Color(30, 90, 170));
            g.setStroke(new BasicStroke(2f));
            int previousX = 0;
            int previousY = 0;
            for (int i = 0; i < points.size(); i++) {
                double x = left + (double) i / (points.size() - 1) * plotWidth;
                double y = top + (1 - (points.get(i) - min) / (max - min)) * plotHeight;
                if (i > 0) {
                    g.drawLine(previousX, previousY, (int) Math.round(x), (int) Math.round(y));
                }
                previousX = (int) Math.round(x);
                previousY = (int) Math.round(y);
            }
            g.setColor(new Color(90, 96, 108));
            g.drawString(String.format("max %.4f", max), 6, top + 12);
            g.drawString(String.format("min %.4f", min), 6, top + plotHeight);
            g.setColor(new Color(30, 90, 170));
            g.fillOval(previousX - 3, previousY - 3, 6, 6);
        } finally {
            g.dispose();
        }
    }
}
