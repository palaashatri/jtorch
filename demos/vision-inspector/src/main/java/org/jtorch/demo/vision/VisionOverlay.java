package org.jtorch.demo.vision;

import java.awt.image.BufferedImage;
import org.jtorch.core.Tensor;

/** Blends a normalized saliency heatmap over an RGB image. */
public final class VisionOverlay {
    private VisionOverlay() {}

    public static BufferedImage blend(BufferedImage base, Tensor heatmap) {
        if (heatmap.dim() != 3 || heatmap.size(0) != 1) {
            throw new IllegalArgumentException("Expected a [1, H, W] heatmap");
        }
        int height = base.getHeight();
        int width = base.getWidth();
        int mapHeight = (int) heatmap.size(1);
        int mapWidth = (int) heatmap.size(2);
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            int mapY = Math.min(mapHeight - 1, y * mapHeight / height);
            for (int x = 0; x < width; x++) {
                int mapX = Math.min(mapWidth - 1, x * mapWidth / width);
                double weight = Math.max(0.0, Math.min(1.0, heatmap.get(0, mapY, mapX))) * 0.6;
                int rgb = base.getRGB(x, y);
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;
                int blendedRed = (int) Math.round(red * (1 - weight) + 255 * weight);
                int blendedGreen = (int) Math.round(green * (1 - weight));
                int blendedBlue = (int) Math.round(blue * (1 - weight));
                output.setRGB(x, y, (blendedRed << 16) | (blendedGreen << 8) | blendedBlue);
            }
        }
        return output;
    }
}
