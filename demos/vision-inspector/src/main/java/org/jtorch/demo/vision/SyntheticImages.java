package org.jtorch.demo.vision;

import java.awt.image.BufferedImage;

/** Deterministic fixture images matching the synthetic classes used to train the fixture model. */
public final class SyntheticImages {
    public static final int SIZE = 32;

    private SyntheticImages() {}

    public static BufferedImage pattern(int label) {
        BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                int rgb;
                if (label == 3) {
                    rgb = ((x / 4 + y / 4) % 2 == 0) ? 0xFFFFFF : 0x000000;
                } else {
                    int[] channels = {25, 25, 25};
                    channels[label] = 255;
                    rgb = (channels[0] << 16) | (channels[1] << 8) | channels[2];
                }
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }
}
