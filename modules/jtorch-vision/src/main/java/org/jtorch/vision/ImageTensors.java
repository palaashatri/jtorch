package org.jtorch.vision;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;

/** Decodes images with ImageIO and converts them to normalized CHW float tensors. */
public final class ImageTensors {
    private ImageTensors() {}

    public static Tensor fromPath(Path path, int size) {
        BufferedImage image;
        try {
            image = ImageIO.read(path.toFile());
        } catch (IOException error) {
            throw new IllegalArgumentException("Unable to read image " + path, error);
        }
        if (image == null) {
            throw new IllegalArgumentException("Unsupported or corrupt image " + path);
        }
        return fromImage(image, size);
    }

    public static Tensor fromImage(BufferedImage image, int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Image size must be positive");
        }
        Tensor chw = Torch.zeros(Shape.of(1, 3, image.getHeight(), image.getWidth()),
                TensorOptions.defaults().dtype(DType.FLOAT32));
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                chw.set(((rgb >> 16) & 0xFF) / 255.0, 0, 0, y, x);
                chw.set(((rgb >> 8) & 0xFF) / 255.0, 0, 1, y, x);
                chw.set((rgb & 0xFF) / 255.0, 0, 2, y, x);
            }
        }
        if (image.getHeight() != size || image.getWidth() != size) {
            chw = chw.interpolate(new int[] {size, size}, "bilinear", false);
        }
        return chw.sub(0.5).div(0.5);
    }

    /** Decodes an image and returns the un-normalized preview in [0, 1] RGB form. */
    public static Tensor preview(Path path, int size) {
        Tensor normalized = fromPath(path, size);
        return normalized.mul(0.5).add(0.5).clamp(0.0, 1.0);
    }

    public static BufferedImage toBufferedImage(Tensor chw) {
        if (chw.dim() != 4 || chw.size(0) != 1 || chw.size(1) != 3) {
            throw new IllegalArgumentException("Expected a [1, 3, H, W] tensor");
        }
        int height = (int) chw.size(2);
        int width = (int) chw.size(3);
        Tensor rgb = chw.mul(0.5).add(0.5).clamp(0.0, 1.0).to(DType.FLOAT64);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (int) Math.round(rgb.get(0, 0, y, x) * 255.0);
                int g = (int) Math.round(rgb.get(0, 1, y, x) * 255.0);
                int b = (int) Math.round(rgb.get(0, 2, y, x) * 255.0);
                image.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return image;
    }
}
