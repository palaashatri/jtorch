package org.jtorch.vision;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;

/** Decodes images with ImageIO and converts them to normalized CHW float tensors. */
public final class ImageTensors {
    /** Largest accepted image side, in pixels. */
    public static final int MAX_IMAGE_DIMENSION = 4096;

    /** Largest accepted decoded image, in pixels. */
    public static final long MAX_IMAGE_PIXELS = 16_777_216L;

    private ImageTensors() {}

    /** Rejects decoded images whose dimensions exceed {@link #MAX_IMAGE_DIMENSION} or pixel budget. */
    public static void requireDecodedSize(int width, int height) {
        if (width <= 0 || height <= 0 || width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION
                || (long) width * height > MAX_IMAGE_PIXELS) {
            throw new IllegalArgumentException(
                    "Image " + width + "x" + height + " exceeds the decoded size limit (max side "
                            + MAX_IMAGE_DIMENSION + ", max " + MAX_IMAGE_PIXELS + " pixels)");
        }
    }

    /** Decodes an image after enforcing the decoded-size limits, without converting to a tensor. */
    public static BufferedImage readDecodedImage(Path path) {
        requireHeaderSizeWithinLimits(path);
        BufferedImage image;
        try {
            image = ImageIO.read(path.toFile());
        } catch (IOException error) {
            throw new IllegalArgumentException("Unable to read image " + path, error);
        }
        if (image == null) {
            throw new IllegalArgumentException("Unsupported or corrupt image " + path);
        }
        requireDecodedSize(image.getWidth(), image.getHeight());
        return image;
    }

    public static Tensor fromPath(Path path, int size) {
        return fromImage(readDecodedImage(path), size);
    }

    public static Tensor fromImage(BufferedImage image, int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Image size must be positive");
        }
        requireDecodedSize(image.getWidth(), image.getHeight());
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
        long heightLong = chw.size(2);
        long widthLong = chw.size(3);
        if (heightLong <= 0 || widthLong <= 0 || heightLong > MAX_IMAGE_DIMENSION
                || widthLong > MAX_IMAGE_DIMENSION || heightLong * widthLong > MAX_IMAGE_PIXELS) {
            throw new IllegalArgumentException("Image exceeds the decoded size limit (max side "
                    + MAX_IMAGE_DIMENSION + ", max " + MAX_IMAGE_PIXELS + " pixels)");
        }
        int height = (int) heightLong;
        int width = (int) widthLong;
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

    private static void requireHeaderSizeWithinLimits(Path path) {
        try (ImageInputStream input = ImageIO.createImageInputStream(path.toFile())) {
            if (input == null) {
                return;
            }
            Iterator<ImageReader> readers = ImageIO.getImageReaders(input);
            if (!readers.hasNext()) {
                return;
            }
            ImageReader reader = readers.next();
            try {
                reader.setInput(input, true, false);
                if (reader.getNumImages(false) < 1) {
                    return;
                }
                requireDecodedSize(reader.getWidth(0), reader.getHeight(0));
            } finally {
                reader.dispose();
            }
        } catch (IllegalArgumentException error) {
            throw error;
        } catch (IOException ignored) {
            // Header was unreadable; full decode below reports the failure.
        }
    }
}
