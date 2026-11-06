package org.jtorch.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.nn.Parameter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class VisionPipelineTest {
    private static BufferedImage image(int label) {
        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
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

    @Test
    void fixtureClassifierRecognizesEverySyntheticClass() {
        FixtureClassifier classifier = new FixtureClassifier();
        List<String> expected = List.of("red", "green", "blue", "checker");
        for (int label = 0; label < expected.size(); label++) {
            Tensor input = ImageTensors.fromImage(image(label), FixtureClassifier.IMAGE_SIZE);
            List<Classification> results = classifier.classify(input);
            assertEquals(expected.get(label), results.get(0).label(), "label " + label);
            assertTrue(results.get(0).confidence() > 0.9, "confidence " + results.get(0).confidence());
        }
    }

    @Test
    void saliencyProducesFiniteNormalizedHeatmap() {
        FixtureClassifier classifier = new FixtureClassifier();
        Tensor input = ImageTensors.fromImage(image(0), FixtureClassifier.IMAGE_SIZE);
        Tensor heatmap = classifier.saliency(input, 0);
        assertEquals(Shape.of(1, 32, 32), heatmap.shape());
        double minimum = Double.POSITIVE_INFINITY;
        double maximum = Double.NEGATIVE_INFINITY;
        for (float value : heatmap.toFloatArray()) {
            assertTrue(Float.isFinite(value), "saliency value must be finite");
            minimum = Math.min(minimum, value);
            maximum = Math.max(maximum, value);
        }
        assertTrue(minimum >= 0.0 && maximum <= 1.0);
    }

    @Test
    void classifyDoesNotBuildAGraphOrTouchParameterGradients() {
        FixtureClassifier classifier = new FixtureClassifier();
        List<Parameter> parameters = classifier.model().parameters();
        Parameter first = parameters.get(0);
        Tensor sentinel = Torch.ones(first.shape());
        first.impl().setGrad(sentinel);
        List<Tensor> previousGrads = new ArrayList<>();
        for (Parameter parameter : parameters) {
            previousGrads.add(parameter.grad());
        }
        Tensor input = ImageTensors.fromImage(image(0), FixtureClassifier.IMAGE_SIZE);
        List<Classification> results = classifier.classify(input);
        assertEquals("red", results.get(0).label());
        for (int i = 0; i < parameters.size(); i++) {
            assertSame(previousGrads.get(i), parameters.get(i).grad(), "parameter " + i + " grad must be unchanged");
        }
        assertSame(sentinel, first.grad());
    }

    @Test
    void saliencyLeavesEveryParameterGradientUnchanged() {
        FixtureClassifier classifier = new FixtureClassifier();
        List<Parameter> parameters = classifier.model().parameters();
        assertEquals(true, !parameters.isEmpty());
        List<Tensor> previousGrads = new ArrayList<>();
        List<double[]> previousFirstValues = new ArrayList<>();
        for (Parameter parameter : parameters) {
            Tensor sentinel = Torch.ones(parameter.shape()).mul(3.0);
            parameter.impl().setGrad(sentinel);
            previousGrads.add(parameter.grad());
            previousFirstValues.add(parameter.grad().toDoubleArray().clone());
        }
        Tensor input = ImageTensors.fromImage(image(0), FixtureClassifier.IMAGE_SIZE);
        Tensor gradient = Saliency.inputGradient(classifier.model(), input, 0);
        assertNotNull(gradient);
        assertEquals(Shape.of(1, 3, 32, 32), gradient.shape());
        assertTrue(gradient.numel() > 0);
        for (int i = 0; i < parameters.size(); i++) {
            assertSame(previousGrads.get(i), parameters.get(i).grad(), "parameter " + i + " grad reference");
            assertEquals(previousFirstValues.get(i).length,
                    parameters.get(i).grad().toDoubleArray().length);
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    previousFirstValues.get(i), parameters.get(i).grad().toDoubleArray(), 0.0);
        }
        Saliency.inputGradient(classifier.model(), input, 0);
        for (int i = 0; i < parameters.size(); i++) {
            assertSame(previousGrads.get(i), parameters.get(i).grad(), "parameter " + i + " grad after second call");
        }
    }

    @Test
    void saliencyKeepsParameterRequiresGrad() {
        FixtureClassifier classifier = new FixtureClassifier();
        Tensor input = ImageTensors.fromImage(image(0), FixtureClassifier.IMAGE_SIZE);
        Saliency.inputGradient(classifier.model(), input, 0);
        for (Parameter parameter : classifier.model().parameters()) {
            assertTrue(parameter.requiresGrad(), "parameter must still require grad after saliency");
        }
    }

    @Test
    void oversizedImagesAreRejectedBeforeTensorConversion() {
        BufferedImage wide = new BufferedImage(ImageTensors.MAX_IMAGE_DIMENSION + 1, 32,
                BufferedImage.TYPE_INT_RGB);
        assertThrows(IllegalArgumentException.class,
                () -> ImageTensors.fromImage(wide, FixtureClassifier.IMAGE_SIZE));
        assertThrows(IllegalArgumentException.class,
                () -> ImageTensors.requireDecodedSize(ImageTensors.MAX_IMAGE_DIMENSION + 1, 32));
        assertThrows(IllegalArgumentException.class,
                () -> ImageTensors.requireDecodedSize(4096, 4097));
        assertThrows(IllegalArgumentException.class,
                () -> ImageTensors.requireDecodedSize(
                        ImageTensors.MAX_IMAGE_DIMENSION + 1, 1));
        ImageTensors.requireDecodedSize(32, 32);
        ImageTensors.requireDecodedSize(
                ImageTensors.MAX_IMAGE_DIMENSION, ImageTensors.MAX_IMAGE_DIMENSION);
    }

    @Test
    void oversizedFilesAreRejected(@TempDir Path directory) throws IOException {
        BufferedImage wide = new BufferedImage(ImageTensors.MAX_IMAGE_DIMENSION + 1, 32,
                BufferedImage.TYPE_INT_RGB);
        Path file = directory.resolve("wide.png");
        assertTrue(ImageIO.write(wide, "png", file.toFile()));
        assertThrows(IllegalArgumentException.class,
                () -> ImageTensors.fromPath(file, FixtureClassifier.IMAGE_SIZE));
        assertThrows(IllegalArgumentException.class, () -> ImageTensors.readDecodedImage(file));
        assertTrue(Files.size(file) > 0);
    }

    @Test
    void decodedImagesRoundTripWithinLimits() throws IOException {
        Path file = Files.createTempFile("vision-image", ".png");
        try {
            assertTrue(ImageIO.write(image(0), "png", file.toFile()));
            Tensor tensor = ImageTensors.fromPath(file, FixtureClassifier.IMAGE_SIZE);
            assertEquals(Shape.of(1, 3, 32, 32), tensor.shape());
            assertTrue(tensor.numel() == 3072);
            for (float value : tensor.toFloatArray()) {
                assertTrue(Float.isFinite(value));
            }
        } finally {
            Files.deleteIfExists(file);
        }
    }
}
