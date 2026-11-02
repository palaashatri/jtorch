package org.jtorch.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.image.BufferedImage;
import java.util.List;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.junit.jupiter.api.Test;

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
}
