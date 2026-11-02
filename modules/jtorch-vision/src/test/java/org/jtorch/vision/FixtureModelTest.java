package org.jtorch.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import org.jtorch.core.Tensor;
import org.jtorch.nn.StateDict;
import org.jtorch.serialization.Safetensors;
import org.jtorch.vision.models.ResNet9;
import org.jtorch.vision.models.ViT;
import org.junit.jupiter.api.Test;

class FixtureModelTest {
    private static Path resource(String name) throws URISyntaxException {
        return Path.of(Objects.requireNonNull(FixtureModelTest.class.getResource("/fixtures/" + name)).toURI());
    }

    private static void assertClose(String label, Tensor expected, Tensor actual) {
        assertEquals(expected.shape(), actual.shape(), label + " shape");
        float[] expectedValues = expected.toFloatArray();
        float[] actualValues = actual.toFloatArray();
        double maximum = 0;
        for (int i = 0; i < expectedValues.length; i++) {
            maximum = Math.max(maximum, Math.abs(expectedValues[i] - actualValues[i]));
        }
        assertTrue(maximum < 5e-4, label + " maximum absolute error " + maximum);
    }

    @Test
    void resNetFixtureMatchesPyTorchLogits() throws URISyntaxException {
        ResNet9 model = new ResNet9(8, 4);
        model.eval();
        model.loadStateDict(new StateDict(Safetensors.load(resource("resnet9-weights.safetensors"))), true);
        Map<String, Tensor> io = Safetensors.load(resource("resnet9-io.safetensors"));
        assertClose("resnet9", io.get("logits"), model.call(io.get("input")));
    }

    @Test
    void vitFixtureMatchesPyTorchLogits() throws URISyntaxException {
        ViT model = new ViT(16, 4, 16, 2, 1, 32, 10);
        model.eval();
        model.loadStateDict(new StateDict(Safetensors.load(resource("vit-weights.safetensors"))), true);
        Map<String, Tensor> io = Safetensors.load(resource("vit-io.safetensors"));
        assertClose("vit", io.get("logits"), model.call(io.get("input")));
    }
}
