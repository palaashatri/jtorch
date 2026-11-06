package org.jtorch.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.junit.jupiter.api.Test;

class AttentionModuleTest {
    @Test
    void multiheadAttentionUsesPyTorchStateDictNamesAndBackpropagates() {
        MultiheadAttention attention = new MultiheadAttention(8, 2);
        assertEquals(List.of("in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"),
                List.copyOf(attention.namedParameters().keySet()));
        Tensor input = Torch.randn(Shape.of(2, 5, 8)).requiresGrad(true);
        Tensor output = attention.call(input);
        assertEquals(Shape.of(2, 5, 8), output.shape());
        output.sum().backward();
        assertTrue(input.grad() != null);
        assertTrue(attention.namedParameters().get("in_proj_weight").grad() != null);
    }

    @Test
    void transformerEncoderLayerMatchesPostNormShape() {
        TransformerEncoderLayer layer = new TransformerEncoderLayer(8, 2, 16, "gelu");
        assertTrue(layer.stateDict().containsKey("self_attn.in_proj_weight"));
        assertTrue(layer.stateDict().containsKey("norm1.weight"));
        assertTrue(layer.stateDict().containsKey("linear1.weight"));
        Tensor input = Torch.randn(Shape.of(2, 5, 8)).requiresGrad(true);
        Tensor output = layer.call(input);
        assertEquals(Shape.of(2, 5, 8), output.shape());
        output.sum().backward();
        assertTrue(layer.namedParameters().get("linear1.weight").grad() != null);
    }
}
