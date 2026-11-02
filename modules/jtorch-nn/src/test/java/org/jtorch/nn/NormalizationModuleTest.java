package org.jtorch.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.junit.jupiter.api.Test;

class NormalizationModuleTest {
    @Test
    void layerNormRegistersAffineParametersAndRuns() {
        LayerNorm norm = new LayerNorm(4);
        assertEquals(List.of("weight", "bias"), List.copyOf(norm.namedParameters().keySet()));
        assertEquals(Shape.of(4), norm.weight().shape());
        Tensor input = Torch.ones(Shape.of(2, 3, 4)).requiresGrad(true);
        Tensor output = norm.call(input);
        assertEquals(Shape.of(2, 3, 4), output.shape());
        output.sum().backward();
        assertEquals(Shape.of(4), norm.weight().grad().shape());
        assertEquals(Shape.of(2, 3, 4), input.grad().shape());
    }

    @Test
    void groupNormValidatesGroupsAndRuns() {
        GroupNorm norm = new GroupNorm(2, 4);
        Tensor input = Torch.ones(Shape.of(1, 4, 2, 2)).requiresGrad(true);
        Tensor output = norm.call(input);
        assertEquals(Shape.of(1, 4, 2, 2), output.shape());
        output.sum().backward();
        assertEquals(Shape.of(4), norm.weight().grad().shape());
        assertThrows(IllegalArgumentException.class, () -> new GroupNorm(3, 4));
    }
}
