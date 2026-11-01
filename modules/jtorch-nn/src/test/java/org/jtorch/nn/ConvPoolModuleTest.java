package org.jtorch.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.junit.jupiter.api.Test;

class ConvPoolModuleTest {
    @Test
    void conv2dRegistersParametersAndProducesExpectedShape() {
        Conv2d conv = new Conv2d(2, 4, 3);
        assertEquals(List.of("weight", "bias"), List.copyOf(conv.namedParameters().keySet()));
        assertEquals(Shape.of(4, 2, 3, 3), conv.weight().shape());
        assertEquals(Shape.of(4), conv.bias().shape());
        Tensor input = Torch.ones(Shape.of(2, 2, 8, 8));
        assertEquals(Shape.of(2, 4, 6, 6), conv.call(input).shape());
        assertTrue(conv.weight().requiresGrad());
    }

    @Test
    void conv2dWithStridePaddingAndBiasOption() {
        Conv2d conv = new Conv2d(1, 3, new int[] {3, 3}, new int[] {2, 2}, new int[] {1, 1});
        assertEquals(Shape.of(1, 3, 4, 4), conv.call(Torch.ones(Shape.of(1, 1, 7, 7))).shape());
        Conv2d withoutBias = new Conv2d(1, 2, new int[] {2, 2}, new int[] {1, 1}, new int[] {0, 0},
                new int[] {1, 1}, 1, false);
        assertEquals(List.of("weight"), List.copyOf(withoutBias.namedParameters().keySet()));
    }

    @Test
    void maxPool2dProducesExpectedShape() {
        MaxPool2d pool = new MaxPool2d(2, 2, 0);
        assertEquals(Shape.of(2, 3, 3, 3), pool.call(Torch.ones(Shape.of(2, 3, 6, 6))).shape());
        assertThrows(IllegalArgumentException.class,
                () -> new MaxPool2d(new int[] {2}, new int[] {2, 2}, new int[] {0, 0}, new int[] {1, 1}, false));
    }

    @Test
    void convAndPoolGradientsFlowToParameters() {
        Conv2d conv = new Conv2d(1, 2, 2);
        MaxPool2d pool = new MaxPool2d(2, 2, 0);
        Tensor input = Torch.ones(Shape.of(1, 1, 4, 4)).requiresGrad(true);
        Tensor loss = pool.call(conv.call(input)).sum();
        loss.backward();
        assertTrue(input.grad() != null);
        assertTrue(conv.weight().grad() != null);
        assertTrue(conv.bias().grad() != null);
    }
}
