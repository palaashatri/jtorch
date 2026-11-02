package org.jtorch.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class VisionModuleTest {
    @Test
    void adaptivePoolModuleMatchesTheFunctionalOperator() {
        Tensor input = Torch.ones(Shape.of(2, 3, 7, 7)).requiresGrad(true);
        Tensor output = new AdaptiveAvgPool2d(3).call(input);
        assertEquals(Shape.of(2, 3, 3, 3), output.shape());
        output.sum().backward();
        assertEquals(Shape.of(2, 3, 7, 7), input.grad().shape());
    }

    @Test
    void geluModuleSupportsBothModes() {
        Tensor input = Torch.tensor(new double[] {-1, 0, 1}, Shape.of(3));
        assertEquals(-0.15865525393145707, new GELU().call(input).get(0), 1e-12);
        assertEquals(-0.15880800939172324, new GELU("tanh").call(input).get(0), 1e-12);
        assertThrows(IllegalArgumentException.class, () -> new GELU("cubic"));
    }

    @Test
    void embeddingModuleRegistersWeightAndHonoursPadding() {
        Embedding embedding = new Embedding(5, 4, 2);
        assertEquals(List.of("weight"), List.copyOf(embedding.namedParameters().keySet()));
        assertEquals(Shape.of(5, 4), embedding.weight().shape());
        Tensor indices = Torch.tensor(new long[] {0, 2, 4}, Shape.of(3),
                TensorOptions.defaults().dtype(DType.INT64));
        Tensor output = embedding.call(indices);
        assertEquals(Shape.of(3, 4), output.shape());
        assertEquals(0.0, output.select(0, 1).sum().item(), 0);

        Embedding negative = new Embedding(5, 4, -1);
        Tensor negativeIndices = Torch.tensor(new long[] {0, 4}, Shape.of(2),
                TensorOptions.defaults().dtype(DType.INT64));
        negative.call(negativeIndices).sum().backward();
        assertEquals(4.0, negative.weight().grad().select(0, 0).sum().item(), 0);
        assertEquals(0.0, negative.weight().grad().select(0, 4).sum().item(), 0);
    }
}
