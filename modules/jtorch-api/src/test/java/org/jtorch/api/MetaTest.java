package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.jtorch.core.DType;
import org.jtorch.core.Device;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

class MetaTest {
    private static TensorOptions metaOptions() {
        return TensorOptions.defaults().device(Device.meta());
    }

    @Test
    void metaFactoriesExposeMetadataOnly() {
        Tensor zeros = Torch.zeros(Shape.of(2, 3), metaOptions());
        assertEquals(Device.meta(), zeros.device());
        assertEquals(DType.FLOAT32, zeros.dtype());
        assertEquals(Shape.of(2, 3), zeros.shape());
        assertTrue(zeros.isContiguous());
        assertThrows(JTorchException.class, zeros::toDoubleArray);

        Tensor full = Torch.full(Shape.of(2), org.jtorch.core.Scalar.of(1.5), metaOptions());
        assertEquals(Shape.of(2), full.shape());
        Tensor arange = Torch.arange(org.jtorch.core.Scalar.of(0), org.jtorch.core.Scalar.of(5),
                org.jtorch.core.Scalar.of(1), metaOptions());
        assertEquals(Shape.of(5), arange.shape());
        assertEquals(Shape.of(3, 3), Torch.eye(3, metaOptions()).shape());
    }

    @Test
    void metaArithmeticAndReductionsAgreeWithCpu() {
        Tensor a = Torch.zeros(Shape.of(2, 3), metaOptions());
        Tensor b = Torch.zeros(Shape.of(3), metaOptions());
        Tensor sum = a.add(b);
        assertEquals(Shape.of(2, 3), sum.shape());
        assertEquals(Shape.of(1, 3), a.sum(0, true).shape());
        assertEquals(Shape.of(2), a.mean(1, false).shape());
        assertEquals(DType.INT64, a.argmax(1, false).dtype());
        assertEquals(Shape.of(2), a.argmax(1, false).shape());
        assertEquals(Shape.of(2, 3), a.softmax(1).shape());
        assertEquals(Shape.of(2, 3), a.logSoftmax(1).shape());

        Tensor weight = Torch.zeros(Shape.of(4, 3), metaOptions());
        assertEquals(Shape.of(2, 4), a.linear(weight, null).shape());
        Tensor logits = Torch.zeros(Shape.of(3, 4), metaOptions());
        Tensor labels = Torch.zeros(Shape.of(3), metaOptions().dtype(DType.INT64));
        assertEquals(Shape.scalar(), logits.crossEntropy(labels, "mean").shape());
        assertEquals(Shape.of(3), logits.crossEntropy(labels, "none").shape());
        assertEquals(Shape.scalar(), logits.mseLoss(logits, "mean").shape());
    }

    @Test
    void metaViewGeometryMatchesCpu() {
        Tensor meta = Torch.zeros(Shape.of(2, 3), metaOptions());
        Tensor cpu = Torch.zeros(Shape.of(2, 3));
        for (Tensor[] pair : new Tensor[][] {
            {meta.transpose(0, 1), cpu.transpose(0, 1)},
            {meta.narrow(1, 1, 2), cpu.narrow(1, 1, 2)},
            {meta.reshape(3, 2), cpu.reshape(3, 2)},
            {meta.unsqueeze(1), cpu.unsqueeze(1)},
        }) {
            assertEquals(pair[1].shape(), pair[0].shape());
            assertEquals(pair[1].strides(), pair[0].strides());
        }
        Tensor expanded = Torch.zeros(Shape.of(2, 1), metaOptions()).expand(2, 4);
        assertEquals(Shape.of(2, 4), expanded.shape());
        assertArrayEquals(new long[] {1, 0}, expanded.strides().toArray());
    }

    @Test
    void metaAutogradProducesMetaGradients() {
        Tensor a = Torch.zeros(Shape.of(2, 3), metaOptions().requiresGrad(true));
        Tensor b = Torch.zeros(Shape.of(3, 1), metaOptions().requiresGrad(true));
        Tensor loss = a.matmul(b).sum();
        assertEquals(Device.meta(), loss.device());
        loss.backward();
        assertEquals(Shape.of(2, 3), a.grad().shape());
        assertEquals(Shape.of(3, 1), b.grad().shape());
        assertEquals(Device.meta(), a.grad().device());
    }

    @Test
    void metaTensorsCannotBeFilledFromArrays() {
        assertThrows(org.jtorch.core.DeviceException.class,
                () -> Torch.tensor(new float[] {1}, Shape.of(1), metaOptions()));
    }

    @Test
    void metaAndCpuTensorsCannotBeMixed() {
        Tensor cpu = Torch.zeros(Shape.of(2));
        Tensor meta = Torch.zeros(Shape.of(2), metaOptions());
        assertThrows(org.jtorch.core.DeviceException.class, () -> cpu.add(meta));
        assertFalse(cpu.device().isMeta());
    }
}
