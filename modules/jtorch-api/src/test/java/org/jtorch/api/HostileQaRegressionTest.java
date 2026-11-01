package org.jtorch.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.jtorch.core.AutogradException;
import org.jtorch.core.DType;
import org.jtorch.core.DTypeException;
import org.jtorch.core.Device;
import org.jtorch.core.JTorchException;
import org.jtorch.core.Scalar;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;
import org.junit.jupiter.api.Test;

/** Regression tests distilled from hostile QA; every expectation was checked against PyTorch 2.11.0. */
class HostileQaRegressionTest {
    private static final double EPS = 1e-9;
    private static final long BIG = (1L << 60) + 1;

    private static Tensor longs(long... values) {
        return Torch.tensor(values, Shape.of(values.length));
    }

    private static Tensor doubles(double... values) {
        return Torch.tensor(values, Shape.of(values.length));
    }

    @Test
    void weakScalarPromotionMatchesPyTorch() {
        Tensor ints = longs(1, 2, 3);
        Tensor multiplied = ints.mul(0.5);
        assertEquals(DType.FLOAT32, multiplied.dtype());
        assertArrayEquals(new double[] {0.5, 1, 1.5}, multiplied.toDoubleArray(), 1e-7);
        assertArrayEquals(new double[] {1.5, 2.5, 3.5}, ints.add(0.5).toDoubleArray(), 1e-7);
        assertEquals(3.6, longs(9).div(2.5).item(), 1e-6);
        Tensor bools = Torch.tensor(new boolean[] {true, false}, Shape.of(2));
        Tensor scaled = bools.mul(2);
        assertEquals(DType.INT64, scaled.dtype());
        assertArrayEquals(new long[] {2, 0}, scaled.toLongArray());
        assertThrows(DTypeException.class, () -> longs(1).add(longs(1), 2.5));
    }

    @Test
    void int64ValuesStayExact() {
        assertArrayEquals(new long[] {BIG, BIG + 2},
                Tensor.cat(List.of(longs(BIG), longs(BIG + 2)), 0).toLongArray());
        assertArrayEquals(new long[] {BIG, BIG + 2},
                Tensor.stack(List.of(longs(BIG), longs(BIG + 2)), 0).toLongArray());
        Tensor filled = Torch.zeros(Shape.of(2), TensorOptions.defaults().dtype(DType.INT64));
        Torch.noGrad(() -> filled.fill_(BIG));
        assertArrayEquals(new long[] {BIG, BIG}, filled.toLongArray());
        Tensor full = Torch.full(Shape.of(2), Scalar.of(BIG), TensorOptions.defaults().dtype(DType.INT64));
        assertArrayEquals(new long[] {BIG, BIG}, full.toLongArray());
        assertArrayEquals(new long[] {BIG, BIG + 1, BIG + 2, BIG + 3},
                Torch.arange(BIG, BIG + 4).toLongArray());
        assertArrayEquals(new long[] {BIG, BIG + 2}, longs(BIG, BIG + 2).clamp(0, Long.MAX_VALUE).toLongArray());
        assertArrayEquals(new long[] {3 * BIG}, longs(BIG).add(longs(BIG), 2L).toLongArray());
        assertArrayEquals(new long[] {BIG}, longs(BIG).pow(1).toLongArray());
        assertArrayEquals(new long[] {BIG + 1}, longs(BIG).addmm(
                Torch.tensor(new long[] {1}, Shape.of(1, 1)),
                Torch.tensor(new long[] {1}, Shape.of(1, 1))).toLongArray());
        assertEquals(BIG, longs(BIG).itemLong());
        assertThrows(JTorchException.class, () -> longs(BIG).item());
    }

    @Test
    void overlappingInPlaceMutationIsRejected() {
        Tensor matrix = Torch.tensor(new double[] {0, 1, 2, 3, 4, 5}, Shape.of(2, 3));
        Tensor row = matrix.select(0, 0);
        assertThrows(JTorchException.class, () -> Torch.noGrad(() -> matrix.add_(row)));
        assertThrows(JTorchException.class, () -> Torch.noGrad(() -> matrix.mul_(row)));
        assertThrows(JTorchException.class, () -> Torch.noGrad(() -> matrix.copy_(row)));

        Tensor sameGeometry = doubles(1, 2, 3);
        Torch.noGrad(() -> sameGeometry.add_(sameGeometry.view(3)));
        assertArrayEquals(new double[] {2, 4, 6}, sameGeometry.toDoubleArray(), 0);

        Tensor expanded = Torch.zeros(Shape.of(1, 3)).expand(2, 3);
        assertThrows(JTorchException.class,
                () -> Torch.noGrad(() -> expanded.add_(Torch.ones(Shape.of(2, 3)))));
    }

    @Test
    void zeroAndEmptySemanticsMatchPyTorch() {
        assertEquals(Shape.of(1), Torch.full(Shape.scalar(), Scalar.of(2.0)).flatten().shape());
        assertEquals(Shape.of(2), doubles(1, 2).t().shape());
        Tensor matrix = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(2, 2));
        assertEquals(10.0, matrix.sum(new int[] {}, false).item(), 0);
        assertEquals(2.5, matrix.mean(new int[] {}, false).item(), EPS);
        assertArrayEquals(new long[] {1, 1}, matrix.sum(new int[] {}, true).shape().toArray());
        assertEquals(Shape.of(0, 3), Torch.zeros(Shape.of(0, 3)).softmax(0).shape());
        assertEquals(Shape.of(0, 3), Torch.zeros(Shape.of(0, 3)).logSoftmax(0).shape());
        assertArrayEquals(new long[] {3, 3, 1}, Torch.zeros(Shape.of(2, 0, 3)).strides().toArray());

        Tensor scalar = Torch.full(Shape.scalar(), Scalar.of(3.0)).requiresGrad(true);
        assertEquals(3.0, scalar.sum(0, false).item(), 0);
        assertEquals(3.0, scalar.transpose(0, 0).item(), 0);
        assertEquals(3.0, scalar.squeeze(0).item(), 0);
        Tensor softmax = scalar.softmax(0);
        assertEquals(1.0, softmax.item(), EPS);
        softmax.backward();
        assertEquals(0.0, scalar.grad().item(), EPS);

        Tensor scalarMax = Torch.full(Shape.scalar(), Scalar.of(2.0));
        assertEquals(2.0, scalarMax.maxWithIndices(0, false).values().item(), 0);
        assertEquals(0, scalarMax.maxWithIndices(0, false).indices().itemLong());
    }

    @Test
    void emptyAndScalarReductionsKeepGradientShapes() {
        Tensor scalar = Torch.full(Shape.scalar(), Scalar.of(2.0)).requiresGrad(true);
        scalar.sum(0, false).backward();
        assertEquals(Shape.scalar(), scalar.grad().shape());
        assertEquals(1.0, scalar.grad().item(), 0);
    }

    @Test
    void resizeAndViewGeometry() {
        Tensor base = Torch.tensor(new double[] {1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
        Tensor transposed = base.t();
        assertEquals(Shape.of(3, 2), transposed.reshape(3, 2).shape());
        assertTrue(transposed.reshape(3, 2).sharesStorageWith(base));
        assertEquals(Shape.of(3, 2), base.reshape(3, -1).shape());
        assertEquals(Shape.of(6), base.reshape(-1).shape());
        assertEquals(Shape.of(3, 2), base.t().unsqueeze(0).view(3, 2).shape());
    }

    @Test
    void castsUseNonZeroSemantics() {
        Tensor values = Torch.tensor(new double[] {0.5, -0.5, 2.0, Double.NaN}, Shape.of(4));
        assertArrayEquals(new boolean[] {true, true, true, true}, values.toBool().toBooleanArray());
        Tensor target = Torch.zeros(Shape.of(2), TensorOptions.defaults().dtype(DType.BOOL));
        Torch.noGrad(() -> target.copy_(Torch.tensor(new double[] {0.5, 2.0}, Shape.of(2))));
        assertArrayEquals(new boolean[] {true, true}, target.toBooleanArray());
    }

    @Test
    void matmulSupportsVectorAndBatchedRanks() {
        assertEquals(Shape.of(4), doubles(1, 2, 3).matmul(Torch.zeros(Shape.of(3, 4))).shape());
        assertEquals(Shape.of(2), Torch.zeros(Shape.of(2, 3)).matmul(doubles(1, 2, 3)).shape());
        Tensor left = Torch.ones(Shape.of(2, 4, 3));
        Tensor right = Torch.ones(Shape.of(3));
        assertEquals(Shape.of(2, 4), left.matmul(right).shape());
        Tensor batchedLeft = Torch.ones(Shape.of(2, 1, 3, 4));
        Tensor batchedRight = Torch.ones(Shape.of(1, 4, 4, 5));
        assertEquals(Shape.of(2, 4, 3, 5), batchedLeft.matmul(batchedRight).shape());

        Tensor vector = doubles(1, 2, 3).requiresGrad(true);
        Tensor matrix = Torch.tensor(new double[] {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1}, Shape.of(3, 4))
                .requiresGrad(true);
        vector.matmul(matrix).sum().backward();
        assertArrayEquals(new double[] {1, 1, 2}, vector.grad().toDoubleArray(), EPS);
        assertArrayEquals(new double[] {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
                matrix.grad().toDoubleArray(), EPS);

        assertThrows(org.jtorch.core.ShapeException.class,
                () -> Torch.zeros(Shape.of(2, 3, 4)).mm(Torch.zeros(Shape.of(2, 4, 5))));
        assertThrows(org.jtorch.core.ShapeException.class,
                () -> Torch.zeros(Shape.of(2, 3)).bmm(Torch.zeros(Shape.of(3, 4))));
    }

    @Test
    void linearSupportsOneAndThreeDimensionalInputs() {
        Tensor weight = Torch.tensor(new double[] {1, 0, -1, 0.5, 1, 0}, Shape.of(2, 3));
        Tensor oneD = doubles(1, 2, 3);
        assertArrayEquals(new double[] {-2.0, 2.5}, oneD.linear(weight, null).toDoubleArray(), EPS);
        Tensor threeD = Torch.ones(Shape.of(2, 4, 3));
        assertEquals(Shape.of(2, 4, 2), threeD.linear(weight, null).shape());
        Tensor weightGrad = weight.clone().requiresGrad(true);
        Tensor input = doubles(1, 2, 3).requiresGrad(true);
        input.linear(weightGrad, null).sum().backward();
        assertArrayEquals(new double[] {1.5, 1, -1}, input.grad().toDoubleArray(), EPS);
        assertArrayEquals(new double[] {1, 2, 3, 1, 2, 3}, weightGrad.grad().toDoubleArray(), EPS);
    }

    @Test
    void extremumGradientsHandleTiesAndNaNs() {
        Tensor ties = doubles(1, 1, 1).requiresGrad(true);
        ties.max().backward();
        assertArrayEquals(new double[] {1.0 / 3, 1.0 / 3, 1.0 / 3}, ties.grad().toDoubleArray(), EPS);

        Tensor tiesMin = doubles(1, 1, 1).requiresGrad(true);
        tiesMin.min().backward();
        assertArrayEquals(new double[] {1.0 / 3, 1.0 / 3, 1.0 / 3}, tiesMin.grad().toDoubleArray(), EPS);

        Tensor nanMax = doubles(Double.NaN, 1).requiresGrad(true);
        nanMax.max().backward();
        assertArrayEquals(new double[] {1, 0}, nanMax.grad().toDoubleArray(), EPS);

        Tensor allNan = doubles(Double.NaN, Double.NaN).requiresGrad(true);
        allNan.max().backward();
        assertArrayEquals(new double[] {0.5, 0.5}, allNan.grad().toDoubleArray(), EPS);

        Tensor base = Torch.tensor(new double[] {1, 2, 3, 0}, Shape.of(2, 2)).requiresGrad(true);
        Tensor other = Torch.tensor(new double[] {2, 1}, Shape.of(1, 2)).requiresGrad(true);
        base.maximum(other).sum().backward();
        assertArrayEquals(new double[] {0, 1, 1, 0}, base.grad().toDoubleArray(), EPS);
        assertArrayEquals(new double[] {1, 1}, other.grad().toDoubleArray(), EPS);
    }

    @Test
    void unaryEdgeCasesMatchPyTorch() {
        Tensor zero = doubles(0).requiresGrad(true);
        zero.pow(0).sum().backward();
        assertArrayEquals(new double[] {0}, zero.grad().toDoubleArray(), EPS);

        Tensor nan = doubles(Double.NaN).requiresGrad(true);
        nan.relu().backward();
        assertArrayEquals(new double[] {1}, nan.grad().toDoubleArray(), EPS);

        Tensor bools = Torch.tensor(new boolean[] {true, false}, Shape.of(2));
        assertEquals(DType.INT64, bools.square().dtype());
        assertArrayEquals(new long[] {1, 0}, bools.square().toLongArray());
        assertThrows(DTypeException.class, bools::abs);
        assertThrows(DTypeException.class, bools::relu);
    }

    @Test
    void reductionsRejectUnsupportedInputs() {
        Tensor bools = Torch.tensor(new boolean[] {true, false}, Shape.of(2));
        assertThrows(DTypeException.class, bools::mean);
        assertThrows(DTypeException.class, bools::argmax);
        assertThrows(DTypeException.class, longs(1, 2)::mean);
    }

    @Test
    void fullInfersDtypeFromScalarKind() {
        assertEquals(DType.INT64, Torch.full(Shape.of(2), Scalar.of(5)).dtype());
        assertEquals(DType.BOOL, Torch.full(Shape.of(2), Scalar.of(true)).dtype());
        assertEquals(DType.FLOAT32, Torch.full(Shape.of(2), Scalar.of(1.5)).dtype());
    }

    @Test
    void randomFactoriesValidateDtypesButRandintAllowsFloatsLikePyTorch() {
        assertThrows(DTypeException.class,
                () -> Torch.randn(Shape.of(2), TensorOptions.defaults().dtype(DType.INT64)));
        assertThrows(DTypeException.class,
                () -> Torch.rand(Shape.of(2), TensorOptions.defaults().dtype(DType.INT64)));
        Tensor floats = Torch.randint(0, 5, Shape.of(4), TensorOptions.defaults().dtype(DType.FLOAT32));
        assertEquals(DType.FLOAT32, floats.dtype());
        for (double value : floats.toDoubleArray()) {
            assertTrue(value >= 0 && value < 5);
        }
    }

    @Test
    void metaAndCpuValidationAgree() {
        Tensor metaBool = Torch.zeros(Shape.of(2),
                TensorOptions.defaults().dtype(DType.BOOL).device(Device.meta()));
        assertThrows(DTypeException.class, metaBool::neg);
        assertThrows(DTypeException.class, metaBool::abs);
        Tensor meta = Torch.zeros(Shape.of(2), TensorOptions.defaults().device(Device.meta()));
        assertThrows(IllegalArgumentException.class, () -> meta.clamp(1, -1));
        Tensor emptyMeta = Torch.zeros(Shape.of(0, 3), TensorOptions.defaults().device(Device.meta()));
        assertThrows(org.jtorch.core.ShapeException.class, () -> emptyMeta.maxWithIndices(0, false));
        Tensor metaInt = Torch.zeros(Shape.of(2),
                TensorOptions.defaults().dtype(DType.INT64).device(Device.meta()));
        assertThrows(DTypeException.class, metaInt::mean);
    }

    @Test
    void addmmIgnoresZeroScaledOperands() {
        Tensor self = Torch.full(Shape.of(2), Scalar.of(Double.NaN));
        Tensor mat1 = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(2, 2));
        Tensor mat2 = Torch.tensor(new double[] {1, 0, 0, 1}, Shape.of(2, 2));
        assertArrayEquals(new double[] {1, 2, 3, 4}, self.addmm(mat1, mat2, 0.0, 1.0).toDoubleArray(), 0);
        Tensor scalarSelf = Torch.full(Shape.scalar(), Scalar.of(2.0));
        assertArrayEquals(new double[] {3, 4, 5, 6}, scalarSelf.addmm(mat1, mat2).toDoubleArray(), 0);
    }

    @Test
    void matmulSavesInputsAgainstMutation() {
        Tensor a = Torch.tensor(new double[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
        Tensor b = Torch.tensor(new double[] {1, 0, 0, 1}, Shape.of(2, 2)).requiresGrad(true);
        Tensor product = a.matmul(b);
        Torch.noGrad(() -> a.fill_(0));
        assertThrows(AutogradException.class, () -> product.sum().backward());
    }

    @Test
    void cloneOfANonDenseViewBecomesContiguousButCastPreservesStrides() {
        Tensor base = Torch.tensor(new double[] {0, 1, 2, 3, 4, 5}, Shape.of(2, 3));
        Tensor narrowed = base.narrow(1, 0, 2);
        assertTrue(narrowed.clone().isContiguous());
        assertArrayEquals(new long[] {2, 1}, narrowed.clone().strides().toArray());
        // PyTorch's to(dtype) preserves strides for non-dense views.
        assertArrayEquals(new long[] {3, 1}, narrowed.to(DType.FLOAT32).strides().toArray());
    }

    @Test
    void emptyClassCrossEntropyProducesEmptyGradient() {
        Tensor logits = Torch.zeros(Shape.of(0, 0), TensorOptions.defaults()
                .dtype(DType.FLOAT64).requiresGrad(true));
        Tensor target = Torch.zeros(Shape.of(0), TensorOptions.defaults().dtype(DType.INT64));
        logits.crossEntropy(target, "mean").backward();
        assertArrayEquals(new long[] {0, 0}, logits.grad().shape().toArray());
    }

    @Test
    void leafGradientsAreIndependentAndContiguous() {
        Tensor x = doubles(1, 2).requiresGrad(true);
        Tensor y = doubles(3, 4).requiresGrad(true);
        x.add(y).sum().backward();
        assertArrayEquals(new double[] {1, 1}, x.grad().toDoubleArray(), 0);
        assertArrayEquals(new double[] {1, 1}, y.grad().toDoubleArray(), 0);
        assertTrue(x.grad().isContiguous());
        assertTrue(!x.grad().sharesStorageWith(y.grad()));

        Tensor a = doubles(1, 2).requiresGrad(true);
        Tensor b = doubles(3, 4).requiresGrad(true);
        Tensor.cat(List.of(a, b), 0).sum().backward();
        assertTrue(!a.grad().sharesStorageWith(b.grad()));
        assertTrue(a.grad().isContiguous());
        assertTrue(b.grad().isContiguous());

        Tensor c = doubles(1, 2).requiresGrad(true);
        Tensor d = doubles(3, 4).requiresGrad(true);
        Tensor.stack(List.of(c, d), 0).sum().backward();
        assertTrue(!c.grad().sharesStorageWith(d.grad()));
        assertTrue(c.grad().isContiguous());

        Tensor sum = doubles(1, 2).requiresGrad(true);
        sum.sum().backward();
        assertTrue(sum.grad().isContiguous());
        assertArrayEquals(new double[] {1, 1}, sum.grad().toDoubleArray(), 0);
    }

    @Test
    void forwardOnlyProdFailsLoudlyUnderGrad() {
        Tensor values = doubles(1, 2, 3).requiresGrad(true);
        assertThrows(org.jtorch.core.NotSupportedException.class, values::prod);
        assertEquals(6.0, Torch.noGrad((java.util.function.Supplier<Tensor>) values::prod).item(), 1e-7);
    }

    @Test
    void constructorRejectsArrayLengthMismatch() {
        assertThrows(IllegalArgumentException.class, () -> Torch.tensor(new float[] {1, 2, 3}, Shape.of(2)));
        assertThrows(IllegalArgumentException.class, () -> Torch.tensor(new float[] {1}, Shape.of(2)));
    }
}
