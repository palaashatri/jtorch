package org.jtorch.autograd;

import java.util.Arrays;
import org.jtorch.core.DType;
import org.jtorch.core.Op;
import org.jtorch.core.OpRegistry;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.core.autograd.Node;
import org.jtorch.core.dispatch.Dispatcher;
import org.jtorch.core.dispatch.DispatchKey;
import org.jtorch.core.value.Value;

/**
 * Native reverse-mode formulas. Every formula is expressed with dispatched operators, so higher-level composition
 * and meta execution behave consistently.
 */
final class BackwardKernels {
    private BackwardKernels() {}

    @FunctionalInterface
    interface Formula {
        Node create(Tensor[] inputs, Tensor[] outputs, Value[] args);
    }

    static void register(Dispatcher dispatcher) {
        // ---------------------------------------------------------------- views
        single(dispatcher, OpRegistry.RESHAPE, (in, out, args) -> new Node("ReshapeBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].reshape(in[0].shape().toArray())};
            }
        });
        single(dispatcher, OpRegistry.VIEW, (in, out, args) -> new Node("ViewBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].reshape(in[0].shape().toArray())};
            }
        });
        single(dispatcher, OpRegistry.FLATTEN, (in, out, args) -> new Node("FlattenBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].reshape(in[0].shape().toArray())};
            }
        });
        single(dispatcher, OpRegistry.TRANSPOSE, (in, out, args) -> new Node("TransposeBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].transpose(args[1].asInt(), args[2].asInt())};
            }
        });
        single(dispatcher, OpRegistry.PERMUTE, (in, out, args) -> {
            long[] raw = args[1].asLongArray();
            int[] inverse = new int[raw.length];
            for (int i = 0; i < raw.length; i++) {
                int dim = in[0].shape().normalizeDim((int) raw[i]);
                inverse[dim] = i;
            }
            return new Node("PermuteBackward0", 1) {
                @Override
                public Tensor[] backward(Tensor[] g) {
                    return new Tensor[] {g[0].permute(inverse)};
                }
            };
        });
        single(dispatcher, OpRegistry.NARROW, (in, out, args) -> new Node("NarrowBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor zeros = Tensor.zerosLike(in[0]);
                zeros.narrow(args[1].asInt(), args[2].asLong(), args[3].asLong()).copy_(g[0]);
                return new Tensor[] {zeros};
            }
        });
        single(dispatcher, OpRegistry.SELECT, (in, out, args) -> new Node("SelectBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor zeros = Tensor.zerosLike(in[0]);
                zeros.select(args[1].asInt(), args[2].asLong()).copy_(g[0]);
                return new Tensor[] {zeros};
            }
        });
        for (Op op : new Op[] {OpRegistry.SQUEEZE, OpRegistry.SQUEEZE_DIM, OpRegistry.UNSQUEEZE}) {
            single(dispatcher, op, (in, out, args) -> new Node("SqueezeBackward0", 1) {
                @Override
                public Tensor[] backward(Tensor[] g) {
                    return new Tensor[] {g[0].reshape(in[0].shape().toArray())};
                }
            });
        }
        single(dispatcher, OpRegistry.EXPAND, (in, out, args) -> new Node("ExpandBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {GradUtil.reduceToShape(g[0], in[0].shape())};
            }
        });
        single(dispatcher, OpRegistry.T, (in, out, args) -> new Node("TBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].t()};
            }
        });
        single(dispatcher, OpRegistry.CONTIGUOUS, (in, out, args) -> new Node("ContiguousBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0]};
            }
        });
        single(dispatcher, OpRegistry.CLONE, (in, out, args) -> new Node("CloneBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0]};
            }
        });
        single(dispatcher, OpRegistry.TO_DTYPE, (in, out, args) -> new Node("ToBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].to(in[0].dtype())};
            }
        });

        // ---------------------------------------------------------------- elementwise
        single(dispatcher, OpRegistry.ADD_TENSOR, (in, out, args) -> new Node("AddBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                double alpha = args[2].asScalar().toDouble();
                return new Tensor[] {
                    GradUtil.reduceToShape(g[0], in[0].shape()),
                    GradUtil.reduceToShape(GradUtil.scale(g[0], alpha), in[1].shape())
                };
            }
        });
        single(dispatcher, OpRegistry.SUB_TENSOR, (in, out, args) -> new Node("SubBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                double alpha = args[2].asScalar().toDouble();
                return new Tensor[] {
                    GradUtil.reduceToShape(g[0], in[0].shape()),
                    GradUtil.reduceToShape(GradUtil.scale(g[0].neg(), alpha), in[1].shape())
                };
            }
        });
        single(dispatcher, OpRegistry.MUL_TENSOR, (in, out, args) -> new Node("MulBackward0", 1) {
            {
                save(in[0]);
                save(in[1]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor a = saved(0);
                Tensor b = saved(1);
                return new Tensor[] {
                    GradUtil.reduceToShape(g[0].mul(b), a.shape()),
                    GradUtil.reduceToShape(g[0].mul(a), b.shape())
                };
            }
        });
        single(dispatcher, OpRegistry.DIV_TENSOR, (in, out, args) -> new Node("DivBackward0", 1) {
            {
                save(in[0]);
                save(in[1]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor a = saved(0);
                Tensor b = saved(1);
                Tensor gradA = g[0].div(b);
                Tensor gradB = g[0].mul(a).div(b.mul(b)).neg();
                return new Tensor[] {
                    GradUtil.reduceToShape(gradA, a.shape()), GradUtil.reduceToShape(gradB, b.shape())
                };
            }
        });
        single(dispatcher, OpRegistry.POW_TENSOR_SCALAR, (in, out, args) -> {
            double exponent = args[1].asScalar().toDouble();
            return new Node("PowBackward0", 1) {
                {
                    save(in[0]);
                }

                @Override
                public Tensor[] backward(Tensor[] g) {
                    if (exponent == 0) {
                        return new Tensor[] {Tensor.zerosLike(in[0])};
                    }
                    if (exponent == 1) {
                        return new Tensor[] {g[0]};
                    }
                    Tensor base = saved(0);
                    Tensor local = base.pow(exponent - 1.0);
                    return new Tensor[] {GradUtil.scale(g[0].mul(local), exponent)};
                }
            };
        });
        single(dispatcher, OpRegistry.POW_TENSOR_TENSOR, (in, out, args) -> new Node("PowBackward1", 1) {
            {
                save(in[0]);
                save(in[1]);
                save(out[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor base = saved(0);
                Tensor exponent = saved(1);
                Tensor result = saved(2);
                Tensor gradBase = g[0].mul(exponent).mul(base.pow(exponent.sub(GradUtil.constant(exponent, 1))));
                Tensor gradExponent = g[0].mul(result).mul(base.log());
                return new Tensor[] {
                    GradUtil.reduceToShape(gradBase, base.shape()),
                    GradUtil.reduceToShape(gradExponent, exponent.shape())
                };
            }
        });
        single(dispatcher, OpRegistry.MAXIMUM, (in, out, args) -> binaryExtremum(in, true));
        single(dispatcher, OpRegistry.MINIMUM, (in, out, args) -> binaryExtremum(in, false));

        single(dispatcher, OpRegistry.NEG, (in, out, args) -> new Node("NegBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].neg()};
            }
        });
        single(dispatcher, OpRegistry.ABS, (in, out, args) -> new Node("AbsBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].mul(saved(0).sign())};
            }
        });
        single(dispatcher, OpRegistry.SQUARE, (in, out, args) -> new Node("SquareBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = saved(0);
                return new Tensor[] {g[0].mul(input).mul(GradUtil.constant(input, 2))};
            }
        });
        single(dispatcher, OpRegistry.SQRT, (in, out, args) -> new Node("SqrtBackward0", 1) {
            {
                save(out[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor result = saved(0);
                return new Tensor[] {
                    g[0].div(result.mul(GradUtil.constant(result, 2)))
                };
            }
        });
        single(dispatcher, OpRegistry.EXP, (in, out, args) -> new Node("ExpBackward0", 1) {
            {
                save(out[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].mul(saved(0))};
            }
        });
        single(dispatcher, OpRegistry.LOG, (in, out, args) -> new Node("LogBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].div(saved(0))};
            }
        });
        single(dispatcher, OpRegistry.TANH, (in, out, args) -> new Node("TanhBackward0", 1) {
            {
                save(out[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor result = saved(0);
                Tensor one = GradUtil.constant(result, 1);
                return new Tensor[] {g[0].mul(one.sub(result.mul(result)))};
            }
        });
        single(dispatcher, OpRegistry.SIGMOID, (in, out, args) -> new Node("SigmoidBackward0", 1) {
            {
                save(out[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor result = saved(0);
                Tensor one = GradUtil.constant(result, 1);
                return new Tensor[] {g[0].mul(result).mul(one.sub(result))};
            }
        });
        single(dispatcher, OpRegistry.RELU, (in, out, args) -> new Node("ReluBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = saved(0);
                Tensor positive = input.gt(GradUtil.constant(input, 0)).to(input.dtype());
                Tensor nan = input.ne(input).to(input.dtype());
                return new Tensor[] {g[0].mul(positive.add(nan))};
            }
        });
        single(dispatcher, OpRegistry.RECIPROCAL, (in, out, args) -> new Node("ReciprocalBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = saved(0);
                return new Tensor[] {g[0].div(input.mul(input)).neg()};
            }
        });
        single(dispatcher, OpRegistry.SIN, (in, out, args) -> new Node("SinBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].mul(saved(0).cos())};
            }
        });
        single(dispatcher, OpRegistry.COS, (in, out, args) -> new Node("CosBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].mul(saved(0).sin()).neg()};
            }
        });
        single(dispatcher, OpRegistry.LOG1P, (in, out, args) -> new Node("Log1pBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = saved(0);
                return new Tensor[] {g[0].div(input.add(GradUtil.constant(input, 1)))};
            }
        });
        single(dispatcher, OpRegistry.CLAMP, (in, out, args) -> new Node("ClampBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = saved(0);
                Tensor mask = null;
                if (!args[1].isNone()) {
                    mask = input.ge(GradUtil.constant(input, args[1].asScalar().toDouble()));
                }
                if (!args[2].isNone()) {
                    Tensor upper = input.le(GradUtil.constant(input, args[2].asScalar().toDouble()));
                    mask = mask == null ? upper : mask.mul(upper);
                }
                if (mask == null) {
                    return new Tensor[] {g[0]};
                }
                return new Tensor[] {g[0].mul(mask.to(input.dtype()))};
            }
        });

        // ---------------------------------------------------------------- reductions
        single(dispatcher, OpRegistry.SUM, (in, out, args) -> new Node("SumBackward0", 1) {
            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {g[0].expand(in[0].shape().toArray())};
            }
        });
        single(dispatcher, OpRegistry.SUM_DIM_INTLIST, (in, out, args) -> {
            int[] dims = scalarDims(in[0], args[1].asLongArray());
            boolean keepdim = args[2].asBoolean();
            return new Node("SumBackward1", 1) {
                @Override
                public Tensor[] backward(Tensor[] g) {
                    return new Tensor[] {GradUtil.summation(g[0], in[0], dims, keepdim)};
                }
            };
        });
        single(dispatcher, OpRegistry.MEAN, (in, out, args) -> {
            long count = in[0].numel();
            return new Node("MeanBackward0", 1) {
                @Override
                public Tensor[] backward(Tensor[] g) {
                    Tensor expanded = g[0].expand(in[0].shape().toArray());
                    return new Tensor[] {GradUtil.scale(expanded, 1.0 / count)};
                }
            };
        });
        single(dispatcher, OpRegistry.MEAN_DIM, (in, out, args) -> {
            int[] dims = scalarDims(in[0], args[1].asLongArray());
            boolean keepdim = args[2].asBoolean();
            long total = dims.length == 0 ? in[0].numel() : 1;
            for (int dim : dims) {
                total *= in[0].size(dim);
            }
            final long count = total;
            return new Node("MeanBackward1", 1) {
                @Override
                public Tensor[] backward(Tensor[] g) {
                    Tensor expanded = GradUtil.summation(g[0], in[0], dims, keepdim);
                    return new Tensor[] {GradUtil.scale(expanded, 1.0 / count)};
                }
            };
        });
        single(dispatcher, OpRegistry.MAX, (in, out, args) -> new Node("MaxBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = saved(0);
                // PyTorch distributes the gradient evenly among all maxima for full reductions.
                Tensor extreme = input.max();
                Tensor mask = extremumMask(input, extreme);
                Tensor count = mask.sum().to(input.dtype());
                return new Tensor[] {mask.mul(g[0]).div(count)};
            }
        });
        single(dispatcher, OpRegistry.MIN, (in, out, args) -> new Node("MinBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = saved(0);
                Tensor extreme = input.min();
                Tensor mask = extremumMask(input, extreme);
                Tensor count = mask.sum().to(input.dtype());
                return new Tensor[] {mask.mul(g[0]).div(count)};
            }
        });
        single(dispatcher, OpRegistry.MAX_DIM, (in, out, args) -> extremumDim(in, out, args));
        single(dispatcher, OpRegistry.MIN_DIM, (in, out, args) -> extremumDim(in, out, args));

        // ---------------------------------------------------------------- matrix
        single(dispatcher, OpRegistry.MATMUL, (in, out, args) -> new Node("MatmulBackward0", 1) {
            {
                save(in[0]);
                save(in[1]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return GradUtil.matmulGrads(saved(0), saved(1), g[0]);
            }
        });
        single(dispatcher, OpRegistry.MM, (in, out, args) -> new Node("MmBackward0", 1) {
            {
                save(in[0]);
                save(in[1]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return GradUtil.matmulGrads(saved(0), saved(1), g[0]);
            }
        });
        single(dispatcher, OpRegistry.BMM, (in, out, args) -> new Node("BmmBackward0", 1) {
            {
                save(in[0]);
                save(in[1]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return GradUtil.matmulGrads(saved(0), saved(1), g[0]);
            }
        });
        single(dispatcher, OpRegistry.ADDMM, (in, out, args) -> {
            double beta = args[3].asScalar().toDouble();
            double alpha = args[4].asScalar().toDouble();
            return new Node("AddmmBackward0", 1) {
                {
                    save(in[0]);
                    save(in[1]);
                    save(in[2]);
                }

                @Override
                public Tensor[] backward(Tensor[] g) {
                    Tensor self = saved(0);
                    Tensor mat1 = saved(1);
                    Tensor mat2 = saved(2);
                    Tensor gradSelf = GradUtil.reduceToShape(GradUtil.scale(g[0], beta), self.shape());
                    Tensor gradMat1 = GradUtil.reduceToShape(
                            GradUtil.scale(g[0].matmul(mat2.t()), alpha), mat1.shape());
                    Tensor gradMat2 = GradUtil.reduceToShape(
                            GradUtil.scale(mat1.t().matmul(g[0]), alpha), mat2.shape());
                    return new Tensor[] {gradSelf, gradMat1, gradMat2};
                }
            };
        });

        // ---------------------------------------------------------------- combination
        single(dispatcher, OpRegistry.CAT, (in, out, args) -> {
            int dim = in[0].shape().normalizeDim(args[1].asInt());
            return new Node("CatBackward0", 1) {
                @Override
                public Tensor[] backward(Tensor[] g) {
                    Tensor[] result = new Tensor[in.length];
                    long offset = 0;
                    for (int i = 0; i < in.length; i++) {
                        result[i] = g[0].narrow(dim, offset, in[i].size(dim));
                        offset += in[i].size(dim);
                    }
                    return result;
                }
            };
        });
        single(dispatcher, OpRegistry.STACK, (in, out, args) -> {
            int dim = args[1].asInt();
            return new Node("StackBackward0", 1) {
                @Override
                public Tensor[] backward(Tensor[] g) {
                    Tensor[] result = new Tensor[in.length];
                    for (int i = 0; i < in.length; i++) {
                        result[i] = g[0].select(dim, i);
                    }
                    return result;
                }
            };
        });

        // ---------------------------------------------------------------- convolution and pooling
        single(dispatcher, OpRegistry.CONV2D, (in, out, args) -> new Node("Conv2dBackward0", 1) {
            {
                save(in[0]);
                save(in[1]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = saved(0);
                Tensor weight = saved(1);
                long[] stride = args[3].asLongArray();
                long[] padding = args[4].asLongArray();
                long[] dilation = args[5].asLongArray();
                Tensor gradInput = Dispatcher.get()
                        .call(
                                OpRegistry.CONV2D_BACKWARD_INPUT,
                                Value.tensor(g[0]),
                                Value.tensor(weight),
                                Value.integers(input.shape().toArray()),
                                Value.integers(stride),
                                Value.integers(padding),
                                Value.integers(dilation))[0]
                        .asTensor();
                Tensor gradWeight = Dispatcher.get()
                        .call(
                                OpRegistry.CONV2D_BACKWARD_WEIGHT,
                                Value.tensor(input),
                                Value.tensor(g[0]),
                                Value.integers(weight.shape().toArray()),
                                Value.integers(stride),
                                Value.integers(padding),
                                Value.integers(dilation))[0]
                        .asTensor();
                Tensor gradBias = args[2].isNone() ? null : g[0].sum(new int[] {0, 2, 3}, false);
                return new Tensor[] {gradInput, gradWeight, gradBias};
            }
        });
        single(dispatcher, OpRegistry.MAX_POOL2D, (in, out, args) -> new Node("MaxPool2dBackward0", 1) {
            {
                save(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                return new Tensor[] {Dispatcher.get()
                        .call(
                                OpRegistry.MAX_POOL2D_BACKWARD,
                                Value.tensor(g[0]),
                                Value.tensor(saved(0)),
                                Value.integers(args[1].asLongArray()),
                                Value.integers(args[2].asLongArray()),
                                Value.integers(args[3].asLongArray()),
                                Value.integers(args[4].asLongArray()))[0]
                        .asTensor()};
            }
        });

        // ---------------------------------------------------------------- losses
        single(dispatcher, OpRegistry.CROSS_ENTROPY, (in, out, args) -> {
            String reduction = args[2].asString();
            int classes = (int) in[0].size(in[0].dim() - 1);
            long rows = classes == 0 ? 0 : in[0].numel() / classes;
            return new Node("CrossEntropyBackward0", 1) {
                {
                    save(in[0]);
                    save(in[1]);
                }

                @Override
                public Tensor[] backward(Tensor[] g) {
                    Tensor logits = saved(0);
                    Tensor target = saved(1);
                    Tensor probabilities = logits.softmax(1);
                    Tensor oneHot = GradUtil.oneHot(target, classes, logits.dtype());
                    Tensor difference = probabilities.sub(oneHot);
                    Tensor gradient = switch (reduction) {
                        case "mean" -> GradUtil.scale(difference.mul(g[0]), 1.0 / rows);
                        case "sum" -> difference.mul(g[0]);
                        default -> difference.mul(g[0].unsqueeze(1));
                    };
                    return new Tensor[] {gradient};
                }
            };
        });
    }

    // ------------------------------------------------------------------ helpers

    private static Tensor extremumMask(Tensor input, Tensor extreme) {
        Tensor equalMask = input.eq(extreme).to(input.dtype());
        Tensor nanInput = input.ne(input).to(input.dtype());
        Tensor nanExtreme = extreme.ne(extreme).to(input.dtype());
        return equalMask.add(nanInput.mul(nanExtreme));
    }

    private static Node binaryExtremum(Tensor[] inputs, boolean maximum) {
        return new Node(maximum ? "MaximumBackward0" : "MinimumBackward0", 1) {
            {
                save(inputs[0]);
                save(inputs[1]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor a = saved(0);
                Tensor b = saved(1);
                Tensor aWins = maximum ? a.gt(b) : a.lt(b);
                Tensor bWins = maximum ? b.gt(a) : b.lt(a);
                Tensor tie = a.eq(b);
                Tensor half = GradUtil.constant(a, 0.5).to(g[0].dtype());
                Tensor weightA = aWins.to(a.dtype()).add(tie.to(a.dtype()).mul(half));
                Tensor weightB = bWins.to(g[0].dtype()).add(tie.to(g[0].dtype()).mul(half));
                return new Tensor[] {
                    GradUtil.reduceToShape(g[0].mul(weightA), a.shape()),
                    GradUtil.reduceToShape(g[0].mul(weightB), b.shape())
                };
            }
        };
    }

    private static int[] scalarDims(Tensor input, long[] rawDims) {
        if (input.dim() == 0) {
            if (rawDims.length == 0
                    || (rawDims.length == 1 && (rawDims[0] == 0 || rawDims[0] == -1))) {
                return new int[0];
            }
            throw new IndexOutOfBoundsException("Dimension out of range for a 0-dimensional tensor");
        }
        return GradUtil.normalizeDims(input, rawDims);
    }

    private static Node extremumDim(Tensor[] inputs, Tensor[] outputs, Value[] args) {
        if (inputs[0].dim() == 0) {
            return new Node("MaxBackward1", 2) {
                @Override
                public Tensor[] backward(Tensor[] g) {
                    return new Tensor[] {g[0].reshape(inputs[0].shape().toArray()), null};
                }
            };
        }
        int dim = inputs[0].shape().normalizeDim(args[1].asInt());
        boolean keepdim = args[2].asBoolean();
        return new Node("MaxBackward1", 2) {
            {
                save(outputs[1]);
            }

            @Override
            public Tensor[] backward(Tensor[] g) {
                Tensor input = inputs[0];
                Tensor indices = saved(0);
                if (!keepdim) {
                    indices = indices.unsqueeze(dim);
                }
                long[] rangeShape = new long[input.dim()];
                java.util.Arrays.fill(rangeShape, 1);
                rangeShape[dim] = input.size(dim);
                Tensor range = GradUtil.arange(0, input.size(dim), DType.INT64, input.device())
                        .reshape(rangeShape);
                Tensor mask = range.eq(indices).to(input.dtype());
                Tensor values = keepdim ? g[0] : g[0].unsqueeze(dim);
                return new Tensor[] {mask.mul(values).expand(input.shape().toArray())};
            }
        };
    }

    private static void single(Dispatcher dispatcher, Op op, Formula formula) {
        dispatcher.register(op, DispatchKey.AUTOGRAD, (called, args, keys) -> {
            Value[] raw = dispatcher.callWithKeys(called, args, keys.without(DispatchKey.AUTOGRAD));
            Tensor[] outputs = new Tensor[raw.length];
            for (int i = 0; i < raw.length; i++) {
                outputs[i] = raw[i].asTensor();
            }
            Tensor[] inputs = extractInputs(args);
            for (Tensor output : outputs) {
                for (Tensor input : inputs) {
                    if (output == input) {
                        // Identity views such as contiguous() on already-contiguous data stay in place.
                        return raw;
                    }
                }
            }
            Node node = formula.create(inputs, outputs, args);
            node.setInputs(inputs);
            node.setOutputs(outputs);
            for (int i = 0; i < outputs.length; i++) {
                if (outputs[i].requiresGrad()) {
                    outputs[i].impl().setGradFn(node, i);
                }
            }
            return raw;
        });
    }

    private static Tensor[] extractInputs(Value[] args) {
        java.util.List<Tensor> tensors = new java.util.ArrayList<>();
        for (Value value : args) {
            if (value instanceof Value.TensorValue tensor) {
                tensors.add(tensor.value());
            } else if (value instanceof Value.TensorListValue list) {
                tensors.addAll(Arrays.asList(list.values()));
            }
        }
        return tensors.toArray(Tensor[]::new);
    }

}
