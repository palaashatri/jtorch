package org.jtorch.onnx;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.jtorch.api.Torch;
import org.jtorch.core.DType;
import org.jtorch.core.Tensor;
import org.jtorch.core.TensorOptions;

/** ONNX op → JTorch dispatched-op mapping for the supported subset. */
final class OnnxOps {
    private OnnxOps() {}

    static void execute(OnnxProto.Node node, Map<String, Tensor> env) {
        List<Tensor> in = new ArrayList<>();
        List<String> inNames = new ArrayList<>();
        for (String name : node.inputs()) {
            if (name == null || name.isEmpty()) {
                in.add(null);
                inNames.add("");
                continue;
            }
            Tensor value = env.get(name);
            if (value == null) {
                throw new OnnxException("ONNX node " + node.opType() + " is missing input '" + name + "'");
            }
            in.add(value);
            inNames.add(name);
        }
        Map<String, OnnxProto.Attribute> attrs = new java.util.LinkedHashMap<>();
        for (OnnxProto.Attribute attr : node.attributes()) {
            attrs.put(attr.name(), attr);
        }
        List<Tensor> out = new ArrayList<>();
        switch (node.opType()) {
            case "Identity" -> out.add(require(in, 0, node));
            case "Add" -> out.add(require(in, 0, node).add(require(in, 1, node)));
            case "Sub" -> out.add(require(in, 0, node).sub(require(in, 1, node)));
            case "Mul" -> out.add(require(in, 0, node).mul(require(in, 1, node)));
            case "Div" -> out.add(require(in, 0, node).div(require(in, 1, node)));
            case "Neg" -> out.add(require(in, 0, node).neg());
            case "Abs" -> out.add(require(in, 0, node).abs());
            case "Exp" -> out.add(require(in, 0, node).exp());
            case "Log" -> out.add(require(in, 0, node).log());
            case "Sigmoid" -> out.add(require(in, 0, node).sigmoid());
            case "Relu" -> out.add(require(in, 0, node).relu());
            case "MatMul" -> out.add(require(in, 0, node).matmul(require(in, 1, node)));
            case "Gemm" -> out.add(gemm(in, attrs));
            case "Softmax" -> out.add(require(in, 0, node).softmax(intAttr(attrs, "axis", -1)));
            case "Flatten" -> {
                int axis = intAttr(attrs, "axis", 1);
                int start = axis < 0 ? require(in, 0, node).dim() + axis : axis;
                out.add(require(in, 0, node).flatten(start, require(in, 0, node).dim() - 1));
            }
            case "Transpose" -> out.add(transpose(require(in, 0, node), intsAttr(attrs, "perm")));
            case "Reshape" -> {
                Tensor shapeTensor = require(in, 1, node);
                long[] shape = shapeTensor.toLongArray();
                int[] shapeInts = new int[shape.length];
                for (int i = 0; i < shape.length; i++) {
                    shapeInts[i] = (int) shape[i];
                }
                out.add(require(in, 0, node).reshape(toLongs(shapeInts)));
            }
            case "Concat" -> {
                int axis = intAttr(attrs, "axis", 0);
                List<Tensor> tensors = new ArrayList<>();
                for (Tensor t : in) {
                    tensors.add(t);
                }
                out.add(Tensor.cat(tensors, axis));
            }
            case "Squeeze" -> {
                List<Long> axes = intsAttr(attrs, "axes");
                Tensor value = require(in, 0, node);
                if (axes.isEmpty()) {
                    value = value.squeeze();
                } else {
                    for (long ax : axes) {
                        int dim = (int) ax;
                        if (dim < 0) {
                            dim += value.dim();
                        }
                        value = value.squeeze(dim);
                    }
                }
                out.add(value);
            }
            case "Unsqueeze" -> {
                List<Long> axes = intsAttr(attrs, "axes");
                Tensor value = require(in, 0, node);
                if (axes.isEmpty() && in.size() > 1 && in.get(1) != null) {
                    for (long ax : in.get(1).toLongArray()) {
                        int dim = (int) ax;
                        if (dim < 0) {
                            dim += value.dim() + 1;
                        }
                        value = value.unsqueeze(dim);
                    }
                    out.add(value);
                } else {
                    for (long ax : axes) {
                        int dim = (int) ax;
                        if (dim < 0) {
                            dim += value.dim() + 1;
                        }
                        value = value.unsqueeze(dim);
                    }
                    out.add(value);
                }
            }
            case "Gather" -> {
                int axis = intAttr(attrs, "axis", 0);
                out.add(require(in, 0, node).gather(axis, require(in, 1, node)));
            }
            case "Constant" -> {
                OnnxProto.Attribute valueAttr = attrs.get("value");
                if (valueAttr == null || valueAttr.tensorValue() == null) {
                    throw new OnnxException("Constant node requires a value tensor attribute");
                }
                out.add(OnnxTensorDecoder.decode(valueAttr.tensorValue()));
            }
            case "Conv" -> out.add(conv(in, attrs));
            case "MaxPool" -> out.add(maxPool(in, attrs));
            case "AveragePool" -> out.add(avgPool(in, attrs));
            case "BatchNormalization" -> out.add(batchNorm(in, attrs));
            default -> throw new OnnxException("ONNX unsupported operator " + node.opType());
        }
        for (int i = 0; i < node.outputs().size(); i++) {
            String name = node.outputs().get(i);
            if (name == null || name.isEmpty()) {
                continue;
            }
            if (i >= out.size()) {
                throw new OnnxException("ONNX node " + node.opType() + " produced fewer outputs than declared");
            }
            env.put(name, out.get(i));
        }
    }

    private static Tensor require(List<Tensor> in, int index, OnnxProto.Node node) {
        if (index >= in.size() || in.get(index) == null) {
            throw new OnnxException("ONNX node " + node.opType() + " is missing required input " + index);
        }
        return in.get(index);
    }

    private static int intAttr(Map<String, OnnxProto.Attribute> attrs, String name, int fallback) {
        OnnxProto.Attribute attr = attrs.get(name);
        if (attr == null) {
            return fallback;
        }
        if (attr.intValue() != null) {
            return attr.intValue().intValue();
        }
        if (!attr.ints().isEmpty()) {
            return attr.ints().get(0).intValue();
        }
        return fallback;
    }

    private static double doubleAttr(Map<String, OnnxProto.Attribute> attrs, String name, double fallback) {
        OnnxProto.Attribute attr = attrs.get(name);
        if (attr == null || attr.floatValue() == null) {
            return fallback;
        }
        return attr.floatValue();
    }

    private static List<Long> intsAttr(Map<String, OnnxProto.Attribute> attrs, String name) {
        OnnxProto.Attribute attr = attrs.get(name);
        if (attr == null) {
            return List.of();
        }
        if (!attr.ints().isEmpty()) {
            return attr.ints();
        }
        if (attr.intValue() != null) {
            return List.of(attr.intValue());
        }
        return List.of();
    }

    private static long[] toLongs(int[] values) {
        long[] out = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }

    private static int[] toInts(List<Long> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = Math.toIntExact(values.get(i));
        }
        return out;
    }

    private static Tensor gemm(List<Tensor> in, Map<String, OnnxProto.Attribute> attrs) {
        double alpha = doubleAttr(attrs, "alpha", 1.0);
        double beta = doubleAttr(attrs, "beta", 1.0);
        int transA = intAttr(attrs, "transA", 0);
        int transB = intAttr(attrs, "transB", 0);
        Tensor a = require(in, 0, new OnnxProto.Node(List.of(), List.of(), "Gemm", List.of()));
        Tensor b = require(in, 1, new OnnxProto.Node(List.of(), List.of(), "Gemm", List.of()));
        if (transA != 0) {
            a = a.t();
        }
        if (transB != 0) {
            b = b.t();
        }
        Tensor product = a.matmul(b);
        if (alpha != 1.0) {
            product = product.mul(alpha);
        }
        if (in.size() > 2 && in.get(2) != null) {
            Tensor c = in.get(2);
            if (beta == 0.0) {
                return product;
            }
            if (beta != 1.0) {
                c = c.mul(beta);
            }
            return product.add(c);
        }
        return product;
    }

    private static Tensor transpose(Tensor input, List<Long> perm) {
        if (perm.isEmpty()) {
            // Reverse order per ONNX spec when perm is empty.
            int[] dims = new int[input.dim()];
            for (int i = 0; i < dims.length; i++) {
                dims[i] = dims.length - 1 - i;
            }
            return input.permute(dims);
        }
        return input.permute(toInts(perm));
    }

    private static Tensor conv(List<Tensor> in, Map<String, OnnxProto.Attribute> attrs) {
        Tensor input = require(in, 0, new OnnxProto.Node(List.of(), List.of(), "Conv", List.of()));
        Tensor weight = require(in, 1, new OnnxProto.Node(List.of(), List.of(), "Conv", List.of()));
        List<Long> strides = intsAttr(attrs, "strides");
        List<Long> pads = intsAttr(attrs, "pads");
        List<Long> dilations = intsAttr(attrs, "dilations");
        int group = intAttr(attrs, "group", 1);
        if (input.dim() != 4 || weight.dim() != 4) {
            throw new OnnxException("ONNX Conv foundations support NCHW 4-D only, got " + input.shape()
                    + " / " + weight.shape());
        }
        int[] s = strides.isEmpty() ? new int[] {1, 1} : toInts(strides);
        int[] p;
        if (!pads.isEmpty()) {
            // ONNX pads are [top, left, bottom, right]; JTorch conv padding is symmetric per side.
            long top = pads.size() > 0 ? pads.get(0) : 0;
            long left = pads.size() > 1 ? pads.get(1) : top;
            long bottom = pads.size() > 2 ? pads.get(2) : top;
            long right = pads.size() > 3 ? pads.get(3) : left;
            if (top != bottom || left != right) {
                throw new OnnxException("ONNX Conv asymmetric pads are not supported in this build");
            }
            p = new int[] {(int) top, (int) left};
        } else {
            p = new int[] {0, 0};
        }
        int[] d = dilations.isEmpty() ? new int[] {1, 1} : toInts(dilations);
        Tensor bias = in.size() > 2 ? in.get(2) : null;
        return input.conv2d(weight, bias, s, p, d, group);
    }

    private static Tensor maxPool(List<Tensor> in, Map<String, OnnxProto.Attribute> attrs) {
        Tensor input = require(in, 0, new OnnxProto.Node(List.of(), List.of(), "MaxPool", List.of()));
        if (input.dim() != 4) {
            throw new OnnxException("ONNX MaxPool foundations support NCHW 4-D only, got " + input.shape());
        }
        int[] kernel = toInts(intsAttr(attrs, "kernel_shape"));
        int[] strides = intsAttr(attrs, "strides").isEmpty() ? kernel : toInts(intsAttr(attrs, "strides"));
        int[] pads = toInts(intsAttr(attrs, "pads"));
        int[] p = pads.length >= 4 ? new int[] {pads[0], pads[1]} : new int[] {0, 0};
        if (pads.length >= 4 && (pads[0] != pads[2] || pads[1] != pads[3])) {
            throw new OnnxException("ONNX MaxPool asymmetric pads are not supported in this build");
        }
        int[] dil = intsAttr(attrs, "dilations").isEmpty() ? new int[] {1, 1} : toInts(intsAttr(attrs, "dilations"));
        boolean ceil = intAttr(attrs, "ceil_mode", 0) != 0;
        return input.maxPool2d(kernel, strides, p, dil, ceil);
    }

    private static Tensor avgPool(List<Tensor> in, Map<String, OnnxProto.Attribute> attrs) {
        Tensor input = require(in, 0, new OnnxProto.Node(List.of(), List.of(), "AveragePool", List.of()));
        if (input.dim() != 4) {
            throw new OnnxException("ONNX AveragePool foundations support NCHW 4-D only, got " + input.shape());
        }
        int[] kernel = toInts(intsAttr(attrs, "kernel_shape"));
        int[] strides = intsAttr(attrs, "strides").isEmpty() ? kernel : toInts(intsAttr(attrs, "strides"));
        int[] pads = toInts(intsAttr(attrs, "pads"));
        int[] p = pads.length >= 4 ? new int[] {pads[0], pads[1]} : new int[] {0, 0};
        if (pads.length >= 4 && (pads[0] != pads[2] || pads[1] != pads[3])) {
            throw new OnnxException("ONNX AveragePool asymmetric pads are not supported in this build");
        }
        boolean ceil = intAttr(attrs, "ceil_mode", 0) != 0;
        boolean countIncludePad = intAttr(attrs, "count_include_pad", 1) != 0;
        return input.avgPool2d(kernel, strides, p, ceil, countIncludePad, null);
    }

    private static Tensor batchNorm(List<Tensor> in, Map<String, OnnxProto.Attribute> attrs) {
        // ONNX BatchNormalization is inference-only with fixed scale/bias/mean/var.
        Tensor input = require(in, 0, new OnnxProto.Node(List.of(), List.of(), "BatchNormalization", List.of()));
        Tensor scale = require(in, 1, new OnnxProto.Node(List.of(), List.of(), "BatchNormalization", List.of()));
        Tensor bias = require(in, 2, new OnnxProto.Node(List.of(), List.of(), "BatchNormalization", List.of()));
        Tensor mean = require(in, 3, new OnnxProto.Node(List.of(), List.of(), "BatchNormalization", List.of()));
        Tensor var = require(in, 4, new OnnxProto.Node(List.of(), List.of(), "BatchNormalization", List.of()));
        double epsilon = doubleAttr(attrs, "epsilon", 1e-5);
        // y = scale * (x - mean) / sqrt(var + epsilon) + bias, applied per channel.
        int channels = Math.toIntExact(scale.size(0));
        Tensor normalized = input.clone();
        // Apply via per-channel affine using broadcast-friendly ops: reshape to [1, C, 1, 1] for NCHW,
        // or [C] for 2-D [N, C].
        if (input.dim() == 4) {
            int[] shape = new int[input.dim()];
            shape[0] = 1;
            shape[1] = channels;
            for (int d = 2; d < input.dim(); d++) {
                shape[d] = 1;
            }
            Tensor invStd = var.add(epsilon).sqrt().reciprocal();
            Tensor mul = scale.mul(invStd).reshape(shape[0], shape[1], shape[2], shape[3]);
            // Recompute with explicit 4-D reshape helper via view.
            Tensor off = mean.mul(invStd).reshape(shape[0], shape[1], shape[2], shape[3]);
            Tensor add = bias.sub(off).reshape(shape[0], shape[1], shape[2], shape[3]);
            return input.mul(mul).add(add);
        }
        if (input.dim() == 2) {
            Tensor invStd = var.add(epsilon).sqrt().reciprocal();
            Tensor mul = scale.mul(invStd).reshape((int) scale.size(0));
            Tensor off = mean.mul(invStd).reshape((int) mean.size(0));
            Tensor add = bias.sub(off).reshape((int) bias.size(0));
            return input.mul(mul).add(add);
        }
        throw new OnnxException("ONNX BatchNormalization foundations support 2-D/4-D only, got "
                + input.shape());
    }
}
