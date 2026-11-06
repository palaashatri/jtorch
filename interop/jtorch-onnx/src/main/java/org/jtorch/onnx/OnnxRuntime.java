package org.jtorch.onnx;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.jtorch.core.Tensor;

/** Graph executor for the supported ONNX subset; unsupported operators fail before execution. */
final class OnnxRuntime {
    private OnnxRuntime() {}

    private static final Set<String> SUPPORTED = Set.of(
            "Identity", "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Exp", "Log", "Sigmoid", "Relu",
            "MatMul", "Gemm", "Softmax", "Flatten", "Transpose", "Reshape", "Concat", "Squeeze", "Unsqueeze",
            "Gather", "Conv", "MaxPool", "AveragePool", "Constant", "BatchNormalization");

    static void validate(OnnxProto.Model model) {
        long opset = model.opsetVersion();
        if (opset < 13 || opset > 23) {
            throw new OnnxException("ONNX opset " + opset + " is not supported (supported 13-23)");
        }
        List<String> unsupported = new ArrayList<>();
        for (OnnxProto.Node node : model.graph().nodes()) {
            if (!SUPPORTED.contains(node.opType())) {
                unsupported.add(node.opType());
            }
            if (node.opType().equals("Conv") || node.opType().equals("MaxPool")
                    || node.opType().equals("AveragePool")) {
                // 2-D only foundations are validated per-node at execution time.
            }
        }
        if (!unsupported.isEmpty()) {
            throw new OnnxException("ONNX unsupported operators " + unsupported + " (reported before execution)");
        }
    }

    static Map<String, Tensor> execute(OnnxProto.Graph graph, Map<String, Tensor> initializers,
            Map<String, Tensor> inputs) {
        Map<String, Tensor> env = new LinkedHashMap<>(initializers);
        for (Map.Entry<String, Tensor> entry : inputs.entrySet()) {
            if (entry.getKey() == null || entry.getKey().isEmpty()) {
                throw new OnnxException("ONNX input name must be non-empty");
            }
            env.put(entry.getKey(), entry.getValue());
        }
        for (OnnxProto.ValueInfo info : graph.inputs()) {
            if (!env.containsKey(info.name())) {
                throw new OnnxException("Missing ONNX graph input '" + info.name() + "'");
            }
        }
        for (OnnxProto.Node node : graph.nodes()) {
            OnnxOps.execute(node, env);
        }
        Map<String, Tensor> outputs = new LinkedHashMap<>();
        for (OnnxProto.ValueInfo info : graph.outputs()) {
            Tensor value = env.get(info.name());
            if (value == null) {
                throw new OnnxException("ONNX graph output '" + info.name() + "' was not produced");
            }
            outputs.put(info.name(), value);
        }
        return Map.copyOf(outputs);
    }
}
