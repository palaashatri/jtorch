package org.jtorch.onnx;

import java.util.List;

/** Parsed ONNX proto subset (Model/Graph/Node/Attribute/Tensor). */
public final class OnnxProto {
    private OnnxProto() {}

    public record Model(long irVersion, long opsetVersion, Graph graph) {}

    public record Graph(String name, List<Node> nodes, List<Tensor> initializers, List<ValueInfo> inputs,
            List<ValueInfo> outputs) {}

    public record ValueInfo(String name) {}

    public record Node(List<String> inputs, List<String> outputs, String opType, List<Attribute> attributes) {}

    public record Attribute(String name, AttrType type, Double floatValue, Long intValue, List<Long> ints,
            List<Double> floats, byte[] bytes, Tensor tensorValue) {
        public double floatOr(double fallback) {
            return floatValue == null ? fallback : floatValue;
        }

        public long intOr(long fallback) {
            return intValue == null ? fallback : intValue;
        }
    }

    public enum AttrType {
        UNDEFINED,
        FLOAT,
        INT,
        STRING,
        TENSOR,
        GRAPH,
        FLOATS,
        INTS,
        STRINGS,
        TENSORS,
        GRAPHS
    }

    public record Tensor(String name, List<Long> dims, long dataType, byte[] rawData, List<Double> doubleData,
            List<Float> floatData, List<Long> int64Data, List<Integer> int32Data, long dataLocation) {
        public boolean isExternal() {
            return dataLocation == 1;
        }
    }
}
