package org.jtorch.onnx;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.jtorch.onnx.protobuf.ProtoReader;

/** Decodes the ONNX ModelProto wire format without external protobuf bindings. */
final class OnnxLoader {
    private OnnxLoader() {}

    static OnnxProto.Model parse(byte[] bytes) {
        ProtoReader reader = new ProtoReader(bytes);
        long irVersion = 0;
        long opsetVersion = -1;
        OnnxProto.Graph graph = null;
        boolean seenDomain = false;
        while (!reader.exhausted()) {
            ProtoReader.Field field = reader.nextField();
            switch (field.number()) {
                case 1 -> {
                    if (field.wireType() != 0) {
                        reader.skipField(field.wireType());
                    } else {
                        irVersion = reader.readInt64();
                    }
                }
                case 8 -> {
                    ProtoReader sub = reader.readMessage();
                    long version = -1;
                    String domain = "";
                    while (!sub.exhausted()) {
                        ProtoReader.Field f = sub.nextField();
                        if (f.number() == 1 && f.wireType() == 2) {
                            domain = sub.readString();
                        } else if (f.number() == 2 && f.wireType() == 0) {
                            version = sub.readInt64();
                        } else {
                            sub.skipField(f.wireType());
                        }
                    }
                    if ((domain == null || domain.isEmpty()) && !seenDomain) {
                        opsetVersion = version;
                        seenDomain = true;
                    } else if (domain != null && domain.isEmpty() && opsetVersion < 0) {
                        opsetVersion = version;
                    } else if (opsetVersion < 0) {
                        opsetVersion = version;
                    }
                }
                case 7 -> graph = parseGraph(reader.readMessage());
                default -> reader.skipField(field.wireType());
            }
        }
        if (graph == null) {
            throw new OnnxException("ONNX ModelProto has no graph");
        }
        return new OnnxProto.Model(irVersion, opsetVersion, graph);
    }

    private static OnnxProto.Graph parseGraph(ProtoReader reader) {
        String name = "";
        List<OnnxProto.Node> nodes = new ArrayList<>();
        List<OnnxProto.Tensor> initializers = new ArrayList<>();
        List<OnnxProto.ValueInfo> inputs = new ArrayList<>();
        List<OnnxProto.ValueInfo> outputs = new ArrayList<>();
        while (!reader.exhausted()) {
            ProtoReader.Field field = reader.nextField();
            switch (field.number()) {
                case 1 -> nodes.add(parseNode(reader.readMessage()));
                case 2 -> name = reader.readString();
                case 5 -> initializers.add(parseTensor(reader.readMessage()));
                case 11 -> inputs.add(parseValueInfo(reader.readMessage()));
                case 12 -> outputs.add(parseValueInfo(reader.readMessage()));
                default -> reader.skipField(field.wireType());
            }
        }
        return new OnnxProto.Graph(name, List.copyOf(nodes), List.copyOf(initializers), List.copyOf(inputs),
                List.copyOf(outputs));
    }

    private static OnnxProto.ValueInfo parseValueInfo(ProtoReader reader) {
        String name = "";
        while (!reader.exhausted()) {
            ProtoReader.Field field = reader.nextField();
            if (field.number() == 1 && field.wireType() == 2) {
                name = reader.readString();
            } else {
                reader.skipField(field.wireType());
            }
        }
        return new OnnxProto.ValueInfo(name);
    }

    private static OnnxProto.Node parseNode(ProtoReader reader) {
        List<String> inputs = new ArrayList<>();
        List<String> outputs = new ArrayList<>();
        String opType = "";
        List<OnnxProto.Attribute> attributes = new ArrayList<>();
        while (!reader.exhausted()) {
            ProtoReader.Field field = reader.nextField();
            switch (field.number()) {
                case 1 -> inputs.add(reader.readString());
                case 2 -> outputs.add(reader.readString());
                case 4 -> opType = reader.readString();
                case 5 -> attributes.add(parseAttribute(reader.readMessage()));
                default -> reader.skipField(field.wireType());
            }
        }
        return new OnnxProto.Node(List.copyOf(inputs), List.copyOf(outputs), opType, List.copyOf(attributes));
    }

    private static OnnxProto.Attribute parseAttribute(ProtoReader reader) {
        String name = "";
        OnnxProto.AttrType type = OnnxProto.AttrType.UNDEFINED;
        Double floatValue = null;
        Long intValue = null;
        List<Long> ints = new ArrayList<>();
        List<Double> floats = new ArrayList<>();
        byte[] bytes = null;
        OnnxProto.Tensor tensorValue = null;
        while (!reader.exhausted()) {
            ProtoReader.Field field = reader.nextField();
            switch (field.number()) {
                case 1 -> name = reader.readString();
                case 20 -> type = attrType(reader.readInt32());
                case 2 -> floatValue = (double) reader.readFloat();
                case 3 -> intValue = reader.readInt64();
                case 4 -> bytes = reader.readBytes();
                case 5 -> tensorValue = parseTensor(reader.readMessage());
                case 7 -> {
                    if (field.wireType() == 2) {
                        for (Float value : reader.readPackedFloat()) {
                            floats.add((double) value);
                        }
                    } else {
                        floats.add((double) reader.readFloat());
                    }
                }
                case 9 -> {
                    if (field.wireType() == 2) {
                        ints.addAll(reader.readPackedInt64());
                    } else {
                        ints.add(reader.readInt64());
                    }
                }
                case 8 -> {
                    if (field.wireType() == 2) {
                        ints.addAll(reader.readPackedInt64());
                    } else {
                        ints.add(reader.readInt64());
                    }
                }
                default -> reader.skipField(field.wireType());
            }
        }
        return new OnnxProto.Attribute(name, type, floatValue, intValue, List.copyOf(ints), List.copyOf(floats),
                bytes, tensorValue);
    }

    private static OnnxProto.AttrType attrType(int value) {
        return switch (value) {
            case 1 -> OnnxProto.AttrType.FLOAT;
            case 2 -> OnnxProto.AttrType.INT;
            case 3 -> OnnxProto.AttrType.STRING;
            case 4 -> OnnxProto.AttrType.TENSOR;
            case 5 -> OnnxProto.AttrType.GRAPH;
            case 6 -> OnnxProto.AttrType.FLOATS;
            case 7 -> OnnxProto.AttrType.INTS;
            case 8 -> OnnxProto.AttrType.STRINGS;
            case 9 -> OnnxProto.AttrType.TENSORS;
            case 10 -> OnnxProto.AttrType.GRAPHS;
            default -> OnnxProto.AttrType.UNDEFINED;
        };
    }

    private static OnnxProto.Tensor parseTensor(ProtoReader reader) {
        List<Long> dims = new ArrayList<>();
        long dataType = 0;
        byte[] rawData = new byte[0];
        List<Double> doubleData = new ArrayList<>();
        List<Float> floatData = new ArrayList<>();
        List<Long> int64Data = new ArrayList<>();
        List<Integer> int32Data = new ArrayList<>();
        String name = "";
        long dataLocation = 0;
        boolean hasRaw = false;
        while (!reader.exhausted()) {
            ProtoReader.Field field = reader.nextField();
            switch (field.number()) {
                case 1 -> {
                    if (field.wireType() == 2) {
                        dims.addAll(reader.readPackedInt64());
                    } else {
                        dims.add(reader.readInt64());
                    }
                }
                case 2 -> dataType = reader.readInt64();
                case 4 -> {
                    if (field.wireType() == 2) {
                        floatData.addAll(reader.readPackedFloat());
                    } else {
                        floatData.add(reader.readFloat());
                    }
                }
                case 5 -> {
                    if (field.wireType() == 2) {
                        // Packed int32 is varint-encoded sequence.
                        List<Long> packed = reader.readPackedInt64();
                        for (Long v : packed) {
                            int32Data.add(v.intValue());
                        }
                    } else {
                        int32Data.add((int) reader.readInt64());
                    }
                }
                case 7 -> {
                    if (field.wireType() == 2) {
                        int64Data.addAll(reader.readPackedInt64());
                    } else {
                        int64Data.add(reader.readInt64());
                    }
                }
                case 8 -> name = reader.readString();
                case 9 -> {
                    rawData = reader.readBytes();
                    hasRaw = true;
                }
                case 10 -> {
                    if (field.wireType() == 2) {
                        doubleData.addAll(reader.readPackedDouble());
                    } else {
                        doubleData.add(reader.readDouble());
                    }
                }
                case 11 -> {
                    if (field.wireType() == 2) {
                        List<Long> packed = reader.readPackedInt64();
                        for (Long v : packed) {
                            int64Data.add(v);
                        }
                    } else {
                        int64Data.add(reader.readInt64());
                    }
                }
                case 14 -> dataLocation = reader.readInt64();
                default -> reader.skipField(field.wireType());
            }
        }
        // Preserve whether raw data was present even if empty (raw_data may be zero-length for empty tensors).
        if (!hasRaw) {
            rawData = null;
        }
        return new OnnxProto.Tensor(name, List.copyOf(dims), dataType, rawData, List.copyOf(doubleData),
                List.copyOf(floatData), List.copyOf(int64Data), List.copyOf(int32Data), dataLocation);
    }

    /** Helper for debugging. */
    static String utf8(byte[] bytes) {
        return bytes == null ? "" : new String(bytes, StandardCharsets.UTF_8);
    }
}
