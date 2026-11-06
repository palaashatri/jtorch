package org.jtorch.onnx;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.jtorch.api.Torch;
import org.jtorch.core.Tensor;

/** Executable ONNX model (import-only foundations; export is not implemented). */
public final class OnnxModel {
    private final OnnxProto.Model proto;
    private final Map<String, Tensor> initializers;

    OnnxModel(OnnxProto.Model proto) {
        this.proto = proto;
        this.initializers = new LinkedHashMap<>();
        for (OnnxProto.Tensor tensor : proto.graph().initializers()) {
            initializers.put(tensor.name(), OnnxTensorDecoder.decode(tensor));
        }
    }

    public long opsetVersion() {
        return proto.opsetVersion();
    }

    public List<String> inputNames() {
        List<String> fromGraph = proto.graph().inputs().stream().map(info -> info.name()).toList();
        List<String> result = new ArrayList<>();
        for (String name : fromGraph) {
            if (!initializers.containsKey(name)) {
                result.add(name);
            }
        }
        return List.copyOf(result);
    }

    public List<String> outputNames() {
        return proto.graph().outputs().stream().map(info -> info.name()).toList();
    }

    /** Runs inference with grad disabled; unsupported ops are reported before any dispatch. */
    public Map<String, Tensor> run(Map<String, Tensor> inputs) {
        OnnxRuntime.validate(proto);
        return Torch.noGrad(() -> OnnxRuntime.execute(proto.graph(), initializers, inputs));
    }

    public static OnnxModel load(Path path) {
        try {
            byte[] bytes = java.nio.file.Files.readAllBytes(path);
            return load(bytes);
        } catch (java.io.IOException error) {
            throw new OnnxException("Unable to read ONNX model " + path, error);
        }
    }

    public static OnnxModel load(byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            throw new OnnxException("ONNX model is empty");
        }
        try {
            return new OnnxModel(OnnxLoader.parse(bytes));
        } catch (OnnxException error) {
            throw error;
        } catch (RuntimeException error) {
            throw new OnnxException("Invalid ONNX model: " + error.getMessage(), error);
        }
    }

    /** Entry-point alias mirroring {@link #load(Path)}. */
    public static OnnxModel loadModel(Path path) {
        return load(path);
    }
}
