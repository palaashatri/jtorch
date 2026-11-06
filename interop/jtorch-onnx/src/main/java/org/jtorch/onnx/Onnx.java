package org.jtorch.onnx;

import java.nio.file.Path;

/** ONNX import adapter (import-only foundations; export is not implemented). */
public final class Onnx {
    private Onnx() {}

    public static OnnxModel importModel(Path path) {
        return OnnxModel.load(path);
    }

    public static OnnxModel importModel(byte[] bytes) {
        return OnnxModel.load(bytes);
    }
}
