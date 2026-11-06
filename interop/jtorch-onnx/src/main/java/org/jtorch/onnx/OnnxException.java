package org.jtorch.onnx;

/** ONNX import/execution failure with explicit unsupported reporting. */
public final class OnnxException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public OnnxException(String message) {
        super(message);
    }

    public OnnxException(String message, Throwable cause) {
        super(message, cause);
    }
}
