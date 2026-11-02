package org.jtorch.core;

/** Base class for all JTorch runtime failures. */
public class JTorchException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public JTorchException(String message) {
        super(message);
    }

    public JTorchException(String message, Throwable cause) {
        super(message, cause);
    }
}
