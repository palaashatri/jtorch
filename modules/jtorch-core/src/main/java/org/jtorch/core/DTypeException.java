package org.jtorch.core;

/** Raised when an operation is not defined for a dtype combination. */
public final class DTypeException extends JTorchException {
    private static final long serialVersionUID = 1L;

    public DTypeException(String message) {
        super(message);
    }
}
