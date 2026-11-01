package org.jtorch.core;

/** Raised when a shape, stride or dimension argument is invalid. */
public final class ShapeException extends JTorchException {
    private static final long serialVersionUID = 1L;

    public ShapeException(String message) {
        super(message);
    }
}
