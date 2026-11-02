package org.jtorch.core;

/** Raised when a requested device, dtype, layout or operator has no implementation in this build. */
public final class NotSupportedException extends JTorchException {
    private static final long serialVersionUID = 1L;

    public NotSupportedException(String message) {
        super(message);
    }
}
