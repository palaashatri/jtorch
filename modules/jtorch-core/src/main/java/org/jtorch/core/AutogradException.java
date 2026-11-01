package org.jtorch.core;

/** Raised for autograd misuse: backward through a detached graph, illegal in-place mutation, or version mismatch. */
public final class AutogradException extends JTorchException {
    private static final long serialVersionUID = 1L;

    public AutogradException(String message) {
        super(message);
    }
}
