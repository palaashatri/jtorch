package org.jtorch.serialization;

/** Raised when a serialized artifact is malformed, truncated, corrupted or uses an unsupported feature. */
public final class SerializationException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public SerializationException(String message) {
        super(message);
    }

    public SerializationException(String message, Throwable cause) {
        super(message, cause);
    }
}
