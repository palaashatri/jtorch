package org.jtorch.json;

/** Raised when JSON input is malformed or a value has the wrong shape for the requested accessor. */
public class JsonException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public JsonException(String message) {
        super(message);
    }

    public JsonException(String message, Throwable cause) {
        super(message, cause);
    }
}
