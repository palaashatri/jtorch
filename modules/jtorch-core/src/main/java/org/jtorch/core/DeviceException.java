package org.jtorch.core;

/** Raised when a device is unavailable or an operation is invalid on a device. */
public final class DeviceException extends JTorchException {
    private static final long serialVersionUID = 1L;

    public DeviceException(String message) {
        super(message);
    }
}
