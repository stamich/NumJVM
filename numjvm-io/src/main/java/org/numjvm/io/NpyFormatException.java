package org.numjvm.io;

import org.numjvm.exception.NumJVMException;

/**
 * Thrown when {@code .npy} data is malformed, truncated, exceeds a configured limit,
 * or uses a feature NumJVM does not support (structured dtypes, Python objects).
 */
public class NpyFormatException extends NumJVMException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a message.
     *
     * @param message what is wrong with the data
     */
    public NpyFormatException(String message) {
        super(message);
    }

    /**
     * Creates an exception with a message and a cause.
     *
     * @param message what is wrong with the data
     * @param cause   the underlying exception
     */
    public NpyFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
