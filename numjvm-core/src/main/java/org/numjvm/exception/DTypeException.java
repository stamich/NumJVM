package org.numjvm.exception;

/**
 * Thrown when a dtype is unknown, unsupported for an operation, or incompatible with
 * the requested conversion.
 */
public class DTypeException extends NumJVMException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a message.
     *
     * @param message what went wrong
     */
    public DTypeException(String message) {
        super(message);
    }
}
