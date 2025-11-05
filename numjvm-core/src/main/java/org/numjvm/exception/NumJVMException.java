package org.numjvm.exception;

/**
 * Root of all exceptions thrown by NumJVM.
 *
 * <p>All NumJVM exceptions are unchecked: they signal programming errors (an invalid
 * shape, an unsupported dtype) or a misuse of memory lifetimes (accessing an array
 * after its scope was closed), not recoverable conditions.
 */
public class NumJVMException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a message.
     *
     * @param message what went wrong, in terms of the caller's arguments
     */
    public NumJVMException(String message) {
        super(message);
    }

    /**
     * Creates an exception with a message and a cause.
     *
     * @param message what went wrong
     * @param cause   the underlying exception
     */
    public NumJVMException(String message, Throwable cause) {
        super(message, cause);
    }
}
