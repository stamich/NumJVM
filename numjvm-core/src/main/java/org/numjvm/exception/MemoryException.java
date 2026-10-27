package org.numjvm.exception;

/**
 * Thrown when array memory cannot be allocated or is no longer accessible: the
 * {@code NJScope} that owns it was closed, it is accessed from a thread that the
 * scope does not allow, or a requested size cannot be allocated.
 */
public class MemoryException extends NumJVMException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a message.
     *
     * @param message what went wrong
     */
    public MemoryException(String message) {
        super(message);
    }

    /**
     * Creates an exception with a message and a cause, typically the
     * {@link IllegalStateException} or {@link WrongThreadException} raised by the
     * Foreign Function &amp; Memory API.
     *
     * @param message what went wrong
     * @param cause   the underlying exception
     */
    public MemoryException(String message, Throwable cause) {
        super(message, cause);
    }
}
