package org.numjvm.exception;

/**
 * Thrown when a shape, axis, stride or view is invalid: negative dimensions, too many
 * dimensions, element counts or byte offsets that overflow {@code long}, shapes that
 * cannot be broadcast or reshaped, or views that would reach outside their buffer.
 */
public class ShapeException extends NumJVMException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a message.
     *
     * @param message what went wrong
     */
    public ShapeException(String message) {
        super(message);
    }

    /**
     * Creates an exception with a message and a cause, typically an
     * {@link ArithmeticException} from overflow-checked arithmetic.
     *
     * @param message what went wrong
     * @param cause   the underlying exception
     */
    public ShapeException(String message, Throwable cause) {
        super(message, cause);
    }
}
