package org.numjvm.exception;

/**
 * Thrown when an argument has an acceptable dtype and shape but a value the operation
 * cannot accept, for example a negative exponent in an integer {@code power}, or a
 * {@code min} over an empty array (NumPy raises {@code ValueError} for both).
 */
public class ValueException extends NumJVMException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a message.
     *
     * @param message what went wrong
     */
    public ValueException(String message) {
        super(message);
    }
}
