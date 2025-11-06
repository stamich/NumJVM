package org.numjvm.exception;

/**
 * Thrown on an attempt to write through an array that is not writeable, such as a
 * broadcast view, whose zero strides would make one write change many logical
 * elements.
 */
public class ReadOnlyArrayException extends NumJVMException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a message.
     *
     * @param message what went wrong
     */
    public ReadOnlyArrayException(String message) {
        super(message);
    }
}
