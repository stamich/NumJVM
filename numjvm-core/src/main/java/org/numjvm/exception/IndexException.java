package org.numjvm.exception;

/**
 * Thrown when an element index or a slice position is outside an axis, or when the
 * number of indices does not match the array's rank.
 */
public class IndexException extends NumJVMException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates an exception with a message.
     *
     * @param message what went wrong
     */
    public IndexException(String message) {
        super(message);
    }

    /**
     * Creates the exception for an index outside {@code [0, length)}.
     *
     * @param index  the offending index
     * @param axis   the axis it was used on
     * @param length the length of that axis
     * @return the exception, ready to throw
     */
    public static IndexException outOfBounds(long index, int axis, long length) {
        return new IndexException("index " + index + " is out of bounds for axis " + axis + " with size " + length);
    }
}
