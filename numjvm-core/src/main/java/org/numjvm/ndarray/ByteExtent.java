package org.numjvm.ndarray;

/**
 * Half-open range of bytes {@code [start, end)} that an array may touch in its buffer.
 *
 * <p>For an array with no elements the extent is empty ({@code start == end}) and touches
 * no memory at all.
 *
 * @param start first byte that may be read or written
 * @param end   one past the last byte that may be read or written
 */
public record ByteExtent(long start, long end) {

    /**
     * Validates the range.
     *
     * @param start first byte
     * @param end   one past the last byte; must be {@code >= start}
     */
    public ByteExtent {
        if (end < start) {
            throw new IllegalArgumentException("end " + end + " < start " + start);
        }
    }

    /**
     * Tells whether the extent touches no memory.
     *
     * @return {@code true} if {@code start == end}
     */
    public boolean isEmpty() {
        return start == end;
    }

    /**
     * Returns the number of bytes in the range.
     *
     * @return {@code end - start}
     */
    public long length() {
        return end - start;
    }

    /**
     * Tells whether two extents have at least one byte in common.
     *
     * @param other the other extent
     * @return {@code false} if either extent is empty or they are disjoint
     */
    public boolean overlaps(ByteExtent other) {
        return !isEmpty() && !other.isEmpty() && start < other.end && other.start < end;
    }
}
