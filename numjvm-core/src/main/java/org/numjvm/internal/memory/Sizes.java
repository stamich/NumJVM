package org.numjvm.internal.memory;

import org.numjvm.exception.ShapeException;

/**
 * Overflow-checked size arithmetic for allocations.
 */
public final class Sizes {

    private Sizes() {
    }

    /**
     * Returns {@code elements * itemSize}.
     *
     * @param elements number of elements, {@code >= 0}
     * @param itemSize bytes per element, {@code > 0}
     * @return the byte size
     * @throws ShapeException if the product overflows {@code long}
     */
    public static long bytesFor(long elements, int itemSize) {
        if (elements < 0 || itemSize <= 0) {
            throw new ShapeException("invalid size: " + elements + " elements of " + itemSize + " bytes");
        }
        try {
            return Math.multiplyExact(elements, (long) itemSize);
        } catch (ArithmeticException e) {
            throw new ShapeException(elements + " elements of " + itemSize + " bytes overflow long", e);
        }
    }
}
