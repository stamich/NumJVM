package org.numjvm.iteration;

import java.util.NoSuchElementException;

import org.numjvm.Experimental;
import org.numjvm.ndarray.Shape;

/**
 * Visits every index of a shape in C order: {@code [0,0], [0,1], [0,2], [1,0], ...}.
 *
 * <p>A reference implementation for tests and for code where clarity matters more than
 * speed; kernels use {@link StridedIterator}. To avoid an allocation per element,
 * {@link #next()} returns the same array every time, updated in place. A 0-D shape has
 * one index (the empty one); a shape with a zero-length axis has none.
 */
@Experimental
public final class IndexIterator {

    private final long[] dims;
    private final long[] index;
    private long remaining;
    private boolean started;

    /**
     * Creates an iterator positioned before the first index.
     *
     * @param shape the shape to traverse
     */
    public IndexIterator(Shape shape) {
        this.dims = shape.dimensions();
        this.index = new long[dims.length];
        this.remaining = shape.size();
    }

    /**
     * Returns {@code true} if another index follows.
     *
     * @return {@code true} if another index follows
     */
    public boolean hasNext() {
        return remaining > 0;
    }

    /**
     * Advances to the next index.
     *
     * @return the index array, shared between calls; do not modify it
     * @throws NoSuchElementException after the last index
     */
    public long[] next() {
        if (remaining <= 0) {
            throw new NoSuchElementException();
        }
        if (started) {
            for (int d = dims.length - 1; d >= 0; d--) {
                if (++index[d] < dims[d]) {
                    break;
                }
                index[d] = 0;
            }
        }
        started = true;
        remaining--;
        return index;
    }
}
