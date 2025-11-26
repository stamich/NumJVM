package org.numjvm.ndarray;

/**
 * One component of a basic NumPy index, used with {@link NDArray#slice(Slice...)}.
 *
 * <table>
 *   <caption>NumPy equivalents</caption>
 *   <tr><th>NumPy</th><th>NumJVM</th></tr>
 *   <tr><td>{@code a[2]}</td><td>{@code a.slice(Slice.index(2))}</td></tr>
 *   <tr><td>{@code a[-1]}</td><td>{@code a.slice(Slice.index(-1))}</td></tr>
 *   <tr><td>{@code a[1:5]}</td><td>{@code a.slice(Slice.range(1, 5))}</td></tr>
 *   <tr><td>{@code a[1:5:2]}</td><td>{@code a.slice(Slice.range(1, 5, 2))}</td></tr>
 *   <tr><td>{@code a[3:]}</td><td>{@code a.slice(Slice.from(3))}</td></tr>
 *   <tr><td>{@code a[:3]}</td><td>{@code a.slice(Slice.to(3))}</td></tr>
 *   <tr><td>{@code a[::-1]}</td><td>{@code a.slice(Slice.step(-1))}</td></tr>
 *   <tr><td>{@code a[:, 0]}</td><td>{@code a.slice(Slice.all(), Slice.index(0))}</td></tr>
 *   <tr><td>{@code a[None]}</td><td>{@code a.slice(Slice.newAxis())}</td></tr>
 *   <tr><td>{@code a[..., 0]}</td><td>{@code a.slice(Slice.ellipsis(), Slice.index(0))}</td></tr>
 * </table>
 *
 * <p>Bounds follow Python: negative positions count from the end, and range bounds
 * outside the axis are clipped rather than rejected. Single indices outside the axis are
 * an error.
 */
public sealed interface Slice permits Slice.All, Slice.Index, Slice.Range, Slice.NewAxis, Slice.Ellipsis {

    /**
     * The whole axis, {@code :}.
     */
    record All() implements Slice {
        @Override
        public String toString() {
            return ":";
        }
    }

    /**
     * A single position; the axis is removed from the result.
     *
     * @param index the position; negative values count from the end
     */
    record Index(long index) implements Slice {
        @Override
        public String toString() {
            return Long.toString(index);
        }
    }

    /**
     * {@code start:stop:step}; {@code null} bounds take Python's defaults, which depend on
     * the sign of the step.
     *
     * @param start first position, or {@code null}
     * @param stop  end position (exclusive), or {@code null}
     * @param step  stride in elements, not zero
     */
    record Range(Long start, Long stop, long step) implements Slice {
        /**
         * Validates the step.
         *
         * @param start first position, or {@code null}
         * @param stop  end position, or {@code null}
         * @param step  stride, not zero
         */
        public Range {
            if (step == 0) {
                throw new IllegalArgumentException("slice step cannot be zero");
            }
        }

        @Override
        public String toString() {
            return (start == null ? "" : start) + ":" + (stop == null ? "" : stop) + (step == 1 ? "" : ":" + step);
        }
    }

    /**
     * A new axis of length 1, NumPy's {@code None} / {@code np.newaxis}.
     */
    record NewAxis() implements Slice {
        @Override
        public String toString() {
            return "None";
        }
    }

    /**
     * As many full axes as needed to index every axis, {@code ...}. At most one per index.
     */
    record Ellipsis() implements Slice {
        @Override
        public String toString() {
            return "...";
        }
    }

    /**
     * Returns {@code :}.
     *
     * @return {@code :}
     */
    static Slice all() {
        return new All();
    }

    /**
     * Returns the single-position index.
     *
     * @param index position; negative values count from the end
     * @return the single-position index
     */
    static Slice index(long index) {
        return new Index(index);
    }

    /**
     * Returns {@code start:stop}.
     *
     * @param start first position
     * @param stop  end position, exclusive
     * @return {@code start:stop}
     */
    static Slice range(long start, long stop) {
        return new Range(start, stop, 1);
    }

    /**
     * Returns {@code start:stop:step}.
     *
     * @param start first position
     * @param stop  end position, exclusive
     * @param step  stride, positive or negative, not zero
     * @return {@code start:stop:step}
     */
    static Slice range(long start, long stop, long step) {
        return new Range(start, stop, step);
    }

    /**
     * Returns {@code start:}.
     *
     * @param start first position
     * @return {@code start:}
     */
    static Slice from(long start) {
        return new Range(start, null, 1);
    }

    /**
     * Returns {@code :stop}.
     *
     * @param stop end position, exclusive
     * @return {@code :stop}
     */
    static Slice to(long stop) {
        return new Range(null, stop, 1);
    }

    /**
     * Returns {@code ::step}.
     *
     * @param step stride, not zero; {@code -1} reverses the axis
     * @return {@code ::step}
     */
    static Slice step(long step) {
        return new Range(null, null, step);
    }

    /**
     * Returns a new axis of length 1.
     *
     * @return a new axis of length 1
     */
    static Slice newAxis() {
        return new NewAxis();
    }

    /**
     * Returns {@code ...}.
     *
     * @return {@code ...}
     */
    static Slice ellipsis() {
        return new Ellipsis();
    }
}
