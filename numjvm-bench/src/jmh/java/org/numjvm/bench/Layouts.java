package org.numjvm.bench;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Slice;

/**
 * Builds float64 arrays of a given element count in the five layouts the M0 benchmarks
 * compare. Every layout has the same logical shape {@code rows x cols}.
 */
public final class Layouts {

    /**
     * Element counts from the M0 specification.
     */
    static final String[] SIZES = {"16", "128", "1024", "65536", "1048576", "16777216"};

    /**
     * The layouts compared by the benchmarks.
     */
    public enum Layout {
        /**
         * Row-major, contiguous.
         */
        C,
        /**
         * Column-major, contiguous.
         */
        F,
        /**
         * A transposed view of a row-major array: strides swapped, no copy.
         */
        TRANSPOSED,
        /**
         * Every other column of a twice-as-wide array: inner stride 16 bytes.
         */
        STRIDED,
        /**
         * One row broadcast to all rows: stride 0 on the outer axis.
         */
        BROADCASTED
    }

    private Layouts() {
    }

    static long rows(long n) {
        return 1L << (63 - Long.numberOfLeadingZeros(n)) / 2;
    }

    static long cols(long n) {
        return n / rows(n);
    }

    /**
     * An array of {@code dtype} in {@code layout} holding {@code 0.5 + 4i/n}: values in
     * {@code [0.5, 4.5)} that every function benchmarked is defined on (integers get 0 to 4).
     */
    static NDArray make(long n, Layout layout, DType dtype) {
        long r = rows(n);
        long c = cols(n);
        return switch (layout) {
            case C -> values(n, dtype).reshapeView(r, c);
            case F -> values(n, dtype).reshapeView(r, c).copy(Order.F);
            case TRANSPOSED -> values(n, dtype).reshapeView(c, r).transpose();
            case STRIDED -> values(2 * n, dtype).reshapeView(r, 2 * c).slice(Slice.all(), Slice.step(2));
            case BROADCASTED -> values(c, dtype).broadcastTo(r, c);
        };
    }

    private static NDArray values(long n, DType dtype) {
        return NJ.add(NJ.multiply(NJ.arange(0.0, n, 1.0), 4.0 / n), 0.5).astype(dtype);
    }

    static NDArray make(long n, Layout layout) {
        long r = rows(n);
        long c = cols(n);
        return switch (layout) {
            case C -> NJ.arange(0.0, n, 1.0).reshapeView(r, c);
            case F -> NJ.arange(0.0, n, 1.0).reshapeView(r, c).copy(Order.F);
            case TRANSPOSED -> NJ.arange(0.0, n, 1.0).reshapeView(c, r).transpose();
            case STRIDED -> NJ.arange(0.0, 2.0 * n, 1.0).reshapeView(r, 2 * c).slice(Slice.all(), Slice.step(2));
            case BROADCASTED -> NJ.arange(0.0, c, 1.0).broadcastTo(r, c);
        };
    }
}
