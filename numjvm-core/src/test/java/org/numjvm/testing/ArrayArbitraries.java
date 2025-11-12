package org.numjvm.testing;

import java.util.Arrays;

import net.jqwik.api.Arbitraries;
import net.jqwik.api.Arbitrary;
import net.jqwik.api.Combinators;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Strides;

/**
 * jqwik generators for shapes, strides and byte offsets.
 *
 * <p>Shapes deliberately over-sample the edge cases that break array code: rank 0,
 * dimensions of length 0 and 1. Strides include negative and zero values.
 */
public final class ArrayArbitraries {

    private ArrayArbitraries() {
    }

    /** One array layout: where element [0, ..., 0] is and how to step from it. */
    public record Layout(Shape shape, Strides strides, long offset, int itemSize) {
        @Override
        public String toString() {
            return "Layout[shape=" + shape + ", strides=" + strides + ", offset=" + offset
                    + ", itemSize=" + itemSize + "]";
        }
    }

    /** Dimension lengths: 0 and 1 are frequent, larger values up to {@code maxDim}. */
    public static Arbitrary<Long> dimensions(long maxDim) {
        return Arbitraries.frequencyOf(
                net.jqwik.api.Tuple.of(1, Arbitraries.just(0L)),
                net.jqwik.api.Tuple.of(3, Arbitraries.just(1L)),
                net.jqwik.api.Tuple.of(8, Arbitraries.longs().between(2, maxDim)));
    }

    /** Shapes of rank 0 to {@code maxRank}. */
    public static Arbitrary<Shape> shapes(int maxRank, long maxDim) {
        return dimensions(maxDim).list().ofMinSize(0).ofMaxSize(maxRank)
                .map(l -> Shape.of(l.stream().mapToLong(Long::longValue).toArray()));
    }

    /** Small shapes (rank ≤ 4, dimensions ≤ 6), cheap enough to enumerate element by element. */
    public static Arbitrary<Shape> smallShapes() {
        return shapes(4, 6);
    }

    /** Shapes with at least one element. */
    public static Arbitrary<Shape> nonEmptyShapes(int maxRank, long maxDim) {
        return shapes(maxRank, maxDim).filter(s -> s.size() > 0);
    }

    /** Element sizes of the existing dtypes. */
    public static Arbitrary<Integer> itemSizes() {
        return Arbitraries.of(1, 2, 4, 8, 16);
    }

    /** Every dtype. */
    public static Arbitrary<DType> dtypes() {
        return Arbitraries.of(DType.class);
    }

    /**
     * Arbitrary strided layouts over small shapes: each stride is a multiple of the item
     * size in {@code [-4, 4]}, and the offset is chosen so the lowest touched byte is
     * {@code >= 0}, plus some slack.
     */
    public static Arbitrary<Layout> layouts() {
        Arbitrary<Shape> shapes = smallShapes();
        return Combinators.combine(shapes, itemSizes(), Arbitraries.longs().between(0, 64))
                .flatAs((shape, item, slack) -> Arbitraries.integers().between(-4, 4).list()
                        .ofSize(shape.rank())
                        .map(mult -> {
                            long[] st = new long[shape.rank()];
                            long negativeSpan = 0;
                            for (int d = 0; d < st.length; d++) {
                                st[d] = (long) mult.get(d) * item;
                                if (st[d] < 0 && shape.dimension(d) > 0) {
                                    negativeSpan += (shape.dimension(d) - 1) * -st[d];
                                }
                            }
                            return new Layout(shape, Strides.of(st), negativeSpan + slack * item, item);
                        }));
    }

    /**
     * C-contiguous float64 arrays over small shapes whose elements are 0, 1, 2, ... in C
     * order, so every element is distinguishable.
     */
    public static Arbitrary<NDArray> arrays() {
        return smallShapes().map(ArrayArbitraries::sequential);
    }

    /** A float64 array of the given shape holding 0, 1, 2, ... in C order. */
    public static NDArray sequential(Shape shape) {
        double[] values = new double[Math.toIntExact(shape.size())];
        for (int i = 0; i < values.length; i++) {
            values[i] = i;
        }
        return shape.rank() == 0 ? NJ.scalar(0.0) : NJ.array(values, shape.dimensions());
    }

    /** Values of an array in logical C order, read through its own strides. */
    public static double[] values(NDArray a) {
        double[] out = new double[Math.toIntExact(a.size())];
        long[] dims = a.shape().dimensions();
        long[] idx = new long[dims.length];
        for (int k = 0; k < out.length; k++) {
            out[k] = a.getDouble(idx);
            for (int d = dims.length - 1; d >= 0; d--) {
                if (++idx[d] < dims[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
        return out;
    }

    /** Element offsets of a layout in C order, by enumerating every index. */
    public static long[] enumerateOffsets(Layout layout) {
        long[] dims = layout.shape().dimensions();
        long[] st = layout.strides().values();
        int n = Math.toIntExact(layout.shape().size());
        long[] out = new long[n];
        long[] idx = new long[dims.length];
        for (int k = 0; k < n; k++) {
            long off = layout.offset();
            for (int d = 0; d < dims.length; d++) {
                off += idx[d] * st[d];
            }
            out[k] = off;
            for (int d = dims.length - 1; d >= 0; d--) {
                if (++idx[d] < dims[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
        return out;
    }

    /** Debug helper for property failure messages. */
    public static String describe(long[] values) {
        return Arrays.toString(values);
    }
}
