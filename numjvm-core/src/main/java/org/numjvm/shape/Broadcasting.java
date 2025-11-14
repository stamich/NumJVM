package org.numjvm.shape;

import java.util.Arrays;

import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Strides;

/**
 * NumPy broadcasting rules ({@code ARRAY_API}).
 *
 * <p>Shapes are aligned at their last axis. Two dimensions are compatible when they are
 * equal or one of them is 1; the result takes the other one. A missing leading dimension
 * counts as 1. So {@code [8, 1, 6, 1]} and {@code [7, 1, 5]} broadcast to
 * {@code [8, 7, 6, 5]}, and {@code [0, 3]} with {@code [1, 3]} gives {@code [0, 3]}.
 */
public final class Broadcasting {

    private Broadcasting() {
    }

    /**
     * Computes the common shape of two shapes.
     *
     * @param a first shape
     * @param b second shape
     * @return the broadcast shape
     * @throws ShapeException if the shapes are incompatible
     */
    public static Shape broadcastShapes(Shape a, Shape b) {
        return broadcastShapes(new Shape[]{a, b});
    }

    /**
     * Computes the common shape of any number of shapes.
     *
     * @param shapes the shapes; none gives the scalar shape
     * @return the broadcast shape
     * @throws ShapeException if any two shapes are incompatible
     */
    public static Shape broadcastShapes(Shape... shapes) {
        int rank = 0;
        for (Shape s : shapes) {
            rank = Math.max(rank, s.rank());
        }
        long[] out = new long[rank];
        Arrays.fill(out, 1);
        int[] from = new int[rank];   // which argument set out[d], for error messages
        Arrays.fill(from, -1);
        for (int k = 0; k < shapes.length; k++) {
            long[] dims = shapes[k].dimensions();
            int lead = rank - dims.length;
            for (int i = 0; i < dims.length; i++) {
                int d = lead + i;
                long current = out[d];
                long next = dims[i];
                if (next == current || next == 1) {
                    continue;
                }
                if (current == 1) {
                    out[d] = next;
                    from[d] = k;
                    continue;
                }
                throw new ShapeException("shape mismatch: objects cannot be broadcast to a single shape. Mismatch is"
                        + " between arg " + Math.max(0, from[d]) + " with shape " + shapes[Math.max(0, from[d])]
                        + " and arg " + k + " with shape " + shapes[k] + ".");
            }
        }
        return Shape.of(out);
    }

    /**
     * Tells whether shapes can be broadcast together.
     *
     * @param shapes the shapes
     * @return {@code true} if {@link #broadcastShapes(Shape...)} would succeed
     */
    public static boolean areBroadcastable(Shape... shapes) {
        try {
            broadcastShapes(shapes);
            return true;
        } catch (ShapeException e) {
            return false;
        }
    }

    /**
     * Computes the strides that make an array of {@code sourceShape} appear to have
     * {@code targetShape}: new leading axes and stretched length-1 axes get stride 0.
     *
     * @param sourceShape   shape of the array
     * @param sourceStrides its byte strides
     * @param targetShape   the shape to broadcast to
     * @return strides of the same rank as {@code targetShape}
     * @throws ShapeException if the source cannot be broadcast to the target
     */
    public static Strides broadcastStrides(Shape sourceShape, Strides sourceStrides, Shape targetShape) {
        long[] src = sourceShape.dimensions();
        long[] srcStrides = sourceStrides.values();
        long[] dst = targetShape.dimensions();
        if (src.length > dst.length) {
            throw new ShapeException("cannot broadcast shape " + sourceShape + " to the lower-rank shape " + targetShape);
        }
        long[] out = new long[dst.length];
        int lead = dst.length - src.length;
        for (int i = 0; i < src.length; i++) {
            int d = lead + i;
            if (src[i] == dst[d]) {
                out[d] = srcStrides[i];
            } else if (src[i] == 1) {
                out[d] = 0;
            } else {
                throw new ShapeException("cannot broadcast shape " + sourceShape + " to " + targetShape
                        + ": axis " + i + " has length " + src[i] + ", target has " + dst[d]);
            }
        }
        return Strides.of(out);
    }
}
