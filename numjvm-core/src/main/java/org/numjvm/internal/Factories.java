package org.numjvm.internal;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Strides;

/**
 * Array construction shared by {@code NJ} (automatic memory) and {@code NJScope}
 * (scoped memory). Every method takes the allocator that decides where memory comes from.
 */
public final class Factories {

    private Factories() {
    }

    private static NDArray allocate(MemoryAllocator allocator, DType dtype, Shape shape) {
        return SharedSecrets.arrayAccess().allocate(allocator, dtype, shape, Order.C);
    }

    private static MemorySegment segment(NDArray a) {
        return SharedSecrets.arrayAccess().buffer(a).segment();
    }

    /**
     * Returns a zero-filled C-contiguous array.
     *
     * @param allocator memory source
     * @param dtype     element type
     * @param shape     dimensions
     * @return a zero-filled C-contiguous array
     */
    public static NDArray zeros(MemoryAllocator allocator, DType dtype, long[] shape) {
        return allocate(allocator, dtype, Shape.of(shape));
    }

    /**
     * Returns a C-contiguous array with every element set to {@code value}.
     *
     * @param allocator memory source
     * @param dtype     element type
     * @param value     fill value, converted like {@code setDouble}
     * @param shape     dimensions
     * @return a C-contiguous array with every element set to {@code value}
     */
    public static NDArray full(MemoryAllocator allocator, DType dtype, double value, long[] shape) {
        NDArray a = allocate(allocator, dtype, Shape.of(shape));
        if (a.size() == 0 || value == 0.0 && Double.doubleToRawLongBits(value) == 0) {
            return a;
        }
        // Encode the value once as element 0, then replicate its bytes with doubling
        // copies: O(log n) bulk copies instead of n element writes.
        flat(a).setDouble(0, value);
        MemorySegment dst = segment(a);
        long item = dtype.size();
        long filled = item;
        long total = a.nbytes();
        while (filled < total) {
            long chunk = Math.min(filled, total - filled);
            MemorySegment.copy(dst, 0, dst, filled, chunk);
            filled += chunk;
        }
        return a;
    }

    /**
     * Returns a float64 array.
     *
     * @param allocator memory source
     * @param values    elements in C order
     * @param shape     dimensions, or empty for a 1-D array of {@code values.length}
     * @return a float64 array
     */
    public static NDArray of(MemoryAllocator allocator, double[] values, long[] shape) {
        NDArray a = allocate(allocator, DType.FLOAT64, shapeFor(values.length, shape));
        MemorySegment.copy(values, 0, segment(a), ValueLayout.JAVA_DOUBLE_UNALIGNED, 0, values.length);
        return a;
    }

    /**
     * Returns a float32 array.
     *
     * @param allocator memory source
     * @param values    elements in C order
     * @param shape     dimensions, or empty for 1-D
     * @return a float32 array
     */
    public static NDArray of(MemoryAllocator allocator, float[] values, long[] shape) {
        NDArray a = allocate(allocator, DType.FLOAT32, shapeFor(values.length, shape));
        MemorySegment.copy(values, 0, segment(a), ValueLayout.JAVA_FLOAT_UNALIGNED, 0, values.length);
        return a;
    }

    /**
     * Returns an int64 array.
     *
     * @param allocator memory source
     * @param values    elements in C order
     * @param shape     dimensions, or empty for 1-D
     * @return an int64 array
     */
    public static NDArray of(MemoryAllocator allocator, long[] values, long[] shape) {
        NDArray a = allocate(allocator, DType.INT64, shapeFor(values.length, shape));
        MemorySegment.copy(values, 0, segment(a), ValueLayout.JAVA_LONG_UNALIGNED, 0, values.length);
        return a;
    }

    /**
     * Returns an int32 array.
     *
     * @param allocator memory source
     * @param values    elements in C order
     * @param shape     dimensions, or empty for 1-D
     * @return an int32 array
     */
    public static NDArray of(MemoryAllocator allocator, int[] values, long[] shape) {
        NDArray a = allocate(allocator, DType.INT32, shapeFor(values.length, shape));
        MemorySegment.copy(values, 0, segment(a), ValueLayout.JAVA_INT_UNALIGNED, 0, values.length);
        return a;
    }

    /**
     * Returns a bool array.
     *
     * @param allocator memory source
     * @param values    elements in C order
     * @param shape     dimensions, or empty for 1-D
     * @return a bool array
     */
    public static NDArray of(MemoryAllocator allocator, boolean[] values, long[] shape) {
        NDArray a = allocate(allocator, DType.BOOL, shapeFor(values.length, shape));
        MemorySegment s = segment(a);
        for (int i = 0; i < values.length; i++) {
            s.set(ValueLayout.JAVA_BYTE, i, values[i] ? (byte) 1 : (byte) 0);
        }
        return a;
    }

    /**
     * Returns the array.
     *
     * @param allocator memory source
     * @param dtype     element type of the result
     * @param values    elements in C order, converted like {@code setDouble}
     * @param shape     dimensions, or empty for 1-D
     * @return the array
     */
    public static NDArray of(MemoryAllocator allocator, DType dtype, double[] values, long[] shape) {
        NDArray a = allocate(allocator, dtype, shapeFor(values.length, shape));
        NDArray flat = flat(a);
        for (int i = 0; i < values.length; i++) {
            flat.setDouble(i, values[i]);
        }
        return a;
    }

    /**
     * Returns a complex array.
     *
     * @param allocator memory source
     * @param dtype     complex64 or complex128
     * @param values    elements in C order, converted like {@code setComplex}
     * @param shape     dimensions, or empty for 1-D
     * @return the array
     * @throws org.numjvm.exception.DTypeException if {@code dtype} is not complex
     */
    public static NDArray of(MemoryAllocator allocator, DType dtype, Complex[] values, long[] shape) {
        if (!dtype.isComplex()) {
            throw new DTypeException("a Complex[] needs a complex dtype, not " + dtype.typeName());
        }
        NDArray a = allocate(allocator, dtype, shapeFor(values.length, shape));
        NDArray flat = flat(a);
        for (int i = 0; i < values.length; i++) {
            flat.setComplex(i, values[i]);
        }
        return a;
    }

    /**
     * Evenly spaced integers in {@code [start, stop)}.
     *
     * @param allocator memory source
     * @param start     first value
     * @param stop      end, exclusive
     * @param step      spacing, not zero
     * @return an int64 array
     */
    public static NDArray arange(MemoryAllocator allocator, long start, long stop, long step) {
        if (step == 0) {
            throw new ShapeException("arange step must not be zero");
        }
        long n = 0;
        if (step > 0 && stop > start) {
            n = Math.ceilDiv(Math.subtractExact(stop, start), step);
        } else if (step < 0 && stop < start) {
            n = Math.ceilDiv(Math.subtractExact(start, stop), -step);
        }
        NDArray a = allocate(allocator, DType.INT64, Shape.of(n));
        MemorySegment s = segment(a);
        long v = start;
        for (long i = 0; i < n; i++, v += step) {
            s.setAtIndex(ValueLayout.JAVA_LONG_UNALIGNED, i, v);
        }
        return a;
    }

    /**
     * Evenly spaced floating-point values in {@code [start, stop)}; element {@code i} is
     * {@code start + i * step}, and the length is {@code ceil((stop - start) / step)}, as in
     * NumPy.
     *
     * @param allocator memory source
     * @param start     first value
     * @param stop      end, exclusive
     * @param step      spacing, not zero
     * @return a float64 array
     */
    public static NDArray arange(MemoryAllocator allocator, double start, double stop, double step) {
        if (step == 0.0 || Double.isNaN(step) || Double.isNaN(start) || Double.isNaN(stop)) {
            throw new ShapeException("arange needs finite start, stop and a non-zero step");
        }
        double len = Math.ceil((stop - start) / step);
        if (Double.isInfinite(len) || len > Long.MAX_VALUE) {
            throw new ShapeException("arange length " + len + " is too large");
        }
        long n = Math.max(0, (long) len);
        NDArray a = allocate(allocator, DType.FLOAT64, Shape.of(n));
        MemorySegment s = segment(a);
        for (long i = 0; i < n; i++) {
            s.setAtIndex(ValueLayout.JAVA_DOUBLE_UNALIGNED, i, start + i * step);
        }
        return a;
    }

    /**
     * 1-D view over a freshly allocated C-contiguous array.
     */
    private static NDArray flat(NDArray a) {
        return SharedSecrets.arrayAccess().create(SharedSecrets.arrayAccess().buffer(a), a.dtype(), 0,
                Shape.of(a.size()), Strides.of(a.itemSize()), true);
    }

    private static Shape shapeFor(int length, long[] shape) {
        if (shape.length == 0) {
            return Shape.of(length);
        }
        Shape s = Shape.of(shape);
        if (s.size() != length) {
            throw new ShapeException(length + " values cannot fill shape " + s + " of size " + s.size());
        }
        return s;
    }
}
