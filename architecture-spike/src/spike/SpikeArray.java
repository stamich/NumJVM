package spike;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * S2 — minimal float64 N-dimensional array: segment, byte offset, shape, byte strides.
 *
 * <p>No validation beyond what the spike needs. The real {@code NDArray} validates the
 * byte extent once at construction (see ADR-002); this class does not.
 */
final class SpikeArray {

    static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    final MemorySegment segment;
    final long offset;
    final long[] shape;
    final long[] strides;

    SpikeArray(MemorySegment segment, long offset, long[] shape, long[] strides) {
        this.segment = segment;
        this.offset = offset;
        this.shape = shape;
        this.strides = strides;
    }

    /** C-contiguous array on the heap ({@code long[]}-backed) or in an automatic arena. */
    static SpikeArray zeros(boolean nativeMemory, long... shape) {
        long n = size(shape);
        MemorySegment s = nativeMemory
                ? Arena.ofAuto().allocate(n * 8, 64)
                : MemorySegment.ofArray(new long[(int) n]);
        return new SpikeArray(s, 0, shape.clone(), cStrides(shape));
    }

    static SpikeArray arange(long n, boolean nativeMemory) {
        SpikeArray a = zeros(nativeMemory, n);
        for (long i = 0; i < n; i++) {
            a.segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, i);
        }
        return a;
    }

    static long size(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    static long[] cStrides(long[] shape) {
        long[] s = new long[shape.length];
        long acc = 8;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = acc;
            acc *= Math.max(1, shape[i]);
        }
        return s;
    }

    int ndim() {
        return shape.length;
    }

    long size() {
        return size(shape);
    }

    // ------------------------------------------------------------------ element access

    /** Rank-specialised accessor: no varargs array, no allocation. */
    double get(long i) {
        return segment.get(F64, offset + check(i, shape[0]) * strides[0]);
    }

    double get(long i, long j) {
        return segment.get(F64, offset + check(i, shape[0]) * strides[0] + check(j, shape[1]) * strides[1]);
    }

    double get(long i, long j, long k) {
        return segment.get(F64, offset + check(i, shape[0]) * strides[0]
                + check(j, shape[1]) * strides[1] + check(k, shape[2]) * strides[2]);
    }

    /** General accessor; the varargs form allocates an array per call at the call site. */
    double getVarargs(long... idx) {
        long off = offset;
        for (int d = 0; d < idx.length; d++) {
            off += check(idx[d], shape[d]) * strides[d];
        }
        return segment.get(F64, off);
    }

    void set(long i, long j, double v) {
        segment.set(F64, offset + check(i, shape[0]) * strides[0] + check(j, shape[1]) * strides[1], v);
    }

    private static long check(long i, long n) {
        if (Long.compareUnsigned(i, n) >= 0) {
            throw new IndexOutOfBoundsException("index " + i + " out of bounds for length " + n);
        }
        return i;
    }

    // ------------------------------------------------------------------ views

    SpikeArray transpose() {
        int n = shape.length;
        long[] sh = new long[n];
        long[] st = new long[n];
        for (int d = 0; d < n; d++) {
            sh[d] = shape[n - 1 - d];
            st[d] = strides[n - 1 - d];
        }
        return new SpikeArray(segment, offset, sh, st);
    }

    SpikeArray reshape(long... newShape) {
        return new SpikeArray(segment, offset, newShape.clone(), cStrides(newShape));
    }

    /** Every {@code step}-th element along axis 0, starting at {@code start}; step may be negative. */
    SpikeArray stepAxis0(long start, long count, long step) {
        long[] sh = shape.clone();
        long[] st = strides.clone();
        sh[0] = count;
        st[0] = strides[0] * step;
        return new SpikeArray(segment, offset + start * strides[0], sh, st);
    }

    /** Broadcast to {@code target}: right-aligned, size-1 or missing axes get stride 0. */
    SpikeArray broadcastTo(long... target) {
        long[] st = new long[target.length];
        int lead = target.length - shape.length;
        for (int d = 0; d < target.length; d++) {
            int src = d - lead;
            if (src < 0) {
                st[d] = 0;
            } else if (shape[src] == target[d]) {
                st[d] = strides[src];
            } else if (shape[src] == 1) {
                st[d] = 0;
            } else {
                throw new IllegalArgumentException("cannot broadcast axis " + src);
            }
        }
        return new SpikeArray(segment, offset, target.clone(), st);
    }

    boolean isCContiguous() {
        long expect = 8;
        for (int d = shape.length - 1; d >= 0; d--) {
            if (shape[d] != 1 && strides[d] != expect) {
                return false;
            }
            expect *= shape[d];
        }
        return true;
    }

    /** Values in logical C order, via the odometer. */
    double[] toArray() {
        double[] out = new double[(int) size()];
        if (out.length == 0) {
            return out;
        }
        Odometer it = new Odometer(shape, new long[][]{strides}, new long[]{offset});
        int k = 0;
        do {
            long p = it.offset(0);
            long s = it.innerStride(0);
            for (long i = 0, n = it.innerSize(); i < n; i++, p += s) {
                out[k++] = segment.get(F64, p);
            }
        } while (it.advance());
        return out;
    }
}
