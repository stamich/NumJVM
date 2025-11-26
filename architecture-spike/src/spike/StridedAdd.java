package spike;

import java.lang.foreign.MemorySegment;

/**
 * S3 — elementwise {@code out = a + b} with broadcasting.
 *
 * <p>Two paths: a contiguous fast path (all three operands C-contiguous with the same
 * shape, a single counted loop) and a generic path driven by {@link Odometer} that
 * handles any strides, including zero (broadcast) and negative strides.
 */
final class StridedAdd {

    private StridedAdd() {
    }

    /**
     * Broadcasting add; allocates a C-contiguous result.
     */
    static SpikeArray add(SpikeArray a, SpikeArray b, boolean nativeOut) {
        long[] shape = broadcastShape(a.shape, b.shape);
        SpikeArray out = SpikeArray.zeros(nativeOut, shape);
        if (out.size() == 0) {
            return out;
        }
        if (a.isCContiguous() && b.isCContiguous()
                && java.util.Arrays.equals(a.shape, shape) && java.util.Arrays.equals(b.shape, shape)) {
            addContiguous(a.segment, a.offset, b.segment, b.offset, out.segment, out.size());
        } else {
            addStrided(a.broadcastTo(shape), b.broadcastTo(shape), out);
        }
        return out;
    }

    /**
     * Forces the generic path, for comparing the two on identical inputs.
     */
    static SpikeArray addGeneric(SpikeArray a, SpikeArray b, boolean nativeOut) {
        long[] shape = broadcastShape(a.shape, b.shape);
        SpikeArray out = SpikeArray.zeros(nativeOut, shape);
        if (out.size() > 0) {
            addStrided(a.broadcastTo(shape), b.broadcastTo(shape), out);
        }
        return out;
    }

    static void addContiguous(MemorySegment a, long oa, MemorySegment b, long ob, MemorySegment o, long n) {
        for (long i = 0; i < n; i++) {
            long k = i << 3;
            o.set(SpikeArray.F64, k, a.get(SpikeArray.F64, oa + k) + b.get(SpikeArray.F64, ob + k));
        }
    }

    static void addStrided(SpikeArray a, SpikeArray b, SpikeArray out) {
        Odometer it = new Odometer(out.shape,
                new long[][]{a.strides, b.strides, out.strides},
                new long[]{a.offset, b.offset, out.offset});
        MemorySegment sa = a.segment;
        MemorySegment sb = b.segment;
        MemorySegment so = out.segment;
        do {
            long pa = it.offset(0);
            long pb = it.offset(1);
            long po = it.offset(2);
            long da = it.innerStride(0);
            long db = it.innerStride(1);
            long dout = it.innerStride(2);
            for (long k = 0, n = it.innerSize(); k < n; k++, pa += da, pb += db, po += dout) {
                so.set(SpikeArray.F64, po, sa.get(SpikeArray.F64, pa) + sb.get(SpikeArray.F64, pb));
            }
        } while (it.advance());
    }

    static long[] broadcastShape(long[] a, long[] b) {
        int n = Math.max(a.length, b.length);
        long[] out = new long[n];
        for (int d = 0; d < n; d++) {
            int ia = d - (n - a.length);
            int ib = d - (n - b.length);
            long da = ia < 0 ? 1 : a[ia];
            long db = ib < 0 ? 1 : b[ib];
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException("shapes not broadcastable");
            }
            out[d] = da == 1 ? db : da;
        }
        return out;
    }
}
