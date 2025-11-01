package spike;

import java.lang.foreign.MemorySegment;

/**
 * S3 — float64 sum reduction.
 *
 * <p>{@link #pairwise} follows NumPy's {@code pairwise_sum}: below 8 elements a plain
 * loop, up to 128 elements eight independent accumulators, above that a recursive
 * split at a multiple of 8. This keeps the rounding error at O(log n) and lets the
 * JIT overlap the eight additions. {@link #naive} is the single-accumulator loop
 * for comparison.
 */
final class SumReduction {

    static final int BLOCK = 128;

    private SumReduction() {
    }

    static double sum(SpikeArray a) {
        if (a.size() == 0) {
            return 0.0;
        }
        if (a.isCContiguous()) {
            return pairwise(a.segment, a.offset, a.size(), 8);
        }
        Odometer it = new Odometer(a.shape, new long[][]{a.strides}, new long[]{a.offset});
        double acc = 0.0;
        do {
            acc += pairwise(a.segment, it.offset(0), it.innerSize(), it.innerStride(0));
        } while (it.advance());
        return acc;
    }

    static double pairwise(MemorySegment s, long off, long n, long stride) {
        if (n < 8) {
            double res = 0.0;
            for (long i = 0; i < n; i++) {
                res += s.get(SpikeArray.F64, off + i * stride);
            }
            return res;
        } else if (n <= BLOCK) {
            double r0 = s.get(SpikeArray.F64, off);
            double r1 = s.get(SpikeArray.F64, off + stride);
            double r2 = s.get(SpikeArray.F64, off + 2 * stride);
            double r3 = s.get(SpikeArray.F64, off + 3 * stride);
            double r4 = s.get(SpikeArray.F64, off + 4 * stride);
            double r5 = s.get(SpikeArray.F64, off + 5 * stride);
            double r6 = s.get(SpikeArray.F64, off + 6 * stride);
            double r7 = s.get(SpikeArray.F64, off + 7 * stride);
            long i;
            long limit = n - (n % 8);
            for (i = 8; i < limit; i += 8) {
                long p = off + i * stride;
                r0 += s.get(SpikeArray.F64, p);
                r1 += s.get(SpikeArray.F64, p + stride);
                r2 += s.get(SpikeArray.F64, p + 2 * stride);
                r3 += s.get(SpikeArray.F64, p + 3 * stride);
                r4 += s.get(SpikeArray.F64, p + 4 * stride);
                r5 += s.get(SpikeArray.F64, p + 5 * stride);
                r6 += s.get(SpikeArray.F64, p + 6 * stride);
                r7 += s.get(SpikeArray.F64, p + 7 * stride);
            }
            double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) {
                res += s.get(SpikeArray.F64, off + i * stride);
            }
            return res;
        } else {
            long n2 = n / 2;
            n2 -= n2 % 8;
            return pairwise(s, off, n2, stride) + pairwise(s, off + n2 * stride, n - n2, stride);
        }
    }

    static double naive(MemorySegment s, long off, long n, long stride) {
        double res = 0.0;
        for (long i = 0; i < n; i++) {
            res += s.get(SpikeArray.F64, off + i * stride);
        }
        return res;
    }
}
