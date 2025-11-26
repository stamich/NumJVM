package spike;

/**
 * S2 — allocation-free multi-operand strided iterator ("external inner loop").
 *
 * <p>The iterator walks the outer axes like an odometer and hands the innermost axis
 * to the caller as a (offset, stride, count) triple per operand. The caller runs a
 * plain counted loop over it. Per element there is no call, no allocation and no
 * virtual dispatch; per row there is one {@link #advance()} call that updates the
 * operand offsets incrementally.
 *
 * <pre>{@code
 * Odometer it = new Odometer(shape, new long[][]{sa, sb}, new long[]{oa, ob});
 * if (size > 0) do {
 *     long pa = it.offset(0), pb = it.offset(1);
 *     long da = it.innerStride(0), db = it.innerStride(1);
 *     for (long k = 0, n = it.innerSize(); k < n; k++, pa += da, pb += db) { ... }
 * } while (it.advance());
 * }</pre>
 *
 * <p>All operands must already be broadcast to {@code shape} (zero strides on
 * broadcast axes). A 0-D shape yields exactly one inner run of length 1.
 */
final class Odometer {

    private final int outerRank;
    private final int nops;
    private final long[] shape;
    private final long[][] strides;
    private final long[] counter;
    private final long[] offsets;
    private final long innerSize;
    private final long[] innerStrides;

    Odometer(long[] shape, long[][] strides, long[] baseOffsets) {
        int rank = shape.length;
        this.nops = strides.length;
        this.shape = shape;
        this.strides = strides;
        this.outerRank = Math.max(0, rank - 1);
        this.counter = new long[outerRank];
        this.offsets = baseOffsets.clone();
        this.innerSize = rank == 0 ? 1 : shape[rank - 1];
        this.innerStrides = new long[nops];
        for (int op = 0; op < nops; op++) {
            innerStrides[op] = rank == 0 ? 0 : strides[op][rank - 1];
        }
    }

    long innerSize() {
        return innerSize;
    }

    long innerStride(int op) {
        return innerStrides[op];
    }

    long offset(int op) {
        return offsets[op];
    }

    /**
     * Moves to the next inner run; returns {@code false} after the last one.
     */
    boolean advance() {
        for (int d = outerRank - 1; d >= 0; d--) {
            long c = counter[d] + 1;
            if (c < shape[d]) {
                counter[d] = c;
                for (int op = 0; op < nops; op++) {
                    offsets[op] += strides[op][d];
                }
                return true;
            }
            counter[d] = 0;
            for (int op = 0; op < nops; op++) {
                offsets[op] -= (shape[d] - 1) * strides[op][d];
            }
        }
        return false;
    }
}
