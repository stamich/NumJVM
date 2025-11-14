package org.numjvm.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.Experimental;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.internal.memory.MemoryErrors;
import org.numjvm.iteration.StridedIterator;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;

/**
 * Two float64 kernels that exercise memory, views, broadcasting and iteration together.
 *
 * <p>They exist to validate the M0 architecture, not as the arithmetic API: the
 * elementwise engine with dtype promotion, ufuncs and reductions arrives in M1 and will
 * replace this class. Both follow ADR-003: a contiguous fast path plus a generic loop
 * over {@link StridedIterator}.
 */
@Experimental
public final class ValidationKernels {

    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    /** Block size of the pairwise summation, as in NumPy. */
    static final int PAIRWISE_BLOCK = 128;

    private ValidationKernels() {
    }

    /**
     * Element-wise {@code a + b} with NumPy broadcasting; the result is a new C-contiguous
     * array in automatically managed memory.
     *
     * @param a first operand, float64
     * @param b second operand, float64
     * @return the sum, with the broadcast shape of {@code a} and {@code b}
     * @throws DTypeException if an operand is not float64
     * @throws org.numjvm.exception.ShapeException if the shapes cannot be broadcast
     */
    public static NDArray add(NDArray a, NDArray b) {
        return add(a, b, HybridAllocator.defaultInstance());
    }

    /**
     * Same as {@link #add(NDArray, NDArray)}, with the result allocated in {@code scope}.
     *
     * @param a     first operand, float64
     * @param b     second operand, float64
     * @param scope scope that owns the result
     * @return the sum, valid until {@code scope} is closed
     */
    public static NDArray add(NDArray a, NDArray b, NJScope scope) {
        return add(a, b, SharedSecrets.scopeAccess().allocator(scope));
    }

    private static NDArray add(NDArray a, NDArray b, MemoryAllocator allocator) {
        requireFloat64(a, "add");
        requireFloat64(b, "add");
        Shape shape = Broadcasting.broadcastShapes(a.shape(), b.shape());
        NDArray out = SharedSecrets.arrayAccess().allocate(allocator, DType.FLOAT64, shape, Order.C);
        if (out.size() == 0) {
            return out;
        }
        SharedSecrets.ArrayAccess access = SharedSecrets.arrayAccess();
        MemorySegment sa = access.buffer(a).segment();
        MemorySegment sb = access.buffer(b).segment();
        MemorySegment so = access.buffer(out).segment();
        try {
            if (a.isCContiguous() && b.isCContiguous() && a.shape().equals(shape) && b.shape().equals(shape)) {
                long oa = access.offset(a);
                long ob = access.offset(b);
                for (long i = 0, n = out.size(); i < n; i++) {
                    long k = i << 3;
                    so.set(F64, k, sa.get(F64, oa + k) + sb.get(F64, ob + k));
                }
                return out;
            }
            StridedIterator it = StridedIterator.of(shape.dimensions(),
                    new long[][]{
                            Broadcasting.broadcastStrides(a.shape(), a.strides(), shape).values(),
                            Broadcasting.broadcastStrides(b.shape(), b.strides(), shape).values(),
                            out.strides().values()},
                    new long[]{access.offset(a), access.offset(b), 0});
            do {
                long pa = it.offset(0);
                long pb = it.offset(1);
                long po = it.offset(2);
                long da = it.innerStride(0);
                long db = it.innerStride(1);
                long dout = it.innerStride(2);
                for (long k = 0, n = it.innerSize(); k < n; k++, pa += da, pb += db, po += dout) {
                    so.set(F64, po, sa.get(F64, pa) + sb.get(F64, pb));
                }
            } while (it.next());
            return out;
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, deadOwner(a, b));
        }
    }

    /**
     * Sums all elements of a float64 array with NumPy's pairwise algorithm: blocks of
     * 128 elements are added with eight independent accumulators and blocks are combined
     * recursively, giving O(log n) rounding error. Non-contiguous arrays are summed pairwise
     * along their innermost axis, and the per-row results are added in order, as NumPy does.
     *
     * @param a the array, float64
     * @return the sum; {@code 0.0} for an empty array
     * @throws DTypeException if the array is not float64
     */
    public static double sumFloat64(NDArray a) {
        requireFloat64(a, "sumFloat64");
        if (a.size() == 0) {
            return 0.0;
        }
        SharedSecrets.ArrayAccess access = SharedSecrets.arrayAccess();
        Buffer buffer = access.buffer(a);
        MemorySegment s = buffer.segment();
        try {
            if (a.isCContiguous()) {
                return pairwise(s, access.offset(a), a.size(), 8);
            }
            StridedIterator it = StridedIterator.of(a.shape().dimensions(), new long[][]{a.strides().values()},
                    new long[]{access.offset(a)});
            double total = 0.0;
            do {
                total += pairwise(s, it.offset(0), it.innerSize(), it.innerStride(0));
            } while (it.next());
            return total;
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, buffer.owner());
        }
    }

    /** NumPy's {@code pairwise_sum} for doubles at byte stride {@code stride}. */
    static double pairwise(MemorySegment s, long off, long n, long stride) {
        if (n < 8) {
            double res = 0.0;
            for (long i = 0; i < n; i++) {
                res += s.get(F64, off + i * stride);
            }
            return res;
        } else if (n <= PAIRWISE_BLOCK) {
            double r0 = s.get(F64, off);
            double r1 = s.get(F64, off + stride);
            double r2 = s.get(F64, off + 2 * stride);
            double r3 = s.get(F64, off + 3 * stride);
            double r4 = s.get(F64, off + 4 * stride);
            double r5 = s.get(F64, off + 5 * stride);
            double r6 = s.get(F64, off + 6 * stride);
            double r7 = s.get(F64, off + 7 * stride);
            long limit = n - (n % 8);
            long i;
            for (i = 8; i < limit; i += 8) {
                long p = off + i * stride;
                r0 += s.get(F64, p);
                r1 += s.get(F64, p + stride);
                r2 += s.get(F64, p + 2 * stride);
                r3 += s.get(F64, p + 3 * stride);
                r4 += s.get(F64, p + 4 * stride);
                r5 += s.get(F64, p + 5 * stride);
                r6 += s.get(F64, p + 6 * stride);
                r7 += s.get(F64, p + 7 * stride);
            }
            double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) {
                res += s.get(F64, off + i * stride);
            }
            return res;
        } else {
            long n2 = n / 2;
            n2 -= n2 % 8;
            return pairwise(s, off, n2, stride) + pairwise(s, off + n2 * stride, n - n2, stride);
        }
    }

    private static void requireFloat64(NDArray a, String op) {
        if (a.dtype() != DType.FLOAT64) {
            throw new DTypeException(op + " is only implemented for float64 in NumJVM 0.1, got " + a.dtype().typeName()
                    + "; convert with astype(DType.FLOAT64)");
        }
    }

    private static org.numjvm.memory.MemoryOwner deadOwner(NDArray a, NDArray b) {
        return !a.isAlive() ? a.memoryOwner() : b.memoryOwner();
    }
}
