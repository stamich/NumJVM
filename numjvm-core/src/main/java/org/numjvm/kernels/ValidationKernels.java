package org.numjvm.kernels;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.Experimental;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.LoopExecutor;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.internal.memory.MemoryErrors;
import org.numjvm.iteration.IterationPlan;
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
 * replace this class. Both run over an {@link IterationPlan} (ADR-004), so every layout
 * that is contiguous in some axis order takes the contiguous path.
 */
@Experimental
public final class ValidationKernels {

    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;
    private static final BinaryLoop ADD = KernelRegistry.binary("add", DType.FLOAT64, DType.FLOAT64);

    /**
     * Block size of the pairwise summation, as in NumPy.
     */
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
     * @throws DTypeException                      if an operand is not float64
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
        LoopExecutor.binary(ADD, a, b, out);
        return out;
    }

    /**
     * Sums all elements of a float64 array with NumPy's pairwise algorithm: blocks of
     * 128 elements are added with eight independent accumulators and blocks are combined
     * recursively, giving O(log n) rounding error.
     *
     * <p>The array is traversed through an {@link IterationPlan} (ADR-004), so a C- or
     * F-contiguous array, or a transposed view of one, is summed pairwise as a single run in
     * memory order. Other layouts are summed pairwise along each run of the plan, and the
     * per-run results are added in order.
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
            IterationPlan plan = IterationPlan.of(a.shape().dimensions(), new long[][]{a.strides().values()},
                    new long[]{access.offset(a)});
            StridedIterator it = plan.iterator();
            long n = it.innerSize();
            long stride = it.innerStride(0);
            double total = 0.0;
            do {
                total += pairwise(s, it.offset(0), n, stride);
            } while (it.next());
            return total;
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, buffer.owner());
        }
    }

    /**
     * NumPy's {@code pairwise_sum} for doubles at byte stride {@code stride}.
     */
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

}
