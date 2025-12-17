package org.numjvm.internal.loops;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;

/**
 * Every registered inner loop must give the same result on each of its paths (ADR-004):
 * all operands contiguous, one input broadcast along the run, and the general strided loop.
 * The contiguous result is the reference; the other paths see the same logical values in
 * other layouts. This runs every generated loop through every path, so a template error in
 * any path of any operation or dtype shows up here.
 */
class LoopPathsTest {

    private static final int N = 13;

    @TestFactory
    Stream<DynamicTest> everyLoopGivesTheSameResultOnEveryPath() {
        List<DynamicTest> tests = new ArrayList<>();
        for (String op : KernelRegistry.operations()) {
            for (LoopSignature sig : KernelRegistry.signatures(op)) {
                tests.add(DynamicTest.dynamicTest(sig.toString(), () -> check(sig)));
            }
        }
        return tests.stream();
    }

    private static void check(LoopSignature sig) {
        DType in = sig.inputs().getFirst();
        DType out = sig.output();
        NDArray a = values(in, N, 0, false);
        if (sig.arity() == 1) {
            UnaryLoop loop = KernelRegistry.unary(sig.operation(), in);
            NDArray reference = unary(loop, a, out);
            // strided input (every other element, reversed) and strided output
            NDArray sa = strided(a);
            NDArray so = NJ.zeros(out, 2 * N).slice(Slice.step(2));
            LoopExecutor.unary(loop, sa, so);
            assertSame(sig + " strided", reference, so);
            // broadcast input: one element repeated, against the contiguous result of a filled array
            NDArray first = a.slice(Slice.range(0, 1)).reshape();
            assertSame(sig + " broadcast", unary(loop, fill(first, N), out), unary(loop, first.broadcastTo(N), out));
            return;
        }
        BinaryLoop loop = KernelRegistry.binary(sig.operation(), in, sig.inputs().get(1));
        NDArray b = values(in, N, 3, sig.operation().equals("power"));
        NDArray reference = binary(loop, a, b, out);
        NDArray so = NJ.zeros(out, 2 * N).slice(Slice.step(2));
        LoopExecutor.binary(loop, strided(a), strided(b), so);
        assertSame(sig + " strided", reference, so);
        NDArray a0 = a.slice(Slice.range(0, 1)).reshape();
        NDArray b0 = b.slice(Slice.range(2, 3)).reshape();
        assertSame(sig + " broadcast first", binary(loop, fill(a0, N), b, out), binary(loop, a0.broadcastTo(N), b, out));
        assertSame(sig + " broadcast second", binary(loop, a, fill(b0, N), out), binary(loop, a, b0.broadcastTo(N), out));
        if (in == out) {
            checkReduction(sig, loop, a0, values(in, 4, 5, sig.operation().equals("power")));
        }
    }

    /**
     * The reduction path (output passed as the first input, both with stride 0) against
     * applying the loop one element at a time. float16 and bfloat16 accumulate in float32
     * without rounding each step, as NumPy does, so for them the path only runs.
     */
    private static void checkReduction(LoopSignature sig, BinaryLoop loop, NDArray start, NDArray b) {
        DType t = start.dtype();
        NDArray acc = start.copy();
        MemorySegment ma = SharedSecrets.arrayAccess().buffer(acc).segment();
        MemorySegment mb = SharedSecrets.arrayAccess().buffer(b).segment();
        loop.run(ma, 0, 0, mb, 0, t.size(), ma, 0, 0, b.size());
        NDArray step = start.copy();
        for (long i = 0; i < b.size(); i++) {
            NDArray next = NJ.zeros(t);
            LoopExecutor.binary(loop, step, b.slice(Slice.range(i, i + 1)).reshape(), next);
            step = next;
        }
        if (t != DType.FLOAT16 && t != DType.BFLOAT16) {
            assertSame(sig + " reduction", step, acc);
        }
    }

    private static NDArray unary(UnaryLoop loop, NDArray a, DType out) {
        NDArray o = NJ.zeros(out, a.shape().dimensions());
        LoopExecutor.unary(loop, a, o);
        return o;
    }

    private static NDArray binary(BinaryLoop loop, NDArray a, NDArray b, DType out) {
        NDArray o = NJ.zeros(out, N);
        LoopExecutor.binary(loop, a, b, o);
        return o;
    }

    /**
     * The same logical values as {@code a}, at stride {@code -2} elements in a larger array.
     */
    private static NDArray strided(NDArray a) {
        int n = Math.toIntExact(a.size());
        NDArray big = NJ.zeros(a.dtype(), 2 * n);
        // big[2n-1], big[2n-3], ... = a[0], a[1], ...: a reversed, every-other-element view
        NDArray view = big.slice(Slice.step(-2));
        LoopExecutor.unary(KernelRegistry.unary("copy", a.dtype()), a, view);
        return view;
    }

    private static NDArray fill(NDArray scalar, int n) {
        NDArray f = NJ.zeros(scalar.dtype(), n);
        LoopExecutor.unary(KernelRegistry.unary("copy", scalar.dtype()), scalar.broadcastTo(n), f);
        return f;
    }

    /**
     * {@code n} distinguishable values for dtype {@code t}: small integers (non-negative when
     * {@code nonNegative}, for integer exponents), halves and quarters for floats, and
     * real-valued complex numbers.
     */
    private static NDArray values(DType t, int n, int shift, boolean nonNegative) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            int k = (i + shift) % n;
            if (t.isBool()) {
                v[i] = k % 3 == 0 ? 1 : 0;
            } else if (t.isUnsigned() || nonNegative && t.isInteger()) {
                v[i] = k % 6;
            } else if (t.isInteger()) {
                v[i] = k % 7 - 3;
            } else if (nonNegative) {
                v[i] = k * 0.25;
            } else {
                v[i] = (k - 4) * 0.75;
            }
        }
        if (t.isComplex()) {
            return NJ.array(v).astype(t);
        }
        return NJ.array(t, v);
    }

    private static void assertSame(String what, NDArray expected, NDArray actual) {
        assertArrayEquals(bytes(expected), bytes(actual), what);
    }

    private static byte[] bytes(NDArray a) {
        NDArray c = NJ.zeros(a.dtype(), a.shape().dimensions());
        LoopExecutor.unary(KernelRegistry.unary("copy", a.dtype()), a, c);
        MemorySegment s = SharedSecrets.arrayAccess().buffer(c).segment();
        return s.asSlice(0, c.nbytes()).toArray(ValueLayout.JAVA_BYTE);
    }
}
