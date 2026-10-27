package org.numjvm.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ShapeException;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Slice;
import org.numjvm.testing.ArrayArbitraries;

class ValidationKernelsTest {

    private static double[] v(NDArray a) {
        return ArrayArbitraries.values(a);
    }

    @Test
    void addContiguous() {
        NDArray a = NJ.array(new double[]{1, 2, 3});
        NDArray b = NJ.array(new double[]{10, 20, 30});
        assertArrayEquals(new double[]{11, 22, 33}, v(ValidationKernels.add(a, b)));
    }

    @Test
    void addBroadcasts() {
        NDArray m = NJ.arange(0.0, 12.0, 1.0).reshape(3, 4);
        NDArray row = NJ.array(new double[]{100, 200, 300, 400});
        NDArray col = NJ.array(new double[]{1000, 2000, 3000}, 3, 1);
        NDArray r1 = ValidationKernels.add(m, row);
        assertEquals(Shape.of(3, 4), r1.shape());
        assertEquals(411.0, r1.getDouble(2, 3));
        NDArray r2 = ValidationKernels.add(col, row);           // (3, 1) + (4,) -> (3, 4)
        assertEquals(Shape.of(3, 4), r2.shape());
        assertEquals(3400.0, r2.getDouble(2, 3));
        assertEquals(12.5, ValidationKernels.add(NJ.scalar(2.5), NJ.scalar(10.0)).getDouble());
        assertEquals(Shape.of(3, 4), ValidationKernels.add(NJ.scalar(1.0), m).shape());
    }

    @Test
    void addOnViews() {
        NDArray m = NJ.arange(0.0, 6.0, 1.0).reshape(2, 3);
        NDArray t = m.transpose();                               // (3, 2)
        NDArray rev = NJ.arange(0.0, 6.0, 1.0).slice(Slice.step(-1)).reshape(3, 2);
        NDArray r = ValidationKernels.add(t, rev);
        assertArrayEquals(new double[]{0 + 5, 3 + 4, 1 + 3, 4 + 2, 2 + 1, 5 + 0}, v(r));
        assertTrue(r.isCContiguous());
        NDArray bc = NJ.array(new double[]{1, 2}).broadcastTo(3, 2);
        assertArrayEquals(new double[]{1, 5, 2, 6, 3, 7}, v(ValidationKernels.add(t, bc)));
    }

    @Test
    void addEmptyAndErrors() {
        assertEquals(Shape.of(0, 3), ValidationKernels.add(NJ.zeros(0, 3), NJ.zeros(3)).shape());
        assertThrows(ShapeException.class, () -> ValidationKernels.add(NJ.zeros(2, 3), NJ.zeros(4, 3)));
        DTypeException e = assertThrows(DTypeException.class,
                () -> ValidationKernels.add(NJ.zeros(DType.FLOAT32, 2), NJ.zeros(2)));
        assertTrue(e.getMessage().contains("float32"), e.getMessage());
    }

    @Test
    void addIntoAScope() {
        NDArray r;
        try (NJScope scope = NJ.scope()) {
            NDArray a = scope.array(new double[]{1, 2});
            r = ValidationKernels.add(a, a, scope);
            assertSame(scope, r.memoryOwner());
            assertEquals(4.0, r.getDouble(1));
        }
        assertFalse(r.isAlive());
    }

    @Test
    void addOnAClosedScopeIsAMemoryException() {
        NDArray a;
        try (NJScope scope = NJ.scope()) {
            a = scope.array(new double[]{1, 2, 3});
        }
        NDArray arr = a;
        assertThrows(MemoryException.class, () -> ValidationKernels.add(arr, NJ.zeros(3)));
        assertThrows(MemoryException.class, () -> ValidationKernels.sumFloat64(arr));
    }

    @Test
    void sumSmallCases() {
        assertEquals(0.0, ValidationKernels.sumFloat64(NJ.zeros(0)));
        assertEquals(2.5, ValidationKernels.sumFloat64(NJ.scalar(2.5)));
        assertEquals(66.0, ValidationKernels.sumFloat64(NJ.arange(0.0, 12.0, 1.0).reshape(3, 4).transpose()));
        assertEquals(30.0, ValidationKernels.sumFloat64(NJ.array(new double[]{1, 2, 3, 4}).broadcastTo(3, 4)));
        assertEquals(45.0, ValidationKernels.sumFloat64(NJ.arange(0.0, 10.0, 1.0).slice(Slice.step(-1))));
        assertThrows(DTypeException.class, () -> ValidationKernels.sumFloat64(NJ.arange(5)));
    }

    /**
     * Integer-valued data has an exact sum, n(n-1)/2, for every block boundary.
     */
    @ParameterizedTest
    @ValueSource(longs = {1, 7, 8, 9, 127, 128, 129, 1000, 65_536, 1_048_577})
    void sumIsExactForIntegers(long n) {
        assertEquals((double) n * (n - 1) / 2, ValidationKernels.sumFloat64(NJ.arange(0.0, n, 1.0)));
    }

    /**
     * Pseudo-random values in [0, 1) that NumPy can reproduce exactly (the oracle uses the
     * same formula); compared with a compensated reference sum.
     */
    @ParameterizedTest
    @ValueSource(ints = {1_000, 65_536, 1_048_576, 16_777_216})
    void sumIsAccurateToOneInATrillionUpTo16M(int n) {
        NDArray a = NJ.zeros(n);
        double reference = 0.0;
        double compensation = 0.0;
        NDArray flat = a.reshapeView(-1);
        for (int i = 0; i < n; i++) {
            double x = hashUnit(i);
            flat.setDouble(i, x);
            double t = reference + x;
            compensation += Math.abs(reference) >= Math.abs(x) ? (reference - t) + x : (x - t) + reference;
            reference = t;
        }
        reference += compensation;
        double sum = ValidationKernels.sumFloat64(a);
        assertEquals(reference, sum, Math.abs(reference) * 1e-12);
    }

    /**
     * (i * 2654435761) mod 2^32, scaled to [0, 1): exact in double precision.
     */
    public static double hashUnit(long i) {
        return ((i * 2654435761L) & 0xFFFF_FFFFL) / 4294967296.0;
    }

    @Provide
    Arbitrary<NDArray> arrays() {
        return ArrayArbitraries.arrays();
    }

    @Property
    void addMatchesElementwiseReference(@ForAll("arrays") NDArray a) {
        NDArray t = a.transpose();
        NDArray sum = ValidationKernels.add(t, t.copy());
        double[] expected = v(t);
        for (int i = 0; i < expected.length; i++) {
            expected[i] *= 2;
        }
        assertArrayEquals(expected, v(sum));
    }

    @Property
    void sumOfSequentialValues(@ForAll("arrays") NDArray a) {
        double n = a.size();
        assertEquals(n * (n - 1) / 2, ValidationKernels.sumFloat64(a.transpose()), 0.0);
    }
}
