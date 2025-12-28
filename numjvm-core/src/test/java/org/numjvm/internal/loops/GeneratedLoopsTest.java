package org.numjvm.internal.loops;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Slice;

class GeneratedLoopsTest {

    static final Set<DType> REAL = EnumSet.complementOf(EnumSet.of(DType.COMPLEX64, DType.COMPLEX128));

    @Test
    void everyRealDtypeHasAddAndCopyInNumpyOrder() {
        List<DType> order = new ArrayList<>();
        for (LoopSignature s : KernelRegistry.signatures("add")) {
            order.add(s.inputs().getFirst());
        }
        assertEquals(List.of(DType.BOOL, DType.INT8, DType.UINT8, DType.INT16, DType.UINT16, DType.INT32,
                DType.UINT32, DType.INT64, DType.UINT64, DType.FLOAT16, DType.FLOAT32, DType.FLOAT64,
                DType.BFLOAT16, DType.COMPLEX64, DType.COMPLEX128), order);
        for (DType t : REAL) {
            assertNotNull(KernelRegistry.find("copy", t), t.typeName());
        }
        int registered = 0;
        for (String op : KernelRegistry.operations()) {
            registered += KernelRegistry.signatures(op).size();
        }
        // generated loops plus the hand-written ones: 8 complex copies and casts, 22 complex
        // arithmetic loops (11 operations), conjugate as copy for the 12 real non-bool dtypes,
        // and 12 exact int64/uint64 comparisons
        assertEquals(GeneratedLoops.LOOP_COUNT + 8 + 22 + 12 + 12, registered);
    }

    /**
     * Small values that every real dtype holds exactly.
     */
    private static NDArray values(DType t, long... shape) {
        NDArray a = NJ.zeros(t, shape);
        long n = a.size();
        NDArray flat = a.reshapeView(n);
        for (long i = 0; i < n; i++) {
            flat.setLong(i, t == DType.BOOL ? i % 3 == 0 ? 1 : 0 : i % 7 + 1);
        }
        return a;
    }

    private static List<NDArray> layouts(DType t) {
        NDArray c = values(t, 4, 6);
        List<NDArray> l = new ArrayList<>();
        l.add(c);
        l.add(values(t, 6, 4).transpose());
        l.add(c.copy(Order.F));
        l.add(c.slice(Slice.step(-1), Slice.step(-1)));
        l.add(values(t, 4, 12).slice(Slice.all(), Slice.step(2)));
        l.add(values(t, 6).broadcastTo(4, 6));
        l.add(values(t, 1, 1).broadcastTo(4, 6));
        return l;
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"COMPLEX64", "COMPLEX128"}, mode = EnumSource.Mode.EXCLUDE)
    void addMatchesScalarArithmeticOnEveryLayout(DType t) {
        BinaryLoop add = KernelRegistry.binary("add", t, t);
        for (NDArray a : layouts(t)) {
            for (NDArray b : layouts(t)) {
                NDArray out = NJ.zeros(t, 4, 6);
                LoopExecutor.binary(add, a, b, out);
                for (long i = 0; i < 4; i++) {
                    for (long j = 0; j < 6; j++) {
                        double x = a.getDouble(i, j);
                        double y = b.getDouble(i, j);
                        double expected = t == DType.BOOL ? (x != 0 || y != 0 ? 1 : 0) : x + y;
                        assertEquals(expected, out.getDouble(i, j), t + " at " + i + "," + j);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"COMPLEX64", "COMPLEX128"}, mode = EnumSource.Mode.EXCLUDE)
    void copyOnEveryLayout(DType t) {
        UnaryLoop copy = KernelRegistry.unary("copy", t);
        for (NDArray a : layouts(t)) {
            NDArray out = NJ.zeros(t, 4, 6);
            LoopExecutor.unary(copy, a, out);
            assertTrue(NJ.arrayEquals(a, out), t.typeName());
        }
    }

    @Test
    void integerAddWrapsAtTheDtypeWidth() {
        NDArray a = NJ.array(DType.INT8, new double[]{127, -128});
        NDArray out = NJ.zeros(DType.INT8, 2);
        LoopExecutor.binary(KernelRegistry.binary("add", DType.INT8, DType.INT8), a, NJ.array(DType.INT8,
                new double[]{1, -1}), out);
        assertEquals(-128, out.getLong(0));
        assertEquals(127, out.getLong(1));
        NDArray u = NJ.array(DType.UINT8, new double[]{255});
        NDArray uo = NJ.zeros(DType.UINT8, 1);
        LoopExecutor.binary(KernelRegistry.binary("add", DType.UINT8, DType.UINT8), u, u, uo);
        assertEquals(254, uo.getLong(0));
        NDArray big = NJ.zeros(DType.UINT64, 1);
        big.setLong(0, -1L);  // 2^64 - 1
        NDArray bo = NJ.zeros(DType.UINT64, 1);
        LoopExecutor.binary(KernelRegistry.binary("add", DType.UINT64, DType.UINT64), big, big, bo);
        assertEquals(-2L, bo.getLong(0));
    }

    @Test
    void float16AddsInFloat32ThenRounds() {
        // NumPy: np.float16(0.1) + np.float16(0.2) == np.float16(0.2998)
        NDArray a = NJ.array(DType.FLOAT16, new double[]{0.1});
        NDArray b = NJ.array(DType.FLOAT16, new double[]{0.2});
        NDArray out = NJ.zeros(DType.FLOAT16, 1);
        LoopExecutor.binary(KernelRegistry.binary("add", DType.FLOAT16, DType.FLOAT16), a, b, out);
        assertEquals(Float.float16ToFloat(Float.floatToFloat16(0.2998f)), out.getDouble(0), 0.0);
    }
}
