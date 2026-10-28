package org.numjvm.internal.loops;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.IntRange;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import org.numjvm.testing.ArrayArbitraries;

class LoopExecutorTest {

    private static NDArray seq(long... shape) {
        return ArrayArbitraries.sequential(org.numjvm.ndarray.Shape.of(shape));
    }

    /**
     * Expected a + b, element by element through logical indices.
     */
    private static void assertAdd(NDArray a, NDArray b) {
        NDArray out = NJ.zeros(DType.FLOAT64, org.numjvm.shape.Broadcasting.broadcastShapes(a.shape(), b.shape())
                .dimensions());
        LoopExecutor.binary(Float64Loops::add, a, b, out);
        NDArray ab = a.broadcastTo(out.shape().dimensions());
        NDArray bb = b.broadcastTo(out.shape().dimensions());
        double[] x = ArrayArbitraries.values(ab);
        double[] y = ArrayArbitraries.values(bb);
        double[] z = ArrayArbitraries.values(out);
        for (int i = 0; i < z.length; i++) {
            assertEquals(x[i] + y[i], z[i], "element " + i);
        }
    }

    static List<NDArray> layouts() {
        NDArray base = seq(4, 6);
        List<NDArray> l = new ArrayList<>();
        l.add(base);
        l.add(seq(6, 4).transpose());
        l.add(base.copy(org.numjvm.ndarray.Order.F));
        l.add(base.slice(Slice.step(-1), Slice.step(-1)));
        l.add(seq(8, 12).slice(Slice.step(2), Slice.step(2)));
        l.add(seq(6).broadcastTo(4, 6));
        l.add(seq(4, 1).broadcastTo(4, 6));
        l.add(NJ.scalar(7.0).broadcastTo(4, 6));
        return l;
    }

    @Test
    void addOverEveryLayoutPair() {
        for (NDArray a : layouts()) {
            for (NDArray b : layouts()) {
                assertAdd(a, b);
            }
        }
    }

    @Test
    void broadcastsInputsToTheOutput() {
        assertAdd(seq(3, 1), seq(4));
        assertAdd(NJ.scalar(2.5), seq(2, 3, 4));
        assertAdd(seq(2, 1, 4), seq(3, 1));
    }

    @Test
    void zeroDimensionalAndEmpty() {
        NDArray out = NJ.zeros(DType.FLOAT64);
        LoopExecutor.binary(Float64Loops::add, NJ.scalar(1.5), NJ.scalar(2.0), out);
        assertEquals(3.5, out.getDouble());
        NDArray empty = NJ.zeros(DType.FLOAT64, 0, 3);
        LoopExecutor.binary(Float64Loops::add, empty, NJ.zeros(3), empty);
        assertEquals(0, empty.size());
    }

    @Test
    void unaryCopiesThroughThePlan() {
        for (NDArray a : layouts()) {
            NDArray out = NJ.zeros(DType.FLOAT64, 4, 6);
            LoopExecutor.unary(Float64Loops::copy, a, out);
            assertTrue(NJ.arrayEquals(a, out));
        }
        NDArray out = NJ.zeros(DType.FLOAT64, 3, 4).transpose();
        LoopExecutor.unary(Float64Loops::copy, seq(4, 3), out);
        assertTrue(NJ.arrayEquals(seq(4, 3), out));
    }

    @Test
    void rejectsReadOnlyOutputAndBadShapes() {
        NDArray ro = NJ.zeros(3).broadcastTo(2, 3);
        assertThrows(ReadOnlyArrayException.class, () -> LoopExecutor.unary(Float64Loops::copy, NJ.zeros(3), ro));
        assertThrows(ShapeException.class,
                () -> LoopExecutor.binary(Float64Loops::add, NJ.zeros(4), NJ.zeros(3), NJ.zeros(3)));
    }

    @Test
    void closedScopeIsAMemoryException() {
        NDArray dead;
        try (NJScope scope = NJ.scope()) {
            dead = scope.zeros(DType.FLOAT64, 4);
        }
        NDArray finalDead = dead;
        MemoryException e = assertThrows(MemoryException.class,
                () -> LoopExecutor.binary(Float64Loops::add, finalDead, NJ.zeros(4), NJ.zeros(4)));
        assertTrue(e.getMessage().contains("closed"), e.getMessage());
    }

    @Property(tries = 200)
    void addMatchesElementwiseOnRandomBroadcasts(@ForAll @IntRange(min = 0, max = 4) int r1,
                                                 @ForAll @IntRange(min = 0, max = 4) int r2,
                                                 @ForAll @IntRange(min = 0, max = 1000) int seed) {
        java.util.Random rnd = new java.util.Random(seed);
        long[] s1 = new long[r1];
        long[] s2 = new long[r2];
        int r = Math.max(r1, r2);
        long[] full = new long[r];
        for (int i = 0; i < r; i++) {
            full[i] = 1 + rnd.nextInt(4);
        }
        for (int i = 0; i < r1; i++) {
            s1[i] = rnd.nextInt(3) == 0 ? 1 : full[r - r1 + i];
        }
        for (int i = 0; i < r2; i++) {
            s2[i] = rnd.nextInt(3) == 0 ? 1 : full[r - r2 + i];
        }
        NDArray a = seq(s1);
        NDArray b = seq(s2);
        if (rnd.nextBoolean()) {
            a = a.transpose().copy().transpose();
        }
        assertAdd(a, b);
    }
}
