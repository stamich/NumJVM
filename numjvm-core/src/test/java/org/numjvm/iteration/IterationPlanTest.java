package org.numjvm.iteration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import net.jqwik.api.Arbitraries;
import net.jqwik.api.Arbitrary;
import net.jqwik.api.Combinators;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;
import org.junit.jupiter.api.Test;
import org.numjvm.exception.ShapeException;

class IterationPlanTest {

    private static long[] c(long... dims) {
        long[] st = new long[dims.length];
        long s = 8;
        for (int d = dims.length - 1; d >= 0; d--) {
            st[d] = s;
            s *= dims[d];
        }
        return st;
    }

    private static long[] f(long... dims) {
        long[] st = new long[dims.length];
        long s = 8;
        for (int d = 0; d < dims.length; d++) {
            st[d] = s;
            s *= dims[d];
        }
        return st;
    }

    @Test
    void cContiguousBecomesOneRun() {
        IterationPlan p = IterationPlan.of(new long[]{2, 3, 4}, new long[][]{c(2, 3, 4), c(2, 3, 4)}, new long[]{0, 0});
        assertEquals(1, p.rank());
        assertArrayEquals(new long[]{24}, p.shape());
        assertArrayEquals(new long[]{8}, p.strides(0));
        assertEquals(24, p.size());
    }

    @Test
    void fortranOrderAndTransposeBecomeOneRun() {
        IterationPlan pf = IterationPlan.of(new long[]{2, 3, 4}, new long[][]{f(2, 3, 4)}, new long[]{0});
        assertArrayEquals(new long[]{24}, pf.shape());
        assertArrayEquals(new long[]{8}, pf.strides(0));
        // the transpose of a C-contiguous 3x4 array: shape [4, 3], strides [8, 32]
        IterationPlan pt = IterationPlan.of(new long[]{4, 3}, new long[][]{{8, 32}}, new long[]{0});
        assertArrayEquals(new long[]{12}, pt.shape());
        assertArrayEquals(new long[]{8}, pt.strides(0));
    }

    @Test
    void reversedArrayCoalescesWithNegativeStride() {
        IterationPlan p = IterationPlan.of(new long[]{3, 4}, new long[][]{{-32, -8}}, new long[]{88});
        assertArrayEquals(new long[]{12}, p.shape());
        assertArrayEquals(new long[]{-8}, p.strides(0));
        assertEquals(88, p.offset(0));
    }

    @Test
    void cOrderWinsWhenOperandsDisagree() {
        // operand 0 C-contiguous, operand 1 transposed: keep C order, no coalescing possible
        IterationPlan p = IterationPlan.of(new long[]{3, 4}, new long[][]{c(3, 4), {8, 24}}, new long[]{0, 0});
        assertArrayEquals(new long[]{3, 4}, p.shape());
        assertArrayEquals(new long[]{32, 8}, p.strides(0));
        assertArrayEquals(new long[]{8, 24}, p.strides(1));
    }

    @Test
    void broadcastOperandsDoNotVote() {
        // reduction over axis 0 of a transposed input: output stride 0 on the reduced axis,
        // input [n, m] with strides [8, 8n]; the reduced (contiguous) axis becomes innermost
        IterationPlan p = IterationPlan.of(new long[]{5, 7}, new long[][]{{8, 40}, {0, 8}}, new long[]{0, 0});
        assertArrayEquals(new long[]{7, 5}, p.shape());
        assertArrayEquals(new long[]{40, 8}, p.strides(0));
        assertArrayEquals(new long[]{8, 0}, p.strides(1));
    }

    @Test
    void rowBroadcastKeepsTwoAxes() {
        IterationPlan p = IterationPlan.of(new long[]{3, 4}, new long[][]{c(3, 4), {0, 8}, c(3, 4)},
                new long[]{0, 0, 0});
        assertArrayEquals(new long[]{3, 4}, p.shape());
        assertArrayEquals(new long[]{0, 8}, p.strides(1));
    }

    @Test
    void scalarBroadcastCoalesces() {
        IterationPlan p = IterationPlan.of(new long[]{3, 4}, new long[][]{c(3, 4), {0, 0}}, new long[]{0, 16});
        assertArrayEquals(new long[]{12}, p.shape());
        assertArrayEquals(new long[]{0}, p.strides(1));
    }

    @Test
    void lengthOneAxesAreDropped() {
        IterationPlan p = IterationPlan.of(new long[]{1, 3, 1, 4, 1}, new long[][]{{999, 32, 999, 8, 999}},
                new long[]{0});
        assertArrayEquals(new long[]{12}, p.shape());
        assertArrayEquals(new long[]{8}, p.strides(0));
    }

    @Test
    void scalarShapeIsOneElement() {
        IterationPlan p = IterationPlan.of(new long[0], new long[][]{{}}, new long[]{40});
        assertEquals(0, p.rank());
        assertEquals(1, p.size());
        assertTrue(p.hasElements());
        StridedIterator it = p.iterator();
        assertEquals(1, it.innerSize());
        assertEquals(40, it.offset(0));
        assertFalse(it.next());
    }

    @Test
    void emptyShapeHasNoElements() {
        IterationPlan p = IterationPlan.of(new long[]{3, 0, 2}, new long[][]{c(3, 0, 2)}, new long[]{0});
        assertEquals(0, p.size());
        assertFalse(p.hasElements());
        assertFalse(p.iterator().hasElements());
    }

    @Test
    void validatesArguments() {
        assertThrows(ShapeException.class, () -> IterationPlan.of(new long[]{2}, new long[][]{{8}}, new long[]{0, 0}));
        assertThrows(ShapeException.class, () -> IterationPlan.of(new long[]{2}, new long[][]{{8, 8}}, new long[]{0}));
        assertThrows(ShapeException.class, () -> IterationPlan.of(new long[]{-1}, new long[][]{{8}}, new long[]{0}));
        assertThrows(ShapeException.class, () -> IterationPlan.of(new long[]{2}, new long[0][], new long[0]));
        assertThrows(ShapeException.class,
                () -> IterationPlan.of(new long[]{1L << 40, 1L << 40}, new long[][]{{0, 0}}, new long[]{0}));
    }

    // ------------------------------------------------------------------ properties

    record Case(long[] shape, long[][] strides, long[] offsets) {
        @Override
        public String toString() {
            return "shape=" + Arrays.toString(shape) + " strides=" + Arrays.deepToString(strides)
                    + " offsets=" + Arrays.toString(offsets);
        }
    }

    @Provide
    Arbitrary<Case> cases() {
        Arbitrary<long[]> shapes = Arbitraries.longs().between(0, 4).array(long[].class).ofMaxSize(4);
        return Combinators.combine(shapes, Arbitraries.integers().between(1, 3)).flatAs((shape, nops) ->
                Arbitraries.integers().between(-5, 5).array(int[].class).ofSize(shape.length * nops)
                        .map(m -> {
                            long[][] st = new long[nops][shape.length];
                            long[] off = new long[nops];
                            for (int op = 0; op < nops; op++) {
                                for (int d = 0; d < shape.length; d++) {
                                    st[op][d] = 8L * m[op * shape.length + d];
                                }
                                off[op] = 1000L * op + 500;
                            }
                            return new Case(shape, st, off);
                        }));
    }

    @Property(tries = 2000)
    void visitsTheSameElementsAsCOrder(@ForAll("cases") Case c) {
        List<long[]> expected = cOrder(c);
        List<long[]> actual = new ArrayList<>();
        IterationPlan plan = IterationPlan.of(c.shape, c.strides, c.offsets);
        assertEquals(expected.size(), plan.size(), c::toString);
        StridedIterator it = plan.iterator();
        if (it.hasElements()) {
            do {
                long[] base = new long[c.offsets.length];
                for (int op = 0; op < base.length; op++) {
                    base[op] = it.offset(op);
                }
                for (long k = 0; k < it.innerSize(); k++) {
                    long[] t = new long[base.length];
                    for (int op = 0; op < base.length; op++) {
                        t[op] = base[op] + k * it.innerStride(op);
                    }
                    actual.add(t);
                }
            } while (it.next());
        }
        Comparator<long[]> cmp = Arrays::compare;
        expected.sort(cmp);
        actual.sort(cmp);
        assertEquals(expected.size(), actual.size(), c::toString);
        for (int i = 0; i < expected.size(); i++) {
            assertArrayEquals(expected.get(i), actual.get(i), c::toString);
        }
    }

    @Property(tries = 500)
    void neverIncreasesRank(@ForAll("cases") Case c) {
        IterationPlan plan = IterationPlan.of(c.shape, c.strides, c.offsets);
        assertTrue(plan.rank() <= Math.max(1, c.shape.length), c::toString);
        for (long d : plan.shape()) {
            assertTrue(d != 1 || plan.size() == 1, c::toString);
        }
    }

    private static List<long[]> cOrder(Case c) {
        List<long[]> out = new ArrayList<>();
        long[] dims = c.shape;
        for (long d : dims) {
            if (d == 0) {
                return out;
            }
        }
        long[] idx = new long[dims.length];
        while (true) {
            long[] t = new long[c.offsets.length];
            for (int op = 0; op < t.length; op++) {
                long o = c.offsets[op];
                for (int d = 0; d < dims.length; d++) {
                    o += idx[d] * c.strides[op][d];
                }
                t[op] = o;
            }
            out.add(t);
            int d = dims.length - 1;
            while (d >= 0 && ++idx[d] == dims[d]) {
                idx[d] = 0;
                d--;
            }
            if (d < 0) {
                return out;
            }
        }
    }
}
