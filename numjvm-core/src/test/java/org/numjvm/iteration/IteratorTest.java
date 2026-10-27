package org.numjvm.iteration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;

import org.junit.jupiter.api.Test;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.Shape;
import org.numjvm.testing.ArrayArbitraries;
import org.numjvm.testing.ArrayArbitraries.Layout;

class IteratorTest {

    /**
     * All (offset of operand 0, offset of operand 1, ...) tuples in visiting order.
     */
    private static List<long[]> visit(StridedIterator it) {
        List<long[]> out = new ArrayList<>();
        if (!it.hasElements()) {
            return out;
        }
        do {
            long[] p = new long[it.operands()];
            for (int op = 0; op < p.length; op++) {
                p[op] = it.offset(op);
            }
            for (long k = 0; k < it.innerSize(); k++) {
                out.add(p.clone());
                for (int op = 0; op < p.length; op++) {
                    p[op] += it.innerStride(op);
                }
            }
        } while (it.next());
        return out;
    }

    private static long[] first(List<long[]> tuples) {
        return tuples.stream().mapToLong(t -> t[0]).toArray();
    }

    @Test
    void contiguousAndTransposed() {
        assertArrayEquals(new long[]{0, 8, 16, 24, 32, 40},
                first(visit(StridedIterator.of(new long[]{2, 3}, new long[][]{{24, 8}}, new long[]{0}))));
        assertArrayEquals(new long[]{0, 24, 8, 32, 16, 40},
                first(visit(StridedIterator.of(new long[]{3, 2}, new long[][]{{8, 24}}, new long[]{0}))));
    }

    @Test
    void negativeAndZeroStrides() {
        assertArrayEquals(new long[]{40, 32, 24, 16, 8, 0},
                first(visit(StridedIterator.of(new long[]{6}, new long[][]{{-8}}, new long[]{40}))));
        assertArrayEquals(new long[]{0, 8, 16, 0, 8, 16},
                first(visit(StridedIterator.of(new long[]{2, 3}, new long[][]{{0, 8}}, new long[]{0}))));
    }

    @Test
    void multipleOperandsAdvanceTogether() {
        // (2, 3) contiguous + (3,) row broadcast + (2, 1) column broadcast
        List<long[]> t = visit(StridedIterator.of(new long[]{2, 3},
                new long[][]{{24, 8}, {0, 8}, {8, 0}}, new long[]{0, 100, 200}));
        assertEquals(6, t.size());
        assertArrayEquals(new long[]{0, 100, 200}, t.get(0));
        assertArrayEquals(new long[]{16, 116, 200}, t.get(2));
        assertArrayEquals(new long[]{24, 100, 208}, t.get(3));
        assertArrayEquals(new long[]{40, 116, 208}, t.get(5));
    }

    @Test
    void scalarAndEmptyShapes() {
        StridedIterator s = StridedIterator.of(new long[0], new long[][]{{}}, new long[]{16});
        assertTrue(s.hasElements());
        assertEquals(1, s.innerSize());
        assertArrayEquals(new long[]{16}, first(visit(s)));
        StridedIterator e = StridedIterator.of(new long[]{3, 0}, new long[][]{{8, 8}}, new long[]{0});
        assertFalse(e.hasElements());
        assertEquals(0, visit(e).size());
        assertFalse(StridedIterator.of(new long[]{0, 3}, new long[][]{{8, 8}}, new long[]{0}).hasElements());
    }

    @Test
    void invalidArguments() {
        assertThrows(ShapeException.class, () -> StridedIterator.of(new long[]{2}, new long[][]{{8, 8}}, new long[]{0}));
        assertThrows(ShapeException.class, () -> StridedIterator.of(new long[]{2}, new long[][]{{8}}, new long[]{0, 0}));
        assertThrows(ShapeException.class, () -> StridedIterator.of(new long[]{-1}, new long[][]{{8}}, new long[]{0}));
    }

    @Test
    void argumentsAreCopied() {
        long[] shape = {2, 2};
        long[][] strides = {{16, 8}};
        long[] offsets = {0};
        StridedIterator it = StridedIterator.of(shape, strides, offsets);
        shape[0] = 100;
        strides[0][0] = 0;
        offsets[0] = 999;
        assertArrayEquals(new long[]{0, 8, 16, 24}, first(visit(it)));
    }

    @Test
    void traversalDoesNotAllocatePerElement() {
        com.sun.management.ThreadMXBean threads =
                (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        long tid = Thread.currentThread().threadId();
        long sink = 0;
        for (int warm = 0; warm < 5; warm++) {
            sink += traverse(1024);
        }
        long before = threads.getThreadAllocatedBytes(tid);
        sink += traverse(1024);       // 1,048,576 elements, transposed layout
        long allocated = threads.getThreadAllocatedBytes(tid) - before;
        assertTrue(allocated < 4096, "allocated " + allocated + " bytes for 1,048,576 elements");
        assertTrue(sink != 0);
    }

    private static long traverse(long n) {
        StridedIterator it = StridedIterator.of(new long[]{n, n}, new long[][]{{8, 8 * n}}, new long[]{0});
        long acc = 0;
        do {
            long p = it.offset(0);
            for (long k = 0, len = it.innerSize(); k < len; k++, p += it.innerStride(0)) {
                acc += p;
            }
        } while (it.next());
        return acc;
    }

    @Test
    void indexIteratorVisitsCOrder() {
        IndexIterator it = new IndexIterator(Shape.of(2, 3));
        List<String> seen = new ArrayList<>();
        while (it.hasNext()) {
            seen.add(java.util.Arrays.toString(it.next()));
        }
        assertEquals(List.of("[0, 0]", "[0, 1]", "[0, 2]", "[1, 0]", "[1, 1]", "[1, 2]"), seen);
        assertThrows(NoSuchElementException.class, it::next);
        IndexIterator scalar = new IndexIterator(Shape.scalar());
        assertTrue(scalar.hasNext());
        assertEquals(0, scalar.next().length);
        assertFalse(scalar.hasNext());
        assertFalse(new IndexIterator(Shape.of(2, 0)).hasNext());
    }

    @Provide
    Arbitrary<Layout> layouts() {
        return ArrayArbitraries.layouts();
    }

    @Property
    void visitsTheSameOffsetsAsIndexEnumeration(@ForAll("layouts") Layout layout) {
        StridedIterator it = StridedIterator.of(layout.shape().dimensions(),
                new long[][]{layout.strides().values()}, new long[]{layout.offset()});
        assertArrayEquals(ArrayArbitraries.enumerateOffsets(layout), first(visit(it)), layout::toString);
    }
}
