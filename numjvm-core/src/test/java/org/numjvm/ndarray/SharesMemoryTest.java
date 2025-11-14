package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.memory.NJScope;

class SharesMemoryTest {

    @Test
    void transposeSharesMemory() {
        // The example from the M0 specification, section 18.1.
        NDArray a = NJ.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray b = a.transpose();
        assertTrue(a.sharesMemoryWith(b));
        assertTrue(b.sharesMemoryWith(a));
    }

    @Test
    void viewsOfOneBuffer() {
        NDArray a = NJ.arange(0.0, 10.0, 1.0);
        assertTrue(a.sharesMemoryWith(a));
        assertTrue(a.sharesMemoryWith(a.slice(Slice.range(3, 4))));
        assertTrue(a.sharesMemoryWith(a.reshape(2, 5)));
        assertTrue(a.sharesMemoryWith(a.slice(Slice.step(-1))));
        assertTrue(a.slice(Slice.range(0, 5)).sharesMemoryWith(a.slice(Slice.range(4, 10))), "overlap at element 4");
    }

    @Test
    void disjointSlicesOfOneBufferDoNotShare() {
        NDArray a = NJ.arange(0.0, 10.0, 1.0);
        assertFalse(a.slice(Slice.to(5)).sharesMemoryWith(a.slice(Slice.from(5))));
        NDArray g = NJ.zeros(4, 4);
        assertFalse(g.slice(Slice.index(0)).sharesMemoryWith(g.slice(Slice.index(1))));
    }

    @Test
    void interleavedSlicesAreReportedConservatively() {
        NDArray a = NJ.arange(0.0, 10.0, 1.0);
        assertTrue(a.slice(Slice.range(0, 10, 2)).sharesMemoryWith(a.slice(Slice.range(1, 10, 2))),
                "may_share_memory semantics: extents overlap");
    }

    @Test
    void separateArraysAndCopiesDoNotShare() {
        NDArray a = NJ.zeros(5);
        assertFalse(a.sharesMemoryWith(NJ.zeros(5)));
        assertFalse(a.sharesMemoryWith(a.copy()));
        NDArray big = NJ.zeros(DType.FLOAT64, 1 << 18);
        assertTrue(big.isNative());
        assertFalse(big.sharesMemoryWith(NJ.zeros(DType.FLOAT64, 1 << 18)));
        assertFalse(big.sharesMemoryWith(a), "native vs heap");
    }

    @Test
    void nativeAndScopedViews() {
        try (NJScope scope = NJ.scope()) {
            NDArray a = scope.zeros(DType.INT32, 8, 8);
            assertTrue(a.sharesMemoryWith(a.transpose()));
            assertFalse(a.slice(Slice.to(4)).sharesMemoryWith(a.slice(Slice.from(4))));
            assertFalse(a.sharesMemoryWith(scope.zeros(DType.INT32, 8, 8)));
        }
    }

    @Test
    void emptyArraysNeverShare() {
        NDArray a = NJ.zeros(5);
        assertFalse(a.slice(Slice.range(2, 2)).sharesMemoryWith(a));
    }

    @Test
    void broadcastViewsShareWithTheirSource() {
        NDArray row = NJ.arange(0.0, 3.0, 1.0);
        assertTrue(row.broadcastTo(4, 3).sharesMemoryWith(row));
    }
}
