package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class TraversalTest {
    @Test void classifiesContiguousAndReversedRuns() {
        NDArray input = NJ.arange(0, 10, 1).astype(org.numjvm.dtype.DType.FLOAT64);
        NDArray out = NJ.zeros(10);
        var contiguous = new Traversal(out.shape(), new NDArray[]{input}, out);
        var it = contiguous.iterator();
        assertTrue(it.hasElements());
        assertEquals(10, it.innerSize());
        assertTrue(contiguous.generatedEligible(it));
        NDArray reversed = input.slice(Slice.step(-1));
        var generic = new Traversal(out.shape(), new NDArray[]{reversed}, out);
        assertFalse(generic.generatedEligible(generic.iterator()));
    }
    @Test void emptyTraversalHasNoRuns() {
        NDArray empty = NJ.zeros(0);
        var plan = new Traversal(empty.shape(), new NDArray[]{empty}, empty);
        assertFalse(plan.iterator().hasElements());
        assertTrue(plan.generatedEligible(plan.iterator()));
    }
}
