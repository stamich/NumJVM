package org.numjvm.backend;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;
class WideningTest {
    @Test void wideningMatchesExplicitCastOnAxesAndReversedInputs() {
        for (DType t : new DType[]{DType.BOOL,DType.INT8,DType.UINT8,DType.INT16,DType.UINT16,DType.INT32,DType.UINT32}) {
            var x = NJ.arange(0, 24, 1).astype(t).reshape(4, 6).slice(Slice.step(-1), Slice.all());
            var wide = x.astype(t.isUnsigned() ? DType.UINT64 : DType.INT64);
            for (int axis : new int[]{0,1}) {
                var sum = NJ.sum(x, axis); var expected = NJ.sum(wide, axis);
                var prod = NJ.prod(x, axis); var ep = NJ.prod(wide, axis);
                for (long i=0; i<sum.size(); i++) {
                    assertEquals(expected.getLong(i),sum.getLong(i));
                    assertEquals(ep.getLong(i),prod.getLong(i));
                }
            }
        }
        assertEquals(0L,NJ.sum(NJ.zeros(DType.INT8,0)).getLong());
        assertEquals(1L,NJ.prod(NJ.zeros(DType.UINT8,0)).getLong());
    }
}
