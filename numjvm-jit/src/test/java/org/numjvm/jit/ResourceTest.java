package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.internal.memory.Scratch;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class ResourceTest {
    @Test void blockScratchAndAliasedTemporaryAreReleased() {
        var storage = NJ.ones(DType.FLOAT64, 1001);
        var source = storage.slice(Slice.range(0, 1000));
        var output = storage.slice(Slice.range(1, 1001));
        try (var context = Jit.context(Jit.Mode.BLOCK)) {
            var expression = context.trace(1, i -> i.input(0).add(0));
            long nativeBefore = NJ.memoryStats().liveNativeBytes();
            long scratchBefore = Scratch.activeBytes();
            for (int n = 0; n < 30; n++) expression.evalInto(output, source);
            assertEquals(scratchBefore, Scratch.activeBytes());
            assertEquals(nativeBefore, NJ.memoryStats().liveNativeBytes());
            assertEquals(1.0, output.getDouble(999));
        }
    }
}
