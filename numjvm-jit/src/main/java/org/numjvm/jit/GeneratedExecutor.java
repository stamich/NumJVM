package org.numjvm.jit;

import java.lang.foreign.MemorySegment;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;

/** Invokes a generated scalar kernel over eligible runs. */
final class GeneratedExecutor {
    private GeneratedExecutor() { }

    static boolean eligible(NDArray[] inputs, NDArray output) {
        var traversal = new Traversal(output.shape(), inputs, output);
        var it = traversal.iterator();
        if (!it.hasElements()) return true;
        do {
            if (!traversal.generatedEligible(it)) return false;
        } while (it.next());
        return true;
    }

    static void run(GeneratedKernel kernel, NDArray[] inputs, NDArray output) {
        var iterator = new Traversal(output.shape(), inputs, output).iterator();
        if (!iterator.hasElements()) return;
        MemorySegment[] segments = new MemorySegment[inputs.length + 1];
        long[] offsets = new long[segments.length];
        long[] strides = new long[segments.length];
        for (int i = 0; i < inputs.length; i++) segments[i] = SharedSecrets.arrayAccess().buffer(inputs[i]).segment();
        segments[inputs.length] = SharedSecrets.arrayAccess().buffer(output).segment();
        do {
            for (int i = 0; i < segments.length; i++) {
                offsets[i] = iterator.offset(i);
                strides[i] = iterator.innerStride(i);
            }
            kernel.run(segments, offsets, strides, iterator.innerSize());
        } while (iterator.next());
    }
}
