package org.numjvm.jit;

import java.lang.foreign.MemorySegment;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;
import org.numjvm.internal.backend.FusedKernel;

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

    static boolean hasVectorRun(NDArray[] inputs, NDArray output, int lanes) {
        var traversal = new Traversal(output.shape(), inputs, output);
        var it = traversal.iterator();
        if (!it.hasElements()) return false;
        do {
            if (it.innerSize() >= lanes) return true;
        } while (it.next());
        return false;
    }

    static void run(GeneratedKernel kernel, NDArray[] inputs, NDArray output) {
        runLoop(kernel::run, inputs, output);
    }

    static void run(FusedKernel kernel, NDArray[] inputs, NDArray output) {
        runLoop(kernel::run, inputs, output);
    }

    @FunctionalInterface private interface Run {
        void run(MemorySegment[] segments, long[] offsets, long[] strides, long size);
    }

    private static void runLoop(Run kernel, NDArray[] inputs, NDArray output) {
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
