package org.numjvm.examples;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import org.numjvm.kernels.ValidationKernels;
import org.numjvm.memory.MemoryStats;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;

/**
 * Scoped memory: deterministic release, keep(), thread confinement and memory statistics.
 */
public final class MemoryScopes {

    private MemoryScopes() {
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("NumJVM " + NJ.version());
        printStats("start");

        // Temporaries in a loop: each scope frees its arrays at the end of the iteration.
        double total = 0;
        for (int step = 0; step < 100; step++) {
            try (NJScope scope = NJ.scope()) {
                NDArray a = scope.full(DType.FLOAT64, step, 512, 512);     // 2 MiB, native
                NDArray b = ValidationKernels.add(a, a.transpose(), scope);
                total += ValidationKernels.sumFloat64(b);
            }
        }
        System.out.println("total over 100 scoped steps: " + total);
        printStats("after the loop (scoped memory is already released)");

        // keep() copies a result out of a scope; everything else from the scope is gone.
        NDArray kept;
        NDArray temporary;
        try (NJScope scope = NJ.scope()) {
            temporary = scope.arange(0, 6, 1).reshape(2, 3);
            kept = scope.keep(temporary.transpose());
        }
        System.out.println();
        System.out.println("kept = " + kept);
        System.out.println("temporary.isAlive() = " + temporary.isAlive());
        System.out.println("temporary = " + temporary);
        try {
            temporary.getLong(0, 0);
        } catch (MemoryException e) {
            System.out.println("temporary.getLong(0, 0) -> MemoryException: " + e.getMessage());
        }

        // A confined scope belongs to its thread; a shared one may be used anywhere.
        try (NJScope confined = NJ.scope(); NJScope shared = NJ.scopeShared()) {
            NDArray c = confined.zeros(DType.INT32, 4);
            NDArray s = shared.zeros(DType.INT32, 4);
            System.out.println();
            System.out.println("shared array read from another thread: "
                    + CompletableFuture.supplyAsync(() -> s.getLong(0)).get());
            try {
                CompletableFuture.supplyAsync(() -> c.getLong(0)).get();
            } catch (ExecutionException e) {
                System.out.println("confined array read from another thread -> " + e.getCause().getClass().getSimpleName()
                        + ": " + e.getCause().getMessage());
            }
        } catch (ExecutionException e) {
            throw new IllegalStateException(e);
        }

        // Large automatic arrays live off-heap and are released by the garbage collector.
        NDArray big = NJ.zeros(DType.FLOAT64, 1 << 20);
        System.out.println();
        System.out.println("8 MiB automatic array is native: " + big.isNative());
        printStats("with one 8 MiB automatic array alive");
    }

    private static void printStats(String when) {
        MemoryStats s = NJ.memoryStats();
        System.out.printf("%n[%s] live native: %,d bytes, peak: %,d bytes, native allocations: %,d, heap threshold: %,d bytes%n",
                when, s.liveNativeBytes(), s.peakNativeBytes(), s.nativeAllocations(), s.heapThreshold());
    }
}
