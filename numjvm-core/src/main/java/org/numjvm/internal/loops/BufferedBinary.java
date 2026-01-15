package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.MemoryErrors;
import org.numjvm.internal.memory.Scratch;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.iteration.StridedIterator;
import org.numjvm.ndarray.NDArray;

/** Converts only the current iteration run chunk; caller protects hazardous aliases. */
public final class BufferedBinary {
    private BufferedBinary() { }

    /** Executes a resolved operation without input-sized conversion allocations. */
    public static void run(BinaryLoop loop, DType inputType, NDArray a, NDArray b, NDArray out) {
        if (a.dtype() == inputType && b.dtype() == inputType) {
            LoopExecutor.binary(loop, a, b, out);
            return;
        }
        LoopExecutor.requireWriteable(out);
        var access = SharedSecrets.arrayAccess();
        var shape = out.shape();
        var plan = IterationPlan.of(shape.dimensions(),
                new long[][]{LoopExecutor.broadcast(a, shape), LoopExecutor.broadcast(b, shape), out.strides().values()},
                new long[]{access.offset(a), access.offset(b), access.offset(out)});
        if (!plan.hasElements()) { return; }
        MemorySegment ma = access.buffer(a).segment();
        MemorySegment mb = access.buffer(b).segment();
        MemorySegment mo = access.buffer(out).segment();
        UnaryLoop ca = a.dtype() == inputType ? null : KernelRegistry.unary("cast:" + inputType.typeName(), a.dtype());
        UnaryLoop cb = b.dtype() == inputType ? null : KernelRegistry.unary("cast:" + inputType.typeName(), b.dtype());
        int slot = Scratch.CHUNK_BYTES / 2;
        long capacity = slot / inputType.size();
        try (Scratch scratch = Scratch.lease(Scratch.CHUNK_BYTES)) {
            MemorySegment xa = scratch.memory().asSlice(0, slot);
            MemorySegment xb = scratch.memory().asSlice(slot, slot);
            StridedIterator it = plan.iterator();
            do {
                for (long i = 0; i < it.innerSize(); i += capacity) {
                    long n = Math.min(capacity, it.innerSize() - i);
                    long pa = it.offset(0) + i * it.innerStride(0);
                    long pb = it.offset(1) + i * it.innerStride(1);
                    if (ca != null) { ca.run(ma, pa, it.innerStride(0), xa, 0, inputType.size(), n); }
                    if (cb != null) { cb.run(mb, pb, it.innerStride(1), xb, 0, inputType.size(), n); }
                    loop.run(ca == null ? ma : xa, ca == null ? pa : 0, ca == null ? it.innerStride(0) : inputType.size(),
                            cb == null ? mb : xb, cb == null ? pb : 0, cb == null ? it.innerStride(1) : inputType.size(),
                            mo, it.offset(2) + i * it.innerStride(2), it.innerStride(2), n);
                }
            } while (it.next());
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, LoopExecutor.failedOwner(a, b, out));
        }
    }
}
