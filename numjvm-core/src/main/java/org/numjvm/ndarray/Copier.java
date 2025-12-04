package org.numjvm.ndarray;

import java.lang.foreign.MemorySegment;

import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.LoopExecutor;
import org.numjvm.internal.loops.UnaryLoop;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.internal.memory.MemoryErrors;

/**
 * Copies arrays into new contiguous storage, optionally converting the dtype.
 *
 * <p>A same-dtype copy of a source that is already contiguous in the requested order is
 * one bulk copy. Everything else runs the registry's {@code copy} or {@code cast:<dtype>}
 * loop over an iteration plan (ADR-004).
 */
final class Copier {

    private Copier() {
    }

    static NDArray copy(NDArray src, MemoryAllocator allocator, DType target, Order order) {
        if (src.dtype.isComplex() && !target.isComplex()) {
            throw new DTypeException("cannot convert " + src.dtype.typeName() + " to " + target.typeName()
                    + " without discarding the imaginary part");
        }
        NDArray dst = NDArray.allocate(allocator, target, src.shape, order);
        if (src.size() == 0) {
            return dst;
        }
        if (target == src.dtype && (order == Order.C ? src.isCContiguous() : src.isFContiguous())) {
            try {
                MemorySegment.copy(src.segment, src.offset, dst.segment, 0, src.nbytes());
            } catch (IllegalStateException | WrongThreadException e) {
                throw MemoryErrors.translate(e, src.buffer.owner());
            }
            return dst;
        }
        UnaryLoop loop = target == src.dtype
                ? KernelRegistry.unary("copy", target)
                : KernelRegistry.unary("cast:" + target.typeName(), src.dtype);
        LoopExecutor.unary(loop, src, dst);
        return dst;
    }
}
