package org.numjvm.linalg;

import java.util.Objects;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;

/** Validate everything that can be checked before writing a user output. */
final class MatmulMemory {
    private MatmulMemory() { }
    static DType dtype(NDArray a,NDArray b) {
        for(NDArray x:new NDArray[]{a,b}) {
            if(x.dtype()!=DType.FLOAT32 && x.dtype()!=DType.FLOAT64) {
                throw new DTypeException("matmul supports only float32 and float64, got "+x.dtype());
            }
        }
        return a.dtype()==DType.FLOAT64 || b.dtype()==DType.FLOAT64?DType.FLOAT64:DType.FLOAT32;
    }
    static void accessible(NDArray a) {
        Objects.requireNonNull(a,"array");
        var memory=SharedSecrets.arrayAccess().buffer(a).segment();
        if(!memory.scope().isAlive()) throw new MemoryException("matmul operand scope is closed");
        if(!memory.isAccessibleBy(Thread.currentThread())) throw new MemoryException("matmul operand is confined to another thread");
    }
    static MemoryAllocator allocator(NDArray a,NDArray b) {
        for(NDArray x:new NDArray[]{a,b}) if(x.memoryOwner() instanceof NJScope scope) {
            return SharedSecrets.scopeAccess().allocator(scope);
        }
        return HybridAllocator.defaultInstance();
    }
    static NDArray allocate(NDArray a,NDArray b,MatmulPlan plan,DType type) {
        return SharedSecrets.arrayAccess().allocate(allocator(a,b),type,plan.output(),Order.C);
    }
    static void output(NDArray out,MatmulPlan plan,DType type) {
        accessible(out);
        if(!out.shape().equals(plan.output())) throw new ShapeException("matmul output must have shape "+plan.output());
        if(out.dtype()!=type) throw new DTypeException("matmul output must have exact dtype "+type);
        if(!out.isWriteable()) throw new ReadOnlyArrayException("matmul output is read-only");
    }
}
