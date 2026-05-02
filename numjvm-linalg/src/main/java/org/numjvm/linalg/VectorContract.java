package org.numjvm.linalg;

import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.backend.Accelerators;
import org.numjvm.ndarray.NDArray;

/** Uses optional vector kernels only for homogeneous, contiguous vector pairs. */
final class VectorContract {
    private VectorContract() { }
    static boolean tryRun(NDArray a, NDArray b, NDArray out, ContractionPlan plan) {
        if (plan.scalarMultiply() || a.ndim() != 1 || b.ndim() != 1 || plan.length() < 128
                || a.dtype() != b.dtype() || a.strides().stride(0) != a.dtype().size()
                || b.strides().stride(0) != b.dtype().size()) return false;
        var loop = Accelerators.dot(plan.dtype());
        if (loop == null) return false;
        var access = SharedSecrets.arrayAccess();
        loop.run(access.buffer(a).segment(), access.offset(a), access.buffer(b).segment(), access.offset(b),
                access.buffer(out).segment(), access.offset(out), plan.length());
        BlasDispatch.dotVector();
        return true;
    }
}
