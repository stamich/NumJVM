package org.numjvm.linalg;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;

/** Optional LP64 DOT binding for native, contiguous, homogeneous vectors. */
@SuppressWarnings("restricted")
final class NativeDot {
    private NativeDot() { }

    static boolean tryRun(NDArray a, NDArray b, NDArray out, ContractionPlan plan) {
        if (plan.scalarMultiply() || a.ndim() != 1 || b.ndim() != 1
                || a.dtype() != b.dtype() || plan.length() == 0 || plan.length() > Integer.MAX_VALUE
                || a.strides().stride(0) != a.dtype().size() || b.strides().stride(0) != b.dtype().size()) return false;
        var access = SharedSecrets.arrayAccess();
        var left = access.buffer(a).segment();
        var right = access.buffer(b).segment();
        if (!left.isNative() || !right.isNative() || OpenBlas.INSTANCE == null) return false;
        // Tiny calls stay in Java in auto mode; the threshold is provisional.
        if (OpenBlas.MODE.equals("auto") && plan.length() < 1024) return false;
        MethodHandle function = plan.dtype() == DType.FLOAT32 ? Bindings.FLOAT : Bindings.DOUBLE;
        if (function == null) {
            if (OpenBlas.MODE.equals("openblas")) throw new IllegalStateException("OpenBLAS DOT symbol unavailable");
            return false;
        }
        long bytes = Math.multiplyExact(plan.length(), plan.dtype().size());
        MemorySegment lp = left.asSlice(access.offset(a), bytes);
        MemorySegment rp = right.asSlice(access.offset(b), bytes);
        try {
            double result;
            if (plan.dtype() == DType.FLOAT32) result = (float) function.invokeExact((int) plan.length(), lp, 1, rp, 1);
            else result = (double) function.invokeExact((int) plan.length(), lp, 1, rp, 1);
            out.setDouble(result);
            BlasDispatch.dotNative();
            return true;
        } catch (Throwable error) {
            throw new IllegalStateException("OpenBLAS DOT failed", error);
        }
    }

    /** DOT symbols are optional and resolved separately from provider/GEMM initialization. */
    private static final class Bindings {
        static final MethodHandle FLOAT = bind("cblas_sdot", ValueLayout.JAVA_FLOAT);
        static final MethodHandle DOUBLE = bind("cblas_ddot", ValueLayout.JAVA_DOUBLE);
        private static MethodHandle bind(String name, MemoryLayout result) {
            return OpenBlas.INSTANCE.symbols.find(name).map(symbol -> Linker.nativeLinker().downcallHandle(symbol,
                    FunctionDescriptor.of(result, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT))).orElse(null);
        }
    }
}
