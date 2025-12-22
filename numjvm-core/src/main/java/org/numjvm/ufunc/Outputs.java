package org.numjvm.ufunc;

import java.util.Arrays;
import java.util.Objects;

import org.numjvm.dtype.Casting;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.TypePromotion;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.LoopExecutor;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;

/**
 * Checks and writes caller-supplied output arrays ({@code out=}).
 *
 * <p>Rules, as in NumPy: the output must have exactly the result's shape, be writeable, and
 * the result dtype must convert to the output's dtype with {@link Casting#SAME_KIND}. An
 * output that overlaps an input other than as the identical view (same memory, offset and
 * strides) is written through a temporary, so {@code a[1:] += a[:-1]} sees the original
 * values of {@code a}.
 */
final class Outputs {

    private Outputs() {
    }

    static void check(String ufunc, NDArray out, Shape shape, DType result) {
        Objects.requireNonNull(out, "out");
        if (!out.shape().equals(shape)) {
            throw new ShapeException("output array with shape " + out.shape() + " does not match the result shape "
                    + shape);
        }
        if (!out.isWriteable()) {
            throw new ReadOnlyArrayException("output array is read-only");
        }
        if (!TypePromotion.canCast(result, out.dtype(), Casting.SAME_KIND)) {
            throw new DTypeException("cannot cast ufunc '" + ufunc + "' output from " + result.typeName() + " to "
                    + out.dtype().typeName() + " with casting rule 'same_kind'");
        }
    }

    /**
     * Whether writing {@code out} element by element could change elements of {@code in}
     * that are still to be read: they share memory and {@code in}, broadcast to the output's
     * shape, is not exactly the same view.
     */
    static boolean unsafeOverlap(NDArray out, NDArray in) {
        if (!in.sharesMemoryWith(out)) {
            return false;
        }
        SharedSecrets.ArrayAccess access = SharedSecrets.arrayAccess();
        if (access.buffer(in).segment() != access.buffer(out).segment()
                || access.offset(in) != access.offset(out) || in.dtype().size() != out.dtype().size()) {
            return true;
        }
        long[] st = Broadcasting.broadcastStrides(in.shape(), in.strides(), out.shape()).values();
        return !Arrays.equals(st, out.strides().values());
    }

    /**
     * Copies {@code src} into {@code out}, converting the dtype if needed.
     */
    static NDArray copyInto(NDArray src, NDArray out) {
        if (src.dtype() == out.dtype()) {
            LoopExecutor.unary(KernelRegistry.unary("copy", src.dtype()), src, out);
        } else {
            LoopExecutor.unary(KernelRegistry.unary("cast:" + out.dtype().typeName(), src.dtype()), src, out);
        }
        return out;
    }

    /**
     * Writes a fully computed result into {@code out} after checking it.
     */
    static NDArray store(String ufunc, NDArray result, NDArray out) {
        check(ufunc, out, result.shape(), result.dtype());
        return copyInto(result, out);
    }
}
