package org.numjvm.ufunc;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.UnaryOperator;

import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.LoopExecutor;
import org.numjvm.internal.loops.LoopSignature;
import org.numjvm.internal.loops.UnaryLoop;
import org.numjvm.ndarray.NDArray;

/**
 * A ufunc with one input, such as {@code negative} or {@code sqrt}.
 *
 * <p>The result has the input's shape and the dtype of the first loop the input's dtype
 * casts to safely (ADR-005): {@code sqrt} of an int8 array is float16, of an int32 array
 * float64, as in NumPy.
 */
public final class UnaryUfunc extends Ufunc {

    private final UnaryOperator<DType> loopOverride;
    private final Map<DType, String> unsupported;
    private final Resolution[] cache = new Resolution[DType.values().length];

    /**
     * The loop chosen for one input dtype.
     */
    record Resolution(DType in, DType out, UnaryLoop loop) {
    }

    private UnaryUfunc(Builder b) {
        super(b.name, b.loopName);
        this.loopOverride = b.loopOverride;
        this.unsupported = Map.copyOf(b.unsupported);
    }

    static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builds a unary ufunc.
     */
    static final class Builder {
        private final String name;
        private String loopName;
        private UnaryOperator<DType> loopOverride = t -> null;
        private final Map<DType, String> unsupported = new EnumMap<>(DType.class);

        private Builder(String name) {
            this.name = name;
            this.loopName = name;
        }

        Builder loops(String registryName) {
            this.loopName = registryName;
            return this;
        }

        Builder loopFor(UnaryOperator<DType> override) {
            this.loopOverride = override;
            return this;
        }

        Builder unsupported(DType t, String message) {
            unsupported.put(t, message);
            return this;
        }

        UnaryUfunc build() {
            return new UnaryUfunc(this);
        }
    }

    @Override
    public int nin() {
        return 1;
    }

    /**
     * Applies the ufunc element-wise.
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     * @throws DTypeException if the ufunc has no loop for the dtype
     */
    public NDArray apply(NDArray x) {
        Objects.requireNonNull(x, "x");
        Resolution r = resolve(x.dtype());
        NDArray in = Operands.cast(x, r.in());
        NDArray out = Operands.allocate(r.out(), in.shape(), in);
        LoopExecutor.unary(r.loop(), in, out);
        return out;
    }

    /**
     * Applies the ufunc and writes the result into {@code out} ({@code NUMPY_COMPATIBILITY}:
     * the {@code out=} argument); {@code ufunc.apply(x, x)} works in place. The rules for
     * {@code out} are those of {@link BinaryUfunc#apply(NDArray, NDArray, NDArray)}.
     *
     * @param x   the input
     * @param out the output, of the input's shape
     * @return {@code out}
     */
    public NDArray apply(NDArray x, NDArray out) {
        Objects.requireNonNull(x, "x");
        Resolution r = resolve(x.dtype());
        Outputs.check(name(), out, x.shape(), r.out());
        NDArray in = Operands.cast(x, r.in());
        if (out.dtype() == r.out() && !Outputs.unsafeOverlap(out, in)) {
            LoopExecutor.unary(r.loop(), in, out);
            return out;
        }
        NDArray tmp = Operands.allocate(r.out(), in.shape(), in);
        LoopExecutor.unary(r.loop(), in, tmp);
        return Outputs.copyInto(tmp, out);
    }

    /**
     * Returns the dtype of the result for an input of this dtype.
     *
     * @param x the input dtype
     * @return the result dtype
     * @throws DTypeException if the ufunc has no loop for the dtype
     */
    public DType resultType(DType x) {
        return resolve(x).out();
    }

    Resolution resolve(DType t) {
        Resolution r = cache[t.ordinal()];
        if (r != null) {
            return r;
        }
        String message = unsupported.get(t);
        if (message != null) {
            throw new DTypeException(message);
        }
        DType override = loopOverride.apply(t);
        LoopSignature sig = override != null ? KernelRegistry.find(loopName(), override) : firstSafeLoop(t);
        if (sig == null) {
            throw noLoop(t);
        }
        DType in = sig.inputs().getFirst();
        r = new Resolution(in, sig.output(), KernelRegistry.unary(loopName(), in));
        cache[t.ordinal()] = r;
        return r;
    }
}
