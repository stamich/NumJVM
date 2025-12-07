package org.numjvm.ufunc;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.UnaryOperator;

import org.numjvm.dtype.DType;
import org.numjvm.dtype.TypePromotion;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.LoopExecutor;
import org.numjvm.internal.loops.LoopSignature;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.shape.Broadcasting;

/**
 * A ufunc with two inputs, such as {@code add} or {@code less}.
 *
 * <p>The inputs are broadcast against each other; the result has the broadcast shape and
 * the dtype of the selected loop (ADR-005). Java {@code long} and {@code double} arguments
 * are weak scalars: {@code add(uint8Array, 1L)} stays uint8, {@code add(int8Array, 1.5)}
 * becomes float64, and a {@code long} that does not fit an integer result dtype is a
 * {@link org.numjvm.exception.ValueException}.
 */
public final class BinaryUfunc extends Ufunc {

    private final UnaryOperator<DType> loopOverride;
    private final Map<DType, String> unsupported;
    private final Resolution[] cache = new Resolution[DType.values().length];

    /**
     * The loop chosen for one promoted dtype.
     */
    record Resolution(DType in, DType out, BinaryLoop loop) {
    }

    private BinaryUfunc(Builder b) {
        super(b.name, b.loopName);
        this.loopOverride = b.loopOverride;
        this.unsupported = Map.copyOf(b.unsupported);
    }

    static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builds a binary ufunc.
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

        /**
         * Uses the loops registered under another name.
         */
        Builder loops(String registryName) {
            this.loopName = registryName;
            return this;
        }

        /**
         * Overrides loop selection: maps a promoted dtype to the loop's input dtype, or
         * returns {@code null} for the default rule.
         */
        Builder loopFor(UnaryOperator<DType> override) {
            this.loopOverride = override;
            return this;
        }

        /**
         * Rejects a promoted dtype with a specific message.
         */
        Builder unsupported(DType t, String message) {
            unsupported.put(t, message);
            return this;
        }

        BinaryUfunc build() {
            return new BinaryUfunc(this);
        }
    }

    @Override
    public int nin() {
        return 2;
    }

    /**
     * Applies the ufunc element-wise.
     *
     * @param a first input
     * @param b second input
     * @return a new array with the broadcast shape
     * @throws org.numjvm.exception.ShapeException if the shapes cannot be broadcast
     * @throws DTypeException                      if the ufunc has no loop for the dtypes
     */
    public NDArray apply(NDArray a, NDArray b) {
        return run(a, b, TypePromotion.resultType(a.dtype(), b.dtype()));
    }

    /**
     * Applies the ufunc with a weak {@code double} second operand.
     *
     * @param a first input
     * @param b second input, a Java scalar
     * @return a new array with the shape of {@code a}
     */
    public NDArray apply(NDArray a, double b) {
        DType common = TypePromotion.resultTypeWithDouble(a.dtype());
        return run(a, Operands.weakDouble(common, b), common);
    }

    /**
     * Applies the ufunc with a weak {@code double} first operand.
     *
     * @param a first input, a Java scalar
     * @param b second input
     * @return a new array with the shape of {@code b}
     */
    public NDArray apply(double a, NDArray b) {
        DType common = TypePromotion.resultTypeWithDouble(b.dtype());
        return run(Operands.weakDouble(common, a), b, common);
    }

    /**
     * Applies the ufunc with a weak {@code long} second operand.
     *
     * @param a first input
     * @param b second input, a Java scalar
     * @return a new array with the shape of {@code a}
     * @throws org.numjvm.exception.ValueException if {@code b} does not fit an integer
     *                                             result dtype
     */
    public NDArray apply(NDArray a, long b) {
        DType common = TypePromotion.resultTypeWithLong(a.dtype());
        return run(a, Operands.weakLong(common, b), common);
    }

    /**
     * Applies the ufunc with a weak {@code long} first operand.
     *
     * @param a first input, a Java scalar
     * @param b second input
     * @return a new array with the shape of {@code b}
     * @throws org.numjvm.exception.ValueException if {@code a} does not fit an integer
     *                                             result dtype
     */
    public NDArray apply(long a, NDArray b) {
        DType common = TypePromotion.resultTypeWithLong(b.dtype());
        return run(Operands.weakLong(common, a), b, common);
    }

    /**
     * Returns the dtype of the result for inputs of these dtypes, without computing anything.
     *
     * @param a first input dtype
     * @param b second input dtype
     * @return the result dtype
     * @throws DTypeException if the ufunc has no loop for the dtypes
     */
    public DType resultType(DType a, DType b) {
        return resolve(TypePromotion.resultType(a, b), a, b).out();
    }

    private NDArray run(NDArray a, NDArray b, DType common) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Resolution r = resolve(common, a.dtype(), b.dtype());
        Shape shape = Broadcasting.broadcastShapes(a.shape(), b.shape());
        NDArray x = Operands.cast(a, r.in());
        NDArray y = Operands.cast(b, r.in());
        NDArray out = Operands.allocate(r.out(), shape, x, y);
        LoopExecutor.binary(r.loop(), x, y, out);
        return out;
    }

    Resolution resolve(DType common, DType a, DType b) {
        Resolution r = cache[common.ordinal()];
        if (r != null) {
            return r;
        }
        String message = unsupported.get(common);
        if (message != null) {
            throw new DTypeException(message);
        }
        DType override = loopOverride.apply(common);
        LoopSignature sig = override != null ? KernelRegistry.find(loopName(), override, override)
                : firstSafeLoop(common);
        if (sig == null) {
            throw noLoop(a, b);
        }
        DType in = sig.inputs().getFirst();
        r = new Resolution(in, sig.output(), KernelRegistry.binary(loopName(), in, in));
        cache[common.ordinal()] = r;
        return r;
    }
}
