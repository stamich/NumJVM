package org.numjvm.ufunc;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.UnaryOperator;

import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.DTypeKind;
import org.numjvm.dtype.TypePromotion;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.KernelRegistry;
import org.numjvm.internal.loops.LoopExecutor;
import org.numjvm.internal.loops.LoopSignature;
import org.numjvm.internal.memory.MemoryAllocator;
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
    private final boolean comparison;
    private final Long identity;
    private final boolean reorderable;
    private final boolean widensSmallIntegers;
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
        this.comparison = b.comparison;
        this.identity = b.identity;
        this.reorderable = b.reorderable;
        this.widensSmallIntegers = b.widensSmallIntegers;
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
        private boolean comparison;
        private Long identity;
        private boolean reorderable;
        private boolean widensSmallIntegers;

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

        /**
         * Marks a comparison: signed integers against uint64, and integers against an
         * out-of-range Java {@code long}, are compared exactly instead of after promotion.
         */
        Builder comparison() {
            this.comparison = true;
            return this;
        }

        /**
         * The reduction's starting value, stored with {@code setLong} ({@code -1} sets every bit).
         */
        Builder identity(long value) {
            this.identity = value;
            return this;
        }

        /**
         * The operation is associative and commutative, so several axes may be reduced at once;
         * without an identity it must also be idempotent (maximum, minimum).
         */
        Builder reorderable() {
            this.reorderable = true;
            return this;
        }

        /**
         * Reductions of bool and integers narrower than 64 bits run in int64 or uint64, as
         * NumPy's add and multiply.
         */
        Builder widensSmallIntegers() {
            this.widensSmallIntegers = true;
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
        if (comparison && signedWithUInt64(a.dtype(), b.dtype())) {
            return exact(a, b);
        }
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
        if (comparison && common.isInteger() && !Operands.fits(common, b)) {
            return exact(a, int64(b));
        }
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
        if (comparison && common.isInteger() && !Operands.fits(common, a)) {
            return exact(int64(a), b);
        }
        return run(Operands.weakLong(common, a), b, common);
    }

    /**
     * Applies the ufunc with a weak {@link Complex} second operand: complex arrays keep their
     * dtype, float32 and narrower floats become complex64, everything else complex128.
     *
     * @param a first input
     * @param b second input, a Java scalar
     * @return a new array with the shape of {@code a}
     */
    public NDArray apply(NDArray a, Complex b) {
        DType common = TypePromotion.resultTypeWithComplex(a.dtype());
        return run(a, Operands.weakComplex(common, b), common);
    }

    /**
     * Applies the ufunc with a weak {@link Complex} first operand; see
     * {@link #apply(NDArray, Complex)}.
     *
     * @param a first input, a Java scalar
     * @param b second input
     * @return a new array with the shape of {@code b}
     */
    public NDArray apply(Complex a, NDArray b) {
        DType common = TypePromotion.resultTypeWithComplex(b.dtype());
        return run(Operands.weakComplex(common, a), b, common);
    }

    /**
     * Applies the ufunc and writes the result into {@code out} ({@code NUMPY_COMPATIBILITY}:
     * the {@code out=} argument). {@code ufunc.apply(a, b, a)} is the in-place operation.
     *
     * <p>{@code out} must have exactly the broadcast shape and be writeable, and the result
     * dtype must convert to its dtype with {@code SAME_KIND} casting: an int8 array can
     * receive an int8 or int16 sum but not a float64 one. When {@code out} has the result
     * dtype and does not overlap the inputs (other than being one of them, viewed
     * identically), the loop writes into it directly without allocating; otherwise the result
     * goes through a temporary.
     *
     * @param a   first input
     * @param b   second input
     * @param out the output
     * @return {@code out}
     * @throws org.numjvm.exception.ShapeException         if {@code out} has another shape
     * @throws org.numjvm.exception.ReadOnlyArrayException if {@code out} is read-only
     * @throws DTypeException                              if the result dtype cannot be cast
     *                                                     to {@code out}'s dtype
     */
    public NDArray apply(NDArray a, NDArray b, NDArray out) {
        if (comparison && signedWithUInt64(a.dtype(), b.dtype())) {
            return Outputs.store(name(), exact(a, b), out);
        }
        return runInto(a, b, TypePromotion.resultType(a.dtype(), b.dtype()), out);
    }

    /**
     * Applies the ufunc with a weak {@code double} second operand into {@code out}; see
     * {@link #apply(NDArray, NDArray, NDArray)}.
     *
     * @param a   first input
     * @param b   second input, a Java scalar
     * @param out the output
     * @return {@code out}
     */
    public NDArray apply(NDArray a, double b, NDArray out) {
        DType common = TypePromotion.resultTypeWithDouble(a.dtype());
        return runInto(a, Operands.weakDouble(common, b), common, out);
    }

    /**
     * Applies the ufunc with a weak {@code long} second operand into {@code out}; see
     * {@link #apply(NDArray, NDArray, NDArray)}.
     *
     * @param a   first input
     * @param b   second input, a Java scalar
     * @param out the output
     * @return {@code out}
     */
    public NDArray apply(NDArray a, long b, NDArray out) {
        DType common = TypePromotion.resultTypeWithLong(a.dtype());
        if (comparison && common.isInteger() && !Operands.fits(common, b)) {
            return Outputs.store(name(), exact(a, int64(b)), out);
        }
        return runInto(a, Operands.weakLong(common, b), common, out);
    }

    /**
     * Applies the ufunc with a weak {@link Complex} second operand into {@code out}; see
     * {@link #apply(NDArray, NDArray, NDArray)}.
     *
     * @param a   first input
     * @param b   second input, a Java scalar
     * @param out the output
     * @return {@code out}
     */
    public NDArray apply(NDArray a, Complex b, NDArray out) {
        DType common = TypePromotion.resultTypeWithComplex(a.dtype());
        return runInto(a, Operands.weakComplex(common, b), common, out);
    }

    private NDArray runInto(NDArray a, NDArray b, DType common, NDArray out) {
        Resolution r = resolve(common, a.dtype(), b.dtype());
        Shape shape = Broadcasting.broadcastShapes(a.shape(), b.shape());
        Outputs.check(name(), out, shape, r.out());
        MemoryAllocator allocator = Operands.allocator(out, a, b);
        if (out.dtype() == r.out() && !Outputs.unsafeOverlap(out, a) && !Outputs.unsafeOverlap(out, b)) {
            org.numjvm.internal.loops.BufferedBinary.run(r.loop(), r.in(), a, b, out);
            return out;
        }
        NDArray x = Operands.cast(a, r.in(), allocator);
        NDArray y = Operands.cast(b, r.in(), allocator);
        if (out.dtype() == r.out() && !Outputs.unsafeOverlap(out, x) && !Outputs.unsafeOverlap(out, y)) {
            LoopExecutor.binary(r.loop(), x, y, out);
            return out;
        }
        NDArray tmp = Operands.allocate(allocator, r.out(), shape, x, y);
        LoopExecutor.binary(r.loop(), x, y, tmp);
        return Outputs.copyInto(tmp, out);
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

    // ------------------------------------------------------------------ reduce

    /**
     * Reduces over every axis ({@code NUMPY_COMPATIBILITY}: {@code ufunc.reduce} with
     * {@code axis=None}).
     *
     * @param x the array
     * @return a 0-D array
     * @throws DTypeException if this ufunc cannot reduce (comparisons) or has no loop
     * @throws org.numjvm.exception.ValueException for an empty array when the ufunc has no
     *                                             identity
     */
    public NDArray reduce(NDArray x) {
        return reduce(x, null, false);
    }

    /**
     * Reduces over one axis.
     *
     * @param x    the array
     * @param axis the axis; negative values count from the end
     * @return an array with that axis removed
     */
    public NDArray reduce(NDArray x, int axis) {
        return reduce(x, new int[]{axis}, false);
    }

    /**
     * Reduces over several axes ({@code NUMPY_COMPATIBILITY}: {@code ufunc.reduce}).
     *
     * <p>A ufunc with an identity (add: 0, multiply: 1, logical and: true, ...) starts from
     * it, so an empty reduction gives the identity. One without (maximum, minimum) starts from
     * the first element and an empty reduction is a {@link org.numjvm.exception.ValueException}.
     * Operations that are not associative (subtract, divide, ...) reduce at most one axis, in
     * index order. add and multiply reduce bool and integers narrower than 64 bits in int64
     * (uint64 for unsigned), as NumPy does.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the reduced array
     * @throws org.numjvm.exception.ShapeException for an axis out of range or repeated
     */
    public NDArray reduce(NDArray x, int[] axes, boolean keepdims) {
        java.util.Objects.requireNonNull(x, "x");
        if (comparison) {
            throw new DTypeException("ufunc '" + name() + "' cannot reduce");
        }
        boolean[] reduced = Reducer.reducedAxes(axes, x.ndim());
        DType t = x.dtype();
        if (widensSmallIntegers && (t.isBool() || t.isInteger() && t.size() < 8)) {
            t = t.isUnsigned() ? DType.UINT64 : DType.INT64;
        }
        Resolution r = resolve(t, t, t);
        if (r.in() != r.out()) {
            throw Reducer.notReducible(name());
        }
        MemoryAllocator allocator = Operands.allocator(x);
        if (widensSmallIntegers && x.dtype() != r.in()) {
            return Reducer.reduceAs(name(), org.numjvm.internal.loops.Widening.loop(x.dtype(), name().equals("multiply")),
                    x, reduced, keepdims, identity, reorderable, allocator, r.in());
        }
        return Reducer.reduce(name(), r.loop(), Operands.cast(x, r.in(), allocator), reduced, keepdims, identity,
                reorderable, allocator);
    }

    /**
     * Reduces into {@code out}; see {@link #reduce(NDArray, int[], boolean)} and, for the rules
     * on {@code out}, {@link #apply(NDArray, NDArray, NDArray)}.
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @param out      the output, of the reduced shape
     * @return {@code out}
     */
    public NDArray reduce(NDArray x, int[] axes, boolean keepdims, NDArray out) {
        boolean[] reduced = Reducer.reducedAxes(axes, x.ndim());
        if (comparison) { throw new DTypeException("ufunc '" + name() + "' cannot reduce"); }
        DType t = x.dtype();
        if (widensSmallIntegers && (t.isBool() || t.isInteger() && t.size() < 8)) {
            t = t.isUnsigned() ? DType.UINT64 : DType.INT64;
        }
        Resolution r = resolve(t, t, t);
        if (r.in() != r.out()) { throw Reducer.notReducible(name()); }
        Outputs.check(name(), out, Reducer.outputShape(x, reduced, keepdims), r.out());
        if (out.dtype() == r.out() && out.isCContiguous() && !out.sharesMemoryWith(x)
                && (x.dtype() == r.in() || widensSmallIntegers)) {
            BinaryLoop loop = x.dtype() == r.in() ? r.loop()
                    : org.numjvm.internal.loops.Widening.loop(x.dtype(), name().equals("multiply"));
            Reducer.reduceAs(name(), loop, x, reduced, keepdims, identity, reorderable,
                    Operands.allocator(out, x), r.out(), out);
            return out;
        }
        return Outputs.store(name(), reduce(x, axes, keepdims), out);
    }

    /**
     * Returns this ufunc's identity, the value an empty reduction gives.
     *
     * @return the identity as stored with {@code setLong}, or empty if there is none
     */
    public java.util.OptionalLong identity() {
        return identity == null ? java.util.OptionalLong.empty() : java.util.OptionalLong.of(identity);
    }

    private static boolean signedWithUInt64(DType a, DType b) {
        return a == DType.UINT64 && b.kind() == DTypeKind.SIGNED_INTEGER
                || b == DType.UINT64 && a.kind() == DTypeKind.SIGNED_INTEGER;
    }

    private static NDArray int64(long v) {
        NDArray s = org.numjvm.NJ.zeros(DType.INT64);
        s.setLong(v);
        return s;
    }

    /**
     * An exact integer comparison: uint64 against a signed integer through the mixed
     * int64/uint64 loops, every other integer pair through int64.
     */
    private NDArray exact(NDArray a, NDArray b) {
        DType ta = a.dtype() == DType.UINT64 ? DType.UINT64 : DType.INT64;
        DType tb = b.dtype() == DType.UINT64 ? DType.UINT64 : DType.INT64;
        MemoryAllocator allocator = Operands.allocator(a, b);
        NDArray x = Operands.cast(a, ta, allocator);
        NDArray y = Operands.cast(b, tb, allocator);
        BinaryLoop loop = KernelRegistry.binary(loopName(), ta, tb);
        Shape shape = Broadcasting.broadcastShapes(a.shape(), b.shape());
        NDArray out = Operands.allocate(allocator, DType.BOOL, shape, x, y);
        LoopExecutor.binary(loop, x, y, out);
        return out;
    }

    private NDArray run(NDArray a, NDArray b, DType common) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Resolution r = resolve(common, a.dtype(), b.dtype());
        Shape shape = Broadcasting.broadcastShapes(a.shape(), b.shape());
        MemoryAllocator allocator = Operands.allocator(a, b);
        NDArray out = Operands.allocate(allocator, r.out(), shape, a, b);
        org.numjvm.internal.loops.BufferedBinary.run(r.loop(), r.in(), a, b, out);
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
