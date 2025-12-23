package org.numjvm.ndarray;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import org.numjvm.dtype.DType;
import org.numjvm.exception.IndexException;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.internal.memory.MemoryErrors;
import org.numjvm.internal.memory.Sizes;
import org.numjvm.memory.MemoryOwner;
import org.numjvm.ufunc.Reductions;
import org.numjvm.ufunc.Rounding;
import org.numjvm.ufunc.Ufuncs;

/**
 * An N-dimensional array: a typed, strided view of a block of memory.
 *
 * <p>An array is defined by its buffer, dtype, the byte offset of its first element,
 * its shape and its byte strides. The metadata is immutable; operations such as
 * transpose, slice, reshape and broadcast return new arrays (views) that share the
 * buffer. The elements may be written unless the array is read-only.
 *
 * <h2>Invariants</h2>
 * Every instance satisfies, checked once at construction:
 * <ol>
 *   <li>every dimension is {@code >= 0} ({@link Shape});</li>
 *   <li>shape and strides have the same rank;</li>
 *   <li>the dtype is not {@code null};</li>
 *   <li>every element lies inside the buffer: the {@linkplain Strides#extent byte extent}
 *       is within {@code [0, buffer size)};</li>
 *   <li>no size or offset computation overflows {@code long};</li>
 *   <li>the buffer's memory was alive when the array was created, and the array keeps
 *       the buffer reachable for as long as the array itself is;</li>
 *   <li>an array with a zero stride on an axis longer than 1 (a broadcast view) is
 *       read-only.</li>
 * </ol>
 * Because of (4), element access only checks each index against its axis length.
 *
 * <h2>Element access</h2>
 * Accessors exist for {@code double}, {@code long} and {@code boolean}, each for rank 0
 * to 3 with one index argument per axis and, for any rank, with a {@code long[]}
 * index (ADR-002). Setters take the value last. Indices must lie in
 * {@code [0, dim)}; negative indices are not wrapped (use {@link Slice} for that).
 * Values are converted as follows:
 * <table>
 *   <caption>Conversion rules</caption>
 *   <tr><th>Accessor</th><th>Rule</th></tr>
 *   <tr><td>{@code getDouble}</td><td>any real dtype; bool reads as 0 or 1, uint64 as
 *       its unsigned value, float16 and bfloat16 are widened exactly</td></tr>
 *   <tr><td>{@code getLong}</td><td>bool and integer dtypes; uint64 returns the raw
 *       bits (see {@link org.numjvm.dtype.Unsigned}); floating-point dtypes throw</td></tr>
 *   <tr><td>{@code getBoolean}</td><td>any real dtype; {@code true} for any non-zero
 *       value, including NaN</td></tr>
 *   <tr><td>{@code setDouble}</td><td>integer dtypes truncate toward zero and wrap to
 *       the dtype's width; bool stores {@code value != 0}; float types round once to
 *       nearest (float16 and bfloat16 directly from the {@code double})</td></tr>
 *   <tr><td>{@code setLong}</td><td>integer dtypes keep the low bits; float types round
 *       to nearest; bool stores {@code value != 0}</td></tr>
 *   <tr><td>{@code setBoolean}</td><td>stores 1 or 0 in any real dtype</td></tr>
 * </table>
 * Complex dtypes are storage-only in this release; their accessors throw
 * {@link org.numjvm.exception.DTypeException}. Writing to a read-only array throws
 * {@link ReadOnlyArrayException}; accessing an array whose scope is closed, or a
 * confined scope's array from another thread, throws {@link MemoryException}.
 */
public final class NDArray {

    static {
        SharedSecrets.setArrayAccess(new SharedSecrets.ArrayAccess() {
            @Override
            public NDArray create(Buffer buffer, DType dtype, long offset, Shape shape, Strides strides,
                                  boolean writeable) {
                return new NDArray(buffer, dtype, offset, shape, strides, writeable);
            }

            @Override
            public NDArray allocate(MemoryAllocator allocator, DType dtype, Shape shape, Order order) {
                return NDArray.allocate(allocator, dtype, shape, order);
            }

            @Override
            public NDArray convert(NDArray array, MemoryAllocator allocator, DType target, Order order) {
                return Copier.copy(array, allocator, target, order);
            }

            @Override
            public Buffer buffer(NDArray array) {
                return array.buffer;
            }

            @Override
            public long offset(NDArray array) {
                return array.offset;
            }
        });
    }

    final Buffer buffer;
    final MemorySegment segment;
    final DType dtype;
    final long offset;
    final Shape shape;
    final Strides strides;
    final boolean writeable;
    final int flags;

    NDArray(Buffer buffer, DType dtype, long offset, Shape shape, Strides strides, boolean writeable) {
        this.buffer = Objects.requireNonNull(buffer, "buffer");
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.shape = Objects.requireNonNull(shape, "shape");
        this.strides = Objects.requireNonNull(strides, "strides");
        this.segment = buffer.segment();
        this.offset = offset;
        if (shape.rank() != strides.rank()) {
            throw new ShapeException("shape " + shape + " has rank " + shape.rank() + " but strides " + strides
                    + " have rank " + strides.rank());
        }
        ByteExtent extent = Strides.extent(shape.dims, strides.values, offset, dtype.size());
        if (!extent.isEmpty() && (extent.start() < 0 || extent.end() > buffer.byteSize())) {
            throw new ShapeException("view with shape " + shape + ", strides " + strides + " and offset " + offset
                    + " reaches bytes [" + extent.start() + ", " + extent.end() + ") outside its "
                    + buffer.byteSize() + "-byte buffer");
        }
        if (writeable && hasBroadcastAxis(shape.dims, strides.values)) {
            throw new ShapeException("an array with a zero stride on an axis longer than 1 must be read-only");
        }
        if (!buffer.isAlive()) {
            throw new MemoryException(buffer.owner().describe() + " is closed; cannot create an array on its memory");
        }
        this.writeable = writeable;
        this.flags = ArrayFlags.compute(shape.dims, strides.values, offset, dtype.size(), dtype.alignment(),
                buffer.baseAlignment(), writeable);
    }

    private static boolean hasBroadcastAxis(long[] dims, long[] strides) {
        for (int i = 0; i < dims.length; i++) {
            if (strides[i] == 0 && dims[i] > 1) {
                return true;
            }
        }
        return false;
    }

    /**
     * Allocates a zero-filled contiguous array.
     */
    static NDArray allocate(MemoryAllocator allocator, DType dtype, Shape shape, Order order) {
        long bytes = Sizes.bytesFor(shape.size(), dtype.size());
        Strides strides = order == Order.F
                ? Strides.fContiguous(shape, dtype.size())
                : Strides.cContiguous(shape, dtype.size());
        return new NDArray(allocator.allocate(bytes), dtype, 0, shape, strides, true);
    }

    // ------------------------------------------------------------------ metadata

    /**
     * Returns the element type.
     *
     * @return the element type
     */
    public DType dtype() {
        return dtype;
    }

    /**
     * Returns the shape, e.g. {@code [2, 3]}.
     *
     * @return the shape, e.g. {@code [2, 3]}
     */
    public Shape shape() {
        return shape;
    }

    /**
     * Returns the byte strides, e.g. {@code [24, 8]} for a C-contiguous 2×3 float64 array.
     *
     * @return the byte strides, e.g. {@code [24, 8]} for a C-contiguous 2×3 float64 array
     */
    public Strides strides() {
        return strides;
    }

    /**
     * Returns the number of dimensions (0 for a scalar array).
     *
     * @return the number of dimensions (0 for a scalar array)
     */
    public int ndim() {
        return shape.dims.length;
    }

    /**
     * Returns the number of elements.
     *
     * @return the number of elements
     */
    public long size() {
        return shape.size();
    }

    /**
     * Returns the length of one axis.
     *
     * @param axis the axis; negative values count from the end
     * @return the dimension
     */
    public long dim(int axis) {
        return shape.dimension(axis);
    }

    /**
     * Returns the size of one element in bytes.
     *
     * @return the size of one element in bytes
     */
    public int itemSize() {
        return dtype.size();
    }

    /**
     * Returns the number of bytes the elements occupy: {@code size() * itemSize()}.
     *
     * @return the number of bytes the elements occupy: {@code size() * itemSize()}
     */
    public long nbytes() {
        return shape.size() * dtype.size();
    }

    /**
     * Returns {@code true} if elements may be written through this array.
     *
     * @return {@code true} if elements may be written through this array
     */
    public boolean isWriteable() {
        return writeable;
    }

    /**
     * Returns layout and access flags (contiguity, writeability, alignment).
     *
     * @return layout and access flags (contiguity, writeability, alignment)
     */
    public ArrayFlags flags() {
        return ArrayFlags.of(flags);
    }

    /**
     * Returns {@code true} if the array is C-contiguous.
     *
     * @return {@code true} if the array is C-contiguous
     */
    public boolean isCContiguous() {
        return (flags & ArrayFlags.C_CONTIGUOUS) != 0;
    }

    /**
     * Returns {@code true} if the array is Fortran-contiguous.
     *
     * @return {@code true} if the array is Fortran-contiguous
     */
    public boolean isFContiguous() {
        return (flags & ArrayFlags.F_CONTIGUOUS) != 0;
    }

    /**
     * Returns the owner that decides how long this array's memory stays valid.
     *
     * @return the owner that decides how long this array's memory stays valid
     */
    public MemoryOwner memoryOwner() {
        return buffer.owner();
    }

    /**
     * Returns {@code true} if the array's memory may still be accessed.
     *
     * @return {@code true} if the array's memory may still be accessed
     */
    public boolean isAlive() {
        return buffer.isAlive();
    }

    /**
     * Returns {@code true} if the elements live in native (off-heap) memory.
     *
     * @return {@code true} if the elements live in native (off-heap) memory
     */
    public boolean isNative() {
        return buffer.isNative();
    }

    /**
     * Formats the array like NumPy's {@code repr}, e.g.
     * <pre>
     * array([[1., 2., 3.],
     *        [4., 5., 6.]])
     * </pre>
     * Arrays with more than 1000 elements are summarised with {@code ...}. An array whose
     * memory is no longer accessible prints its shape and dtype instead of its values.
     *
     * @return the formatted array
     */
    @Override
    public String toString() {
        return ArrayFormatter.repr(this);
    }

    // ------------------------------------------------------------------ views

    /**
     * Tells whether this array and {@code other} may refer to the same bytes
     * ({@code NUMPY_COMPATIBILITY}: {@code np.may_share_memory}).
     *
     * <p>The test compares the byte ranges the two arrays can touch, so it is exact for
     * contiguous blocks and conservative for interleaved views: {@code a[::2]} and
     * {@code a[1::2]} report {@code true} although no element is shared, while
     * {@code a[:5]} and {@code a[5:]} report {@code false}. Arrays without elements never
     * share memory.
     *
     * @param other another array
     * @return {@code true} if the byte ranges of the two arrays overlap
     */
    public boolean sharesMemoryWith(NDArray other) {
        ByteExtent mine = Strides.extent(shape.dims, strides.values, offset, dtype.size());
        ByteExtent theirs = Strides.extent(other.shape.dims, other.strides.values, other.offset, other.dtype.size());
        if (mine.isEmpty() || theirs.isEmpty()) {
            return false;
        }
        MemorySegment s1 = segment;
        MemorySegment s2 = other.segment;
        if (s1.isNative() != s2.isNative()) {
            return false;
        }
        if (!s1.isNative()) {
            Object base1 = s1.heapBase().orElse(null);
            if (base1 == null || base1 != s2.heapBase().orElse(null)) {
                return false;
            }
        }
        // address() is the absolute address for native segments and the offset into the
        // backing array for heap segments.
        long a1 = s1.address();
        long a2 = s2.address();
        return new ByteExtent(a1 + mine.start(), a1 + mine.end())
                .overlaps(new ByteExtent(a2 + theirs.start(), a2 + theirs.end()));
    }

    /**
     * A view on the same buffer; the caller hands over ownership of both arrays.
     */
    private NDArray view(long newOffset, long[] dims, long[] newStrides, boolean newWriteable) {
        return new NDArray(buffer, dtype, newOffset, Shape.ofOwned(dims), Strides.ofOwned(newStrides), newWriteable);
    }

    /**
     * Returns a view with the axes reversed ({@code a.T} in NumPy). For a matrix this is
     * the transpose: shape {@code [2, 3]} with strides {@code [24, 8]} becomes shape
     * {@code [3, 2]} with strides {@code [8, 24]}. No data is copied.
     *
     * @return the transposed view; the same as this array's layout for rank 0 and 1
     */
    public NDArray transpose() {
        int n = ndim();
        int[] axes = new int[n];
        for (int i = 0; i < n; i++) {
            axes[i] = n - 1 - i;
        }
        return permuteDims(axes);
    }

    /**
     * Returns a view with the axes reordered ({@code ARRAY_API}: {@code permute_dims}).
     * Axis {@code i} of the result is axis {@code axes[i]} of this array. No data is copied.
     *
     * @param axes a permutation of {@code 0 .. ndim-1}; negative values count from the end
     * @return the view
     * @throws ShapeException if {@code axes} is not a permutation of the axes
     */
    public NDArray permuteDims(int... axes) {
        int n = ndim();
        if (axes.length != n) {
            throw new ShapeException("permutation " + java.util.Arrays.toString(axes) + " has " + axes.length
                    + " axes, array has " + n);
        }
        long[] dims = new long[n];
        long[] st = new long[n];
        boolean[] used = new boolean[n];
        for (int i = 0; i < n; i++) {
            int ax = Shape.normalizeAxis(axes[i], n);
            if (used[ax]) {
                throw new ShapeException("axis " + ax + " repeated in permutation " + java.util.Arrays.toString(axes));
            }
            used[ax] = true;
            dims[i] = shape.dims[ax];
            st[i] = strides.values[ax];
        }
        return view(offset, dims, st, writeable);
    }

    /**
     * Returns a view with two axes exchanged. No data is copied.
     *
     * @param axis1 first axis; negative values count from the end
     * @param axis2 second axis
     * @return the view
     */
    public NDArray swapAxes(int axis1, int axis2) {
        int n = ndim();
        int a = Shape.normalizeAxis(axis1, n);
        int b = Shape.normalizeAxis(axis2, n);
        int[] axes = new int[n];
        for (int i = 0; i < n; i++) {
            axes[i] = i;
        }
        axes[a] = b;
        axes[b] = a;
        return permuteDims(axes);
    }

    /**
     * Returns a read-only view that repeats this array to fill {@code shape}
     * ({@code ARRAY_API}: {@code broadcast_to}). Stretched and new axes get stride 0, so no
     * data is copied: broadcasting {@code [1, 2, 3]} to {@code [4, 3]} gives strides
     * {@code [0, 8]}.
     *
     * <p>The view is always read-only: a write through a zero stride would change many
     * logical elements at once. Copy it to get a writeable array.
     *
     * @param shape the target shape
     * @return the read-only view
     * @throws ShapeException if this shape cannot be broadcast to {@code shape}
     */
    public NDArray broadcastTo(long... shape) {
        Shape target = Shape.of(shape);
        Strides st = org.numjvm.shape.Broadcasting.broadcastStrides(this.shape, strides, target);
        return new NDArray(buffer, dtype, offset, target, st, false);
    }

    /**
     * Returns an array with the same elements in a new shape: a view when the layout
     * allows it, otherwise a C-contiguous copy ({@code NUMPY_COMPATIBILITY}: NumPy's
     * {@code reshape}). One dimension may be {@code -1} and is inferred.
     *
     * <p>Code that relies on writes reaching the original must use
     * {@link #reshapeView(long...)}, which never copies.
     *
     * @param newShape the new dimensions; their product must equal {@link #size()}
     * @return a view or a copy
     * @throws ShapeException if the element count differs or more than one dimension is -1
     */
    public NDArray reshape(long... newShape) {
        long[] dims = Reshaper.resolve(newShape, size());
        long[] st = Reshaper.noCopyStrides(shape.dims, strides.values, dims, dtype.size());
        if (st != null) {
            return view(offset, dims, st, writeable);
        }
        NDArray c = copy();
        return c.view(0, dims, Strides.cStrides(dims, dtype.size()), true);
    }

    /**
     * Returns a view with a new shape, never a copy ({@code NUMJVM_SPECIFIC}). One dimension
     * may be {@code -1}.
     *
     * <p>A view exists exactly when NumPy's {@code reshape} would return one: the axes of
     * this array that are merged or split must be contiguous with each other. A
     * C-contiguous array can always be reshaped this way; a transposed one usually not.
     *
     * @param newShape the new dimensions
     * @return the view
     * @throws ShapeException if the element count differs or no view with that shape exists
     */
    public NDArray reshapeView(long... newShape) {
        long[] dims = Reshaper.resolve(newShape, size());
        long[] st = Reshaper.noCopyStrides(shape.dims, strides.values, dims, dtype.size());
        if (st == null) {
            throw new ShapeException("cannot reshape an array with shape " + shape + " and strides " + strides
                    + " into " + java.util.Arrays.toString(dims) + " without copying; use reshape() to allow a copy");
        }
        return view(offset, dims, st, writeable);
    }

    /**
     * Returns a 1-D array of all elements in C order: a view when possible, else a copy.
     *
     * @return {@code reshape(-1)}
     */
    public NDArray ravel() {
        return reshape(-1);
    }

    /**
     * Returns a view selected by a basic NumPy index: single positions, ranges with
     * positive or negative steps, new axes and an ellipsis. Missing trailing components
     * select whole axes. No data is copied; see {@link Slice} for the NumPy equivalents.
     *
     * @param index the index components
     * @return the view; single positions remove their axis, new axes add one of length 1
     * @throws IndexException if there are more positional components than axes, more than
     *                        one ellipsis, or a single position is outside its axis
     */
    public NDArray slice(Slice... index) {
        int ndim = ndim();
        int consuming = 0;
        int ellipses = 0;
        int newAxes = 0;
        for (Slice s : index) {
            switch (s) {
                case Slice.Ellipsis _ -> ellipses++;
                case Slice.NewAxis _ -> newAxes++;
                default -> consuming++;
            }
        }
        if (ellipses > 1) {
            throw new IndexException("an index can only have a single ellipsis");
        }
        if (consuming > ndim) {
            throw new IndexException("too many indices for array: array is " + ndim + "-dimensional, but "
                    + consuming + " were indexed");
        }
        int fill = ndim - consuming;  // axes covered by the ellipsis or implicitly at the end
        long[] dims = new long[ndim - countIndices(index) + newAxes];
        long[] st = new long[dims.length];
        long off = offset;
        int src = 0;
        int dst = 0;
        boolean ellipsisSeen = false;
        for (Slice s : index) {
            switch (s) {
                case Slice.Ellipsis _ -> {
                    ellipsisSeen = true;
                    for (int k = 0; k < fill; k++, src++, dst++) {
                        dims[dst] = shape.dims[src];
                        st[dst] = strides.values[src];
                    }
                }
                case Slice.NewAxis _ -> {
                    dims[dst] = 1;
                    st[dst] = 0;
                    dst++;
                }
                case Slice.All _ -> {
                    dims[dst] = shape.dims[src];
                    st[dst] = strides.values[src];
                    src++;
                    dst++;
                }
                case Slice.Index i -> {
                    long n = shape.dims[src];
                    long p = i.index() < 0 ? i.index() + n : i.index();
                    if (p < 0 || p >= n) {
                        throw IndexException.outOfBounds(i.index(), src, n);
                    }
                    off += p * strides.values[src];
                    src++;
                }
                case Slice.Range r -> {
                    long n = shape.dims[src];
                    long[] sl = adjust(r, n);
                    dims[dst] = sl[2];
                    st[dst] = strides.values[src] * r.step();
                    if (sl[2] > 0) {
                        off += sl[0] * strides.values[src];
                    }
                    src++;
                    dst++;
                }
            }
        }
        if (!ellipsisSeen) {
            for (; src < ndim; src++, dst++) {
                dims[dst] = shape.dims[src];
                st[dst] = strides.values[src];
            }
        }
        return view(off, dims, st, writeable);
    }

    private static int countIndices(Slice[] index) {
        int n = 0;
        for (Slice s : index) {
            if (s instanceof Slice.Index) {
                n++;
            }
        }
        return n;
    }

    /**
     * Python's {@code PySlice_AdjustIndices}: returns {start, stop, length}.
     */
    static long[] adjust(Slice.Range r, long n) {
        long step = r.step();
        long start;
        long stop;
        if (r.start() == null) {
            start = step < 0 ? n - 1 : 0;
        } else {
            start = r.start();
            if (start < 0) {
                start += n;
                if (start < 0) {
                    start = step < 0 ? -1 : 0;
                }
            } else if (start >= n) {
                start = step < 0 ? n - 1 : n;
            }
        }
        if (r.stop() == null) {
            stop = step < 0 ? -1 : n;
        } else {
            stop = r.stop();
            if (stop < 0) {
                stop += n;
                if (stop < 0) {
                    stop = step < 0 ? -1 : 0;
                }
            } else if (stop >= n) {
                stop = step < 0 ? n - 1 : n;
            }
        }
        long length;
        if (step < 0) {
            length = stop < start ? (start - stop - 1) / -step + 1 : 0;
        } else {
            length = start < stop ? (stop - start - 1) / step + 1 : 0;
        }
        return new long[]{start, stop, length};
    }

    // ------------------------------------------------------------------ copies

    /**
     * Returns a C-contiguous copy with independent, automatically managed memory.
     *
     * <p>The copy is writeable even if this array is a read-only view, and it does not
     * depend on this array's scope.
     *
     * @return the copy
     */
    public NDArray copy() {
        return copy(Order.C);
    }

    /**
     * Returns a contiguous copy in the requested order.
     *
     * @param order memory order of the copy
     * @return the copy, with independent, automatically managed memory
     */
    public NDArray copy(Order order) {
        return Copier.copy(this, HybridAllocator.defaultInstance(), dtype, order);
    }

    /**
     * Returns a C-contiguous copy converted to another dtype ({@code NUMPY_COMPATIBILITY}:
     * NumPy's {@code astype} with {@code casting="unsafe"}).
     *
     * <p>Integer and bool conversions keep the low bits of the value (int64 and uint64 keep
     * all of them) and wrap to the target's width. Every conversion to a floating-point
     * type rounds once to nearest, so int64 to float32 and float64 to float16 are
     * correctly rounded as in NumPy. Floating-point values convert to integers by truncation
     * toward zero; NaN and out-of-range values follow Java's saturating conversion to
     * {@code long} and then wrap ({@code NUMJVM_SPECIFIC}: NumPy leaves them undefined).
     * Real values become complex with a zero imaginary part; converting complex to bool gives
     * {@code true} when either part is non-zero; converting complex to any other real dtype is
     * rejected rather than silently discarding the imaginary part.
     *
     * @param target the dtype of the result
     * @return the converted copy; a plain copy if {@code target} equals this dtype
     * @throws org.numjvm.exception.DTypeException for complex to real conversions other than bool
     */
    public NDArray astype(DType target) {
        return astype(target, Order.C);
    }

    /**
     * Same as {@link #astype(DType)} with an explicit memory order.
     *
     * @param target the dtype of the result
     * @param order  memory order of the result
     * @return the converted copy
     */
    public NDArray astype(DType target, Order order) {
        return Copier.copy(this, HybridAllocator.defaultInstance(), Objects.requireNonNull(target, "target"), order);
    }

    // ------------------------------------------------------------------ arithmetic

    /**
     * Returns {@code this + other}, element-wise with broadcasting
     * ({@link org.numjvm.ufunc.Ufuncs#ADD}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape and the promoted dtype
     */
    public NDArray add(NDArray other) {
        return Ufuncs.ADD.apply(this, other);
    }

    /**
     * Returns {@code this + other} with a weak scalar ({@link org.numjvm.ufunc.Ufuncs#ADD}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray add(double other) {
        return Ufuncs.ADD.apply(this, other);
    }

    /**
     * Returns {@code this + other} with a weak scalar ({@link org.numjvm.ufunc.Ufuncs#ADD}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray add(long other) {
        return Ufuncs.ADD.apply(this, other);
    }

    /**
     * Returns {@code this - other}, element-wise with broadcasting: not defined for two bool operands
     * ({@link org.numjvm.ufunc.Ufuncs#SUBTRACT}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray subtract(NDArray other) {
        return Ufuncs.SUBTRACT.apply(this, other);
    }

    /**
     * Returns {@code this - other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#SUBTRACT}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray subtract(double other) {
        return Ufuncs.SUBTRACT.apply(this, other);
    }

    /**
     * Returns {@code this - other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#SUBTRACT}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray subtract(long other) {
        return Ufuncs.SUBTRACT.apply(this, other);
    }

    /**
     * Returns {@code this * other}, element-wise with broadcasting: logical and for bool
     * ({@link org.numjvm.ufunc.Ufuncs#MULTIPLY}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray multiply(NDArray other) {
        return Ufuncs.MULTIPLY.apply(this, other);
    }

    /**
     * Returns {@code this * other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#MULTIPLY}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray multiply(double other) {
        return Ufuncs.MULTIPLY.apply(this, other);
    }

    /**
     * Returns {@code this * other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#MULTIPLY}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray multiply(long other) {
        return Ufuncs.MULTIPLY.apply(this, other);
    }

    /**
     * Returns {@code this / other}, element-wise with broadcasting: true division; integers give float64
     * ({@link org.numjvm.ufunc.Ufuncs#DIVIDE}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray divide(NDArray other) {
        return Ufuncs.DIVIDE.apply(this, other);
    }

    /**
     * Returns {@code this / other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#DIVIDE}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray divide(double other) {
        return Ufuncs.DIVIDE.apply(this, other);
    }

    /**
     * Returns {@code this / other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#DIVIDE}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray divide(long other) {
        return Ufuncs.DIVIDE.apply(this, other);
    }

    /**
     * Returns {@code floor(this / other)}, element-wise with broadcasting: integer division by zero gives 0
     * ({@link org.numjvm.ufunc.Ufuncs#FLOOR_DIVIDE}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray floorDivide(NDArray other) {
        return Ufuncs.FLOOR_DIVIDE.apply(this, other);
    }

    /**
     * Returns {@code floor(this / other)} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#FLOOR_DIVIDE}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray floorDivide(double other) {
        return Ufuncs.FLOOR_DIVIDE.apply(this, other);
    }

    /**
     * Returns {@code floor(this / other)} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#FLOOR_DIVIDE}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray floorDivide(long other) {
        return Ufuncs.FLOOR_DIVIDE.apply(this, other);
    }

    /**
     * Returns {@code this % other}, element-wise with broadcasting: Python-style remainder with the sign of the divisor
     * ({@link org.numjvm.ufunc.Ufuncs#REMAINDER}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray remainder(NDArray other) {
        return Ufuncs.REMAINDER.apply(this, other);
    }

    /**
     * Returns {@code this % other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#REMAINDER}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray remainder(double other) {
        return Ufuncs.REMAINDER.apply(this, other);
    }

    /**
     * Returns {@code this % other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#REMAINDER}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray remainder(long other) {
        return Ufuncs.REMAINDER.apply(this, other);
    }

    /**
     * Returns {@code this ** other}, element-wise with broadcasting: integer powers wrap
     * ({@link org.numjvm.ufunc.Ufuncs#POWER}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray power(NDArray other) {
        return Ufuncs.POWER.apply(this, other);
    }

    /**
     * Returns {@code this ** other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#POWER}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray power(double other) {
        return Ufuncs.POWER.apply(this, other);
    }

    /**
     * Returns {@code this ** other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#POWER}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray power(long other) {
        return Ufuncs.POWER.apply(this, other);
    }

    /**
     * Returns {@code max(this, other)}, element-wise with broadcasting: NaN propagates
     * ({@link org.numjvm.ufunc.Ufuncs#MAXIMUM}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray maximum(NDArray other) {
        return Ufuncs.MAXIMUM.apply(this, other);
    }

    /**
     * Returns {@code max(this, other)} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#MAXIMUM}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray maximum(double other) {
        return Ufuncs.MAXIMUM.apply(this, other);
    }

    /**
     * Returns {@code max(this, other)} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#MAXIMUM}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray maximum(long other) {
        return Ufuncs.MAXIMUM.apply(this, other);
    }

    /**
     * Returns {@code min(this, other)}, element-wise with broadcasting: NaN propagates
     * ({@link org.numjvm.ufunc.Ufuncs#MINIMUM}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray minimum(NDArray other) {
        return Ufuncs.MINIMUM.apply(this, other);
    }

    /**
     * Returns {@code min(this, other)} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#MINIMUM}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray minimum(double other) {
        return Ufuncs.MINIMUM.apply(this, other);
    }

    /**
     * Returns {@code min(this, other)} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#MINIMUM}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray minimum(long other) {
        return Ufuncs.MINIMUM.apply(this, other);
    }

    /**
     * Returns {@code this == other}, element-wise with broadcasting: giving bool
     * ({@link org.numjvm.ufunc.Ufuncs#EQUAL}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    @SuppressWarnings("java:S1221") // NumPy's name for the ufunc; not Object.equals
    public NDArray equal(NDArray other) {
        return Ufuncs.EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this == other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#EQUAL}).
     *
     * @param other the scalar
     * @return a new array
     */
    @SuppressWarnings("java:S1221") // NumPy's name for the ufunc; not Object.equals
    public NDArray equal(double other) {
        return Ufuncs.EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this == other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#EQUAL}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    @SuppressWarnings("java:S1221") // NumPy's name for the ufunc; not Object.equals
    public NDArray equal(long other) {
        return Ufuncs.EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this != other}, element-wise with broadcasting: giving bool
     * ({@link org.numjvm.ufunc.Ufuncs#NOT_EQUAL}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray notEqual(NDArray other) {
        return Ufuncs.NOT_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this != other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#NOT_EQUAL}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray notEqual(double other) {
        return Ufuncs.NOT_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this != other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#NOT_EQUAL}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray notEqual(long other) {
        return Ufuncs.NOT_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this < other}, element-wise with broadcasting: giving bool
     * ({@link org.numjvm.ufunc.Ufuncs#LESS}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray less(NDArray other) {
        return Ufuncs.LESS.apply(this, other);
    }

    /**
     * Returns {@code this < other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#LESS}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray less(double other) {
        return Ufuncs.LESS.apply(this, other);
    }

    /**
     * Returns {@code this < other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#LESS}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray less(long other) {
        return Ufuncs.LESS.apply(this, other);
    }

    /**
     * Returns {@code this <= other}, element-wise with broadcasting: giving bool
     * ({@link org.numjvm.ufunc.Ufuncs#LESS_EQUAL}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray lessEqual(NDArray other) {
        return Ufuncs.LESS_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this <= other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#LESS_EQUAL}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray lessEqual(double other) {
        return Ufuncs.LESS_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this <= other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#LESS_EQUAL}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray lessEqual(long other) {
        return Ufuncs.LESS_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this > other}, element-wise with broadcasting: giving bool
     * ({@link org.numjvm.ufunc.Ufuncs#GREATER}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray greater(NDArray other) {
        return Ufuncs.GREATER.apply(this, other);
    }

    /**
     * Returns {@code this > other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#GREATER}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray greater(double other) {
        return Ufuncs.GREATER.apply(this, other);
    }

    /**
     * Returns {@code this > other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#GREATER}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray greater(long other) {
        return Ufuncs.GREATER.apply(this, other);
    }

    /**
     * Returns {@code this >= other}, element-wise with broadcasting: giving bool
     * ({@link org.numjvm.ufunc.Ufuncs#GREATER_EQUAL}).
     *
     * @param other the other operand
     * @return a new array with the broadcast shape
     */
    public NDArray greaterEqual(NDArray other) {
        return Ufuncs.GREATER_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this >= other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#GREATER_EQUAL}).
     *
     * @param other the scalar
     * @return a new array
     */
    public NDArray greaterEqual(double other) {
        return Ufuncs.GREATER_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code this >= other} with a weak scalar
     * ({@link org.numjvm.ufunc.Ufuncs#GREATER_EQUAL}).
     *
     * @param other the scalar; it must fit an integer dtype
     * @return a new array
     */
    public NDArray greaterEqual(long other) {
        return Ufuncs.GREATER_EQUAL.apply(this, other);
    }

    /**
     * Returns {@code -this}, element-wise: integers wrap; not defined for bool
     * ({@link org.numjvm.ufunc.Ufuncs#NEGATIVE}).
     *
     * @return a new array with this array's shape
     */
    public NDArray negative() {
        return Ufuncs.NEGATIVE.apply(this);
    }

    /**
     * Returns {@code |this|}, element-wise: abs(MIN) wraps to MIN
     * ({@link org.numjvm.ufunc.Ufuncs#ABSOLUTE}).
     *
     * @return a new array with this array's shape
     */
    public NDArray abs() {
        return Ufuncs.ABSOLUTE.apply(this);
    }

    /**
     * Returns {@code sign(this)}, element-wise: -1, 0 or 1; NaN for NaN
     * ({@link org.numjvm.ufunc.Ufuncs#SIGN}).
     *
     * @return a new array with this array's shape
     */
    public NDArray sign() {
        return Ufuncs.SIGN.apply(this);
    }

    /**
     * Returns {@code this * this}, element-wise: integers wrap
     * ({@link org.numjvm.ufunc.Ufuncs#SQUARE}).
     *
     * @return a new array with this array's shape
     */
    public NDArray square() {
        return Ufuncs.SQUARE.apply(this);
    }

    /**
     * Returns {@code sqrt(this)}, element-wise: integers give the first float dtype they convert to safely
     * ({@link org.numjvm.ufunc.Ufuncs#SQRT}).
     *
     * @return a new array with this array's shape
     */
    public NDArray sqrt() {
        return Ufuncs.SQRT.apply(this);
    }

    /**
     * Returns {@code exp(this)}, element-wise: integers give the first float dtype they convert to safely
     * ({@link org.numjvm.ufunc.Ufuncs#EXP}).
     *
     * @return a new array with this array's shape
     */
    public NDArray exp() {
        return Ufuncs.EXP.apply(this);
    }

    /**
     * Returns {@code log(this)}, element-wise: integers give the first float dtype they convert to safely
     * ({@link org.numjvm.ufunc.Ufuncs#LOG}).
     *
     * @return a new array with this array's shape
     */
    public NDArray log() {
        return Ufuncs.LOG.apply(this);
    }

    /**
     * Returns {@code sin(this)}, element-wise: integers give the first float dtype they convert to safely
     * ({@link org.numjvm.ufunc.Ufuncs#SIN}).
     *
     * @return a new array with this array's shape
     */
    public NDArray sin() {
        return Ufuncs.SIN.apply(this);
    }

    /**
     * Returns {@code cos(this)}, element-wise: integers give the first float dtype they convert to safely
     * ({@link org.numjvm.ufunc.Ufuncs#COS}).
     *
     * @return a new array with this array's shape
     */
    public NDArray cos() {
        return Ufuncs.COS.apply(this);
    }

    /**
     * Returns {@code tanh(this)}, element-wise: integers give the first float dtype they convert to safely
     * ({@link org.numjvm.ufunc.Ufuncs#TANH}).
     *
     * @return a new array with this array's shape
     */
    public NDArray tanh() {
        return Ufuncs.TANH.apply(this);
    }

    /**
     * Returns {@code trunc(this)}, element-wise: integers keep their dtype
     * ({@link org.numjvm.ufunc.Ufuncs#TRUNC}).
     *
     * @return a new array with this array's shape
     */
    public NDArray trunc() {
        return Ufuncs.TRUNC.apply(this);
    }

    /**
     * Returns this array rounded to the nearest integer, halves to even
     * ({@link org.numjvm.ufunc.Rounding#round(NDArray)}).
     *
     * @return a new array; integer arrays keep their dtype
     */
    public NDArray round() {
        return Rounding.round(this);
    }

    /**
     * Returns this array rounded to {@code decimals} decimal places, halves to even
     * ({@link org.numjvm.ufunc.Rounding#round(NDArray, int)}).
     *
     * @param decimals decimal places; negative values round to tens, hundreds, ...
     * @return a new array of the same dtype
     */
    public NDArray round(int decimals) {
        return Rounding.round(this, decimals);
    }

    /**
     * Returns {@code floor(this)}, element-wise: integers keep their dtype
     * ({@link org.numjvm.ufunc.Ufuncs#FLOOR}).
     *
     * @return a new array with this array's shape
     */
    public NDArray floor() {
        return Ufuncs.FLOOR.apply(this);
    }

    /**
     * Returns {@code ceil(this)}, element-wise: integers keep their dtype
     * ({@link org.numjvm.ufunc.Ufuncs#CEIL}).
     *
     * @return a new array with this array's shape
     */
    public NDArray ceil() {
        return Ufuncs.CEIL.apply(this);
    }

    /**
     * Sum over all elements ({@link org.numjvm.ufunc.Reductions#sum(NDArray)}).
     *
     * @return a 0-D array
     */
    public NDArray sum() {
        return Reductions.sum(this);
    }

    /**
     * Sum over one axis ({@link org.numjvm.ufunc.Reductions#sum(NDArray, int)}).
     *
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public NDArray sum(int axis) {
        return Reductions.sum(this, axis);
    }

    /**
     * Sum over several axes
     * ({@link org.numjvm.ufunc.Reductions#sum(NDArray, int[], boolean)}).
     *
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public NDArray sum(int[] axes, boolean keepdims) {
        return Reductions.sum(this, axes, keepdims);
    }

    /**
     * Product over all elements ({@link org.numjvm.ufunc.Reductions#prod(NDArray)}).
     *
     * @return a 0-D array
     */
    public NDArray prod() {
        return Reductions.prod(this);
    }

    /**
     * Product over one axis ({@link org.numjvm.ufunc.Reductions#prod(NDArray, int)}).
     *
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public NDArray prod(int axis) {
        return Reductions.prod(this, axis);
    }

    /**
     * Product over several axes
     * ({@link org.numjvm.ufunc.Reductions#prod(NDArray, int[], boolean)}).
     *
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public NDArray prod(int[] axes, boolean keepdims) {
        return Reductions.prod(this, axes, keepdims);
    }

    /**
     * Maximum over all elements ({@link org.numjvm.ufunc.Reductions#max(NDArray)}).
     *
     * @return a 0-D array
     */
    public NDArray max() {
        return Reductions.max(this);
    }

    /**
     * Maximum over one axis ({@link org.numjvm.ufunc.Reductions#max(NDArray, int)}).
     *
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public NDArray max(int axis) {
        return Reductions.max(this, axis);
    }

    /**
     * Maximum over several axes
     * ({@link org.numjvm.ufunc.Reductions#max(NDArray, int[], boolean)}).
     *
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public NDArray max(int[] axes, boolean keepdims) {
        return Reductions.max(this, axes, keepdims);
    }

    /**
     * Minimum over all elements ({@link org.numjvm.ufunc.Reductions#min(NDArray)}).
     *
     * @return a 0-D array
     */
    public NDArray min() {
        return Reductions.min(this);
    }

    /**
     * Minimum over one axis ({@link org.numjvm.ufunc.Reductions#min(NDArray, int)}).
     *
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public NDArray min(int axis) {
        return Reductions.min(this, axis);
    }

    /**
     * Minimum over several axes
     * ({@link org.numjvm.ufunc.Reductions#min(NDArray, int[], boolean)}).
     *
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public NDArray min(int[] axes, boolean keepdims) {
        return Reductions.min(this, axes, keepdims);
    }

    /**
     * Mean over all elements ({@link org.numjvm.ufunc.Reductions#mean(NDArray)}).
     *
     * @return a 0-D array
     */
    public NDArray mean() {
        return Reductions.mean(this);
    }

    /**
     * Mean over one axis ({@link org.numjvm.ufunc.Reductions#mean(NDArray, int)}).
     *
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public NDArray mean(int axis) {
        return Reductions.mean(this, axis);
    }

    /**
     * Mean over several axes
     * ({@link org.numjvm.ufunc.Reductions#mean(NDArray, int[], boolean)}).
     *
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public NDArray mean(int[] axes, boolean keepdims) {
        return Reductions.mean(this, axes, keepdims);
    }

    /**
     * Whether any element is true over all elements ({@link org.numjvm.ufunc.Reductions#any(NDArray)}).
     *
     * @return a 0-D array
     */
    public NDArray any() {
        return Reductions.any(this);
    }

    /**
     * Whether any element is true over one axis ({@link org.numjvm.ufunc.Reductions#any(NDArray, int)}).
     *
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public NDArray any(int axis) {
        return Reductions.any(this, axis);
    }

    /**
     * Whether any element is true over several axes
     * ({@link org.numjvm.ufunc.Reductions#any(NDArray, int[], boolean)}).
     *
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public NDArray any(int[] axes, boolean keepdims) {
        return Reductions.any(this, axes, keepdims);
    }

    /**
     * Whether every element is true over all elements ({@link org.numjvm.ufunc.Reductions#all(NDArray)}).
     *
     * @return a 0-D array
     */
    public NDArray all() {
        return Reductions.all(this);
    }

    /**
     * Whether every element is true over one axis ({@link org.numjvm.ufunc.Reductions#all(NDArray, int)}).
     *
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public NDArray all(int axis) {
        return Reductions.all(this, axis);
    }

    /**
     * Whether every element is true over several axes
     * ({@link org.numjvm.ufunc.Reductions#all(NDArray, int[], boolean)}).
     *
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public NDArray all(int[] axes, boolean keepdims) {
        return Reductions.all(this, axes, keepdims);
    }

    /**
     * {@code this += other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.ADD.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.addInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray addInPlace(NDArray other) {
        return Ufuncs.ADD.apply(this, other, this);
    }

    /**
     * {@code this += other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.ADD.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.addInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand; a weak scalar
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray addInPlace(double other) {
        return Ufuncs.ADD.apply(this, other, this);
    }

    /**
     * {@code this += other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.ADD.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.addInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand; a weak scalar
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray addInPlace(long other) {
        return Ufuncs.ADD.apply(this, other, this);
    }

    /**
     * {@code this -= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.SUBTRACT.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.subtractInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray subtractInPlace(NDArray other) {
        return Ufuncs.SUBTRACT.apply(this, other, this);
    }

    /**
     * {@code this -= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.SUBTRACT.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.subtractInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand; a weak scalar
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray subtractInPlace(double other) {
        return Ufuncs.SUBTRACT.apply(this, other, this);
    }

    /**
     * {@code this -= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.SUBTRACT.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.subtractInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand; a weak scalar
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray subtractInPlace(long other) {
        return Ufuncs.SUBTRACT.apply(this, other, this);
    }

    /**
     * {@code this *= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.MULTIPLY.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.multiplyInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray multiplyInPlace(NDArray other) {
        return Ufuncs.MULTIPLY.apply(this, other, this);
    }

    /**
     * {@code this *= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.MULTIPLY.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.multiplyInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand; a weak scalar
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray multiplyInPlace(double other) {
        return Ufuncs.MULTIPLY.apply(this, other, this);
    }

    /**
     * {@code this *= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.MULTIPLY.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.multiplyInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand; a weak scalar
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray multiplyInPlace(long other) {
        return Ufuncs.MULTIPLY.apply(this, other, this);
    }

    /**
     * {@code this /= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.DIVIDE.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.divideInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray divideInPlace(NDArray other) {
        return Ufuncs.DIVIDE.apply(this, other, this);
    }

    /**
     * {@code this /= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.DIVIDE.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.divideInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand; a weak scalar
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray divideInPlace(double other) {
        return Ufuncs.DIVIDE.apply(this, other, this);
    }

    /**
     * {@code this /= other} in place, with broadcasting of {@code other}
     * ({@code Ufuncs.DIVIDE.apply(this, other, this)}). The result is converted to this array's
     * dtype with {@code SAME_KIND} casting, so {@code int8Array.divideInPlace(1.5)} is a
     * {@link org.numjvm.exception.DTypeException}, as in NumPy.
     *
     * @param other the other operand; a weak scalar
     * @return this array
     * @throws ReadOnlyArrayException if this array is read-only
     */
    public NDArray divideInPlace(long other) {
        return Ufuncs.DIVIDE.apply(this, other, this);
    }

    // ------------------------------------------------------------------ element access: helpers

    private long idx(long i, int axis) {
        long n = shape.dims[axis];
        if (Long.compareUnsigned(i, n) >= 0) {
            throw IndexException.outOfBounds(i, axis, n);
        }
        return i;
    }

    private void requireRank(int rank) {
        if (shape.dims.length != rank) {
            throw new IndexException(rank + (rank == 1 ? " index" : " indices") + " given for an array of rank "
                    + shape.dims.length);
        }
    }

    private long at0() {
        requireRank(0);
        return offset;
    }

    private long at1(long i) {
        requireRank(1);
        return offset + idx(i, 0) * strides.values[0];
    }

    private long at2(long i, long j) {
        requireRank(2);
        long[] st = strides.values;
        return offset + idx(i, 0) * st[0] + idx(j, 1) * st[1];
    }

    private long at3(long i, long j, long k) {
        requireRank(3);
        long[] st = strides.values;
        return offset + idx(i, 0) * st[0] + idx(j, 1) * st[1] + idx(k, 2) * st[2];
    }

    /**
     * Byte offset of the element at {@code index}; validates rank and every index.
     */
    long elementOffset(long[] index) {
        requireRank(index.length);
        long[] st = strides.values;
        long off = offset;
        for (int d = 0; d < index.length; d++) {
            off += idx(index[d], d) * st[d];
        }
        return off;
    }

    private void requireWriteable() {
        if (!writeable) {
            throw new ReadOnlyArrayException("array is read-only"
                    + (hasBroadcastAxis(shape.dims, strides.values) ? " (broadcast view)" : ""));
        }
    }

    private double readDouble(long off) {
        try {
            return ElementCodec.readDouble(segment, off, dtype);
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, buffer.owner());
        }
    }

    private long readLong(long off) {
        try {
            return ElementCodec.readLong(segment, off, dtype);
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, buffer.owner());
        }
    }

    private boolean readBoolean(long off) {
        try {
            return ElementCodec.readBoolean(segment, off, dtype);
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, buffer.owner());
        }
    }

    private void writeDouble(long off, double v) {
        requireWriteable();
        try {
            ElementCodec.writeDouble(segment, off, dtype, v);
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, buffer.owner());
        }
    }

    private void writeLong(long off, long v) {
        requireWriteable();
        try {
            ElementCodec.writeLong(segment, off, dtype, v);
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, buffer.owner());
        }
    }

    // ------------------------------------------------------------------ double

    /**
     * Returns the single element of a 0-D array, as a {@code double}.
     *
     * @return the single element of a 0-D array, as a {@code double}
     */
    public double getDouble() {
        return readDouble(at0());
    }

    /**
     * Returns the element, as a {@code double}.
     *
     * @param i index on axis 0
     * @return the element, as a {@code double}
     */
    public double getDouble(long i) {
        return readDouble(at1(i));
    }

    /**
     * Returns the element, as a {@code double}.
     *
     * @param i index on axis 0
     * @param j index on axis 1
     * @return the element, as a {@code double}
     */
    public double getDouble(long i, long j) {
        return readDouble(at2(i, j));
    }

    /**
     * Returns the element, as a {@code double}.
     *
     * @param i index on axis 0
     * @param j index on axis 1
     * @param k index on axis 2
     * @return the element, as a {@code double}
     */
    public double getDouble(long i, long j, long k) {
        return readDouble(at3(i, j, k));
    }

    /**
     * Returns the element, as a {@code double}.
     *
     * @param index one index per axis
     * @return the element, as a {@code double}
     */
    public double getDouble(long[] index) {
        return readDouble(elementOffset(index));
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param value new value of the single element of a 0-D array
     */
    public void setDouble(double value) {
        writeDouble(at0(), value);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param value new value
     */
    public void setDouble(long i, double value) {
        writeDouble(at1(i), value);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param value new value
     */
    public void setDouble(long i, long j, double value) {
        writeDouble(at2(i, j), value);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param k     index on axis 2
     * @param value new value
     */
    public void setDouble(long i, long j, long k, double value) {
        writeDouble(at3(i, j, k), value);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param index one index per axis
     * @param value new value
     */
    public void setDouble(long[] index, double value) {
        writeDouble(elementOffset(index), value);
    }

    // ------------------------------------------------------------------ long

    /**
     * Returns the single element of a 0-D array, as a {@code long}.
     *
     * @return the single element of a 0-D array, as a {@code long}
     */
    public long getLong() {
        return readLong(at0());
    }

    /**
     * Returns the element, as a {@code long}.
     *
     * @param i index on axis 0
     * @return the element, as a {@code long}
     */
    public long getLong(long i) {
        return readLong(at1(i));
    }

    /**
     * Returns the element, as a {@code long}.
     *
     * @param i index on axis 0
     * @param j index on axis 1
     * @return the element, as a {@code long}
     */
    public long getLong(long i, long j) {
        return readLong(at2(i, j));
    }

    /**
     * Returns the element, as a {@code long}.
     *
     * @param i index on axis 0
     * @param j index on axis 1
     * @param k index on axis 2
     * @return the element, as a {@code long}
     */
    public long getLong(long i, long j, long k) {
        return readLong(at3(i, j, k));
    }

    /**
     * Returns the element, as a {@code long}.
     *
     * @param index one index per axis
     * @return the element, as a {@code long}
     */
    public long getLong(long[] index) {
        return readLong(elementOffset(index));
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param value new value of the single element of a 0-D array
     */
    public void setLong(long value) {
        writeLong(at0(), value);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param value new value
     */
    public void setLong(long i, long value) {
        writeLong(at1(i), value);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param value new value
     */
    public void setLong(long i, long j, long value) {
        writeLong(at2(i, j), value);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param k     index on axis 2
     * @param value new value
     */
    public void setLong(long i, long j, long k, long value) {
        writeLong(at3(i, j, k), value);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param index one index per axis
     * @param value new value
     */
    public void setLong(long[] index, long value) {
        writeLong(elementOffset(index), value);
    }

    // ------------------------------------------------------------------ boolean

    /**
     * Returns the single element of a 0-D array, as a {@code boolean}.
     *
     * @return the single element of a 0-D array, as a {@code boolean}
     */
    public boolean getBoolean() {
        return readBoolean(at0());
    }

    /**
     * Returns {@code true} if the element is non-zero.
     *
     * @param i index on axis 0
     * @return {@code true} if the element is non-zero
     */
    public boolean getBoolean(long i) {
        return readBoolean(at1(i));
    }

    /**
     * Returns {@code true} if the element is non-zero.
     *
     * @param i index on axis 0
     * @param j index on axis 1
     * @return {@code true} if the element is non-zero
     */
    public boolean getBoolean(long i, long j) {
        return readBoolean(at2(i, j));
    }

    /**
     * Returns {@code true} if the element is non-zero.
     *
     * @param i index on axis 0
     * @param j index on axis 1
     * @param k index on axis 2
     * @return {@code true} if the element is non-zero
     */
    public boolean getBoolean(long i, long j, long k) {
        return readBoolean(at3(i, j, k));
    }

    /**
     * Returns {@code true} if the element is non-zero.
     *
     * @param index one index per axis
     * @return {@code true} if the element is non-zero
     */
    public boolean getBoolean(long[] index) {
        return readBoolean(elementOffset(index));
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param value new value of the single element of a 0-D array
     */
    public void setBoolean(boolean value) {
        writeLong(at0(), value ? 1 : 0);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param value new value
     */
    public void setBoolean(long i, boolean value) {
        writeLong(at1(i), value ? 1 : 0);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param value new value
     */
    public void setBoolean(long i, long j, boolean value) {
        writeLong(at2(i, j), value ? 1 : 0);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param k     index on axis 2
     * @param value new value
     */
    public void setBoolean(long i, long j, long k, boolean value) {
        writeLong(at3(i, j, k), value ? 1 : 0);
    }

    /**
     * Writes one element, converting the value as described in the class documentation.
     *
     * @param index one index per axis
     * @param value new value
     */
    public void setBoolean(long[] index, boolean value) {
        writeLong(elementOffset(index), value ? 1 : 0);
    }
}
