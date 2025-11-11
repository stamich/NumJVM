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
 *       the dtype's width; bool stores {@code value != 0}; float types round to nearest</td></tr>
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

    /** Allocates a zero-filled contiguous array. */
    static NDArray allocate(MemoryAllocator allocator, DType dtype, Shape shape, Order order) {
        long bytes = Sizes.bytesFor(shape.size(), dtype.size());
        Strides strides = order == Order.F
                ? Strides.fContiguous(shape, dtype.size())
                : Strides.cContiguous(shape, dtype.size());
        return new NDArray(allocator.allocate(bytes), dtype, 0, shape, strides, true);
    }

    // ------------------------------------------------------------------ metadata

    /** @return the element type */
    public DType dtype() {
        return dtype;
    }

    /** @return the shape, e.g. {@code [2, 3]} */
    public Shape shape() {
        return shape;
    }

    /** @return the byte strides, e.g. {@code [24, 8]} for a C-contiguous 2×3 float64 array */
    public Strides strides() {
        return strides;
    }

    /** @return the number of dimensions (0 for a scalar array) */
    public int ndim() {
        return shape.dims.length;
    }

    /** @return the number of elements */
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

    /** @return the size of one element in bytes */
    public int itemSize() {
        return dtype.size();
    }

    /** @return the number of bytes the elements occupy: {@code size() * itemSize()} */
    public long nbytes() {
        return shape.size() * dtype.size();
    }

    /** @return {@code true} if elements may be written through this array */
    public boolean isWriteable() {
        return writeable;
    }

    /** @return layout and access flags (contiguity, writeability, alignment) */
    public ArrayFlags flags() {
        return ArrayFlags.of(flags);
    }

    /** @return {@code true} if the array is C-contiguous */
    public boolean isCContiguous() {
        return (flags & ArrayFlags.C_CONTIGUOUS) != 0;
    }

    /** @return {@code true} if the array is Fortran-contiguous */
    public boolean isFContiguous() {
        return (flags & ArrayFlags.F_CONTIGUOUS) != 0;
    }

    /** @return the owner that decides how long this array's memory stays valid */
    public MemoryOwner memoryOwner() {
        return buffer.owner();
    }

    /** @return {@code true} if the array's memory may still be accessed */
    public boolean isAlive() {
        return buffer.isAlive();
    }

    /** @return {@code true} if the elements live in native (off-heap) memory */
    public boolean isNative() {
        return buffer.isNative();
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
     * <p>Integer and bool conversions go through {@code long}, so int64 and uint64 values
     * keep all their bits; conversions involving a floating-point type go through
     * {@code double} with the rules of {@code setDouble}. Real values become complex
     * with a zero imaginary part; converting complex to real is rejected rather than
     * silently discarding the imaginary part.
     *
     * @param target the dtype of the result
     * @return the converted copy; a plain copy if {@code target} equals this dtype
     * @throws org.numjvm.exception.DTypeException for complex to real conversions
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

    /** Byte offset of the element at {@code index}; validates rank and every index. */
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

    /** @return the single element of a 0-D array, as a {@code double} */
    public double getDouble() {
        return readDouble(at0());
    }

    /**
     * @param i index on axis 0
     * @return the element, as a {@code double}
     */
    public double getDouble(long i) {
        return readDouble(at1(i));
    }

    /**
     * @param i index on axis 0
     * @param j index on axis 1
     * @return the element, as a {@code double}
     */
    public double getDouble(long i, long j) {
        return readDouble(at2(i, j));
    }

    /**
     * @param i index on axis 0
     * @param j index on axis 1
     * @param k index on axis 2
     * @return the element, as a {@code double}
     */
    public double getDouble(long i, long j, long k) {
        return readDouble(at3(i, j, k));
    }

    /**
     * @param index one index per axis
     * @return the element, as a {@code double}
     */
    public double getDouble(long[] index) {
        return readDouble(elementOffset(index));
    }

    /** @param value new value of the single element of a 0-D array */
    public void setDouble(double value) {
        writeDouble(at0(), value);
    }

    /**
     * @param i     index on axis 0
     * @param value new value
     */
    public void setDouble(long i, double value) {
        writeDouble(at1(i), value);
    }

    /**
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param value new value
     */
    public void setDouble(long i, long j, double value) {
        writeDouble(at2(i, j), value);
    }

    /**
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param k     index on axis 2
     * @param value new value
     */
    public void setDouble(long i, long j, long k, double value) {
        writeDouble(at3(i, j, k), value);
    }

    /**
     * @param index one index per axis
     * @param value new value
     */
    public void setDouble(long[] index, double value) {
        writeDouble(elementOffset(index), value);
    }

    // ------------------------------------------------------------------ long

    /** @return the single element of a 0-D array, as a {@code long} */
    public long getLong() {
        return readLong(at0());
    }

    /**
     * @param i index on axis 0
     * @return the element, as a {@code long}
     */
    public long getLong(long i) {
        return readLong(at1(i));
    }

    /**
     * @param i index on axis 0
     * @param j index on axis 1
     * @return the element, as a {@code long}
     */
    public long getLong(long i, long j) {
        return readLong(at2(i, j));
    }

    /**
     * @param i index on axis 0
     * @param j index on axis 1
     * @param k index on axis 2
     * @return the element, as a {@code long}
     */
    public long getLong(long i, long j, long k) {
        return readLong(at3(i, j, k));
    }

    /**
     * @param index one index per axis
     * @return the element, as a {@code long}
     */
    public long getLong(long[] index) {
        return readLong(elementOffset(index));
    }

    /** @param value new value of the single element of a 0-D array */
    public void setLong(long value) {
        writeLong(at0(), value);
    }

    /**
     * @param i     index on axis 0
     * @param value new value
     */
    public void setLong(long i, long value) {
        writeLong(at1(i), value);
    }

    /**
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param value new value
     */
    public void setLong(long i, long j, long value) {
        writeLong(at2(i, j), value);
    }

    /**
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param k     index on axis 2
     * @param value new value
     */
    public void setLong(long i, long j, long k, long value) {
        writeLong(at3(i, j, k), value);
    }

    /**
     * @param index one index per axis
     * @param value new value
     */
    public void setLong(long[] index, long value) {
        writeLong(elementOffset(index), value);
    }

    // ------------------------------------------------------------------ boolean

    /** @return the single element of a 0-D array, as a {@code boolean} */
    public boolean getBoolean() {
        return readBoolean(at0());
    }

    /**
     * @param i index on axis 0
     * @return {@code true} if the element is non-zero
     */
    public boolean getBoolean(long i) {
        return readBoolean(at1(i));
    }

    /**
     * @param i index on axis 0
     * @param j index on axis 1
     * @return {@code true} if the element is non-zero
     */
    public boolean getBoolean(long i, long j) {
        return readBoolean(at2(i, j));
    }

    /**
     * @param i index on axis 0
     * @param j index on axis 1
     * @param k index on axis 2
     * @return {@code true} if the element is non-zero
     */
    public boolean getBoolean(long i, long j, long k) {
        return readBoolean(at3(i, j, k));
    }

    /**
     * @param index one index per axis
     * @return {@code true} if the element is non-zero
     */
    public boolean getBoolean(long[] index) {
        return readBoolean(elementOffset(index));
    }

    /** @param value new value of the single element of a 0-D array */
    public void setBoolean(boolean value) {
        writeLong(at0(), value ? 1 : 0);
    }

    /**
     * @param i     index on axis 0
     * @param value new value
     */
    public void setBoolean(long i, boolean value) {
        writeLong(at1(i), value ? 1 : 0);
    }

    /**
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param value new value
     */
    public void setBoolean(long i, long j, boolean value) {
        writeLong(at2(i, j), value ? 1 : 0);
    }

    /**
     * @param i     index on axis 0
     * @param j     index on axis 1
     * @param k     index on axis 2
     * @param value new value
     */
    public void setBoolean(long i, long j, long k, boolean value) {
        writeLong(at3(i, j, k), value ? 1 : 0);
    }

    /**
     * @param index one index per axis
     * @param value new value
     */
    public void setBoolean(long[] index, boolean value) {
        writeLong(elementOffset(index), value ? 1 : 0);
    }
}
