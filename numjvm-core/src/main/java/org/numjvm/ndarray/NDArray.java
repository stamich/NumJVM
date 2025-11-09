package org.numjvm.ndarray;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.internal.memory.MemoryAllocator;
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
}
