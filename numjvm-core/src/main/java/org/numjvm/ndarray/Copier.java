package org.numjvm.ndarray;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.internal.memory.MemoryErrors;

/**
 * Copies arrays into new contiguous storage, optionally converting the dtype.
 */
final class Copier {

    private Copier() {
    }

    static NDArray copy(NDArray src, MemoryAllocator allocator, DType target, Order order) {
        NDArray dst = NDArray.allocate(allocator, target, src.shape, order);
        if (src.size() == 0) {
            return dst;
        }
        try {
            if (target == src.dtype) {
                copySameType(src, dst, order);
            } else {
                convert(src, dst);
            }
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, src.buffer.owner());
        }
        return dst;
    }

    private static void copySameType(NDArray src, NDArray dst, Order order) {
        boolean bulk = order == Order.C ? src.isCContiguous() : src.isFContiguous();
        if (bulk) {
            MemorySegment.copy(src.segment, src.offset, dst.segment, 0, src.nbytes());
            return;
        }
        int item = src.dtype.size();
        Walk w = new Walk(src, dst);
        do {
            long ps = w.srcOff;
            long pd = w.dstOff;
            long ss = w.srcInner;
            long sd = w.dstInner;
            for (long k = 0; k < w.inner; k++, ps += ss, pd += sd) {
                copyElement(src.segment, ps, dst.segment, pd, item);
            }
        } while (w.next());
    }

    private static void copyElement(MemorySegment s, long ps, MemorySegment d, long pd, int item) {
        switch (item) {
            case 1 -> d.set(ValueLayout.JAVA_BYTE, pd, s.get(ValueLayout.JAVA_BYTE, ps));
            case 2 -> d.set(ElementCodec.I16, pd, s.get(ElementCodec.I16, ps));
            case 4 -> d.set(ElementCodec.I32, pd, s.get(ElementCodec.I32, ps));
            case 8 -> d.set(ElementCodec.I64, pd, s.get(ElementCodec.I64, ps));
            default -> {
                d.set(ElementCodec.I64, pd, s.get(ElementCodec.I64, ps));
                d.set(ElementCodec.I64, pd + 8, s.get(ElementCodec.I64, ps + 8));
            }
        }
    }

    private static void convert(NDArray src, NDArray dst) {
        DType from = src.dtype;
        DType to = dst.dtype;
        if (from.isComplex() && !to.isComplex()) {
            throw new DTypeException("cannot convert " + from.typeName() + " to " + to.typeName()
                    + " without discarding the imaginary part");
        }
        boolean integral = (from.isInteger() || from.isBool()) && (to.isInteger() || to.isBool());
        MemorySegment s = src.segment;
        MemorySegment d = dst.segment;
        Walk w = new Walk(src, dst);
        do {
            long ps = w.srcOff;
            long pd = w.dstOff;
            for (long k = 0; k < w.inner; k++, ps += w.srcInner, pd += w.dstInner) {
                if (to.isComplex()) {
                    double re = from.isComplex() ? component(s, ps, from, 0) : ElementCodec.readDouble(s, ps, from);
                    double im = from.isComplex() ? component(s, ps, from, 1) : 0.0;
                    setComponent(d, pd, to, 0, re);
                    setComponent(d, pd, to, 1, im);
                } else if (integral) {
                    ElementCodec.writeLong(d, pd, to, ElementCodec.readLong(s, ps, from));
                } else {
                    ElementCodec.writeDouble(d, pd, to, ElementCodec.readDouble(s, ps, from));
                }
            }
        } while (w.next());
    }

    private static double component(MemorySegment s, long off, DType t, int part) {
        return t == DType.COMPLEX64
                ? s.get(ElementCodec.F32, off + 4L * part)
                : s.get(ElementCodec.F64, off + 8L * part);
    }

    private static void setComponent(MemorySegment s, long off, DType t, int part, double v) {
        if (t == DType.COMPLEX64) {
            s.set(ElementCodec.F32, off + 4L * part, (float) v);
        } else {
            s.set(ElementCodec.F64, off + 8L * part, v);
        }
    }

    /**
     * Two-operand walk over the logical shape in C order: the outer axes advance like an
     * odometer, the innermost axis is left to the caller's loop (ADR-003).
     */
    private static final class Walk {
        final long[] dims;
        final long[] ss;
        final long[] ds;
        final long[] counter;
        final long inner;
        final long srcInner;
        final long dstInner;
        long srcOff;
        long dstOff;

        Walk(NDArray src, NDArray dst) {
            dims = src.shape.dims;
            ss = src.strides.values;
            ds = dst.strides.values;
            int rank = dims.length;
            counter = new long[Math.max(0, rank - 1)];
            inner = rank == 0 ? 1 : dims[rank - 1];
            srcInner = rank == 0 ? 0 : ss[rank - 1];
            dstInner = rank == 0 ? 0 : ds[rank - 1];
            srcOff = src.offset;
            dstOff = dst.offset;
        }

        boolean next() {
            for (int d = counter.length - 1; d >= 0; d--) {
                if (++counter[d] < dims[d]) {
                    srcOff += ss[d];
                    dstOff += ds[d];
                    return true;
                }
                counter[d] = 0;
                srcOff -= (dims[d] - 1) * ss[d];
                dstOff -= (dims[d] - 1) * ds[d];
            }
            return false;
        }
    }
}
