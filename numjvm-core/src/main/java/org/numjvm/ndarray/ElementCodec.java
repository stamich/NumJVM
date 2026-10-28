package org.numjvm.ndarray;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.Unsigned;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.HalfFloat;

/**
 * Reads and writes single elements, converting between dtypes and Java primitives.
 *
 * <p>Layouts are unaligned constants so the JIT can intrinsify every access (ADR-002).
 * The conversion rules are documented on {@link NDArray}.
 */
final class ElementCodec {

    static final ValueLayout.OfByte I8 = ValueLayout.JAVA_BYTE;
    static final ValueLayout.OfShort I16 = ValueLayout.JAVA_SHORT_UNALIGNED;
    static final ValueLayout.OfInt I32 = ValueLayout.JAVA_INT_UNALIGNED;
    static final ValueLayout.OfLong I64 = ValueLayout.JAVA_LONG_UNALIGNED;
    static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED;
    static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    private ElementCodec() {
    }

    static double readDouble(MemorySegment s, long off, DType t) {
        return switch (t) {
            case BOOL -> s.get(I8, off) != 0 ? 1.0 : 0.0;
            case INT8 -> s.get(I8, off);
            case UINT8 -> s.get(I8, off) & 0xFF;
            case INT16 -> s.get(I16, off);
            case UINT16 -> s.get(I16, off) & 0xFFFF;
            case INT32 -> s.get(I32, off);
            case UINT32 -> s.get(I32, off) & 0xFFFF_FFFFL;
            case INT64 -> s.get(I64, off);
            case UINT64 -> Unsigned.toDouble(s.get(I64, off));
            case FLOAT16 -> Float.float16ToFloat(s.get(I16, off));
            case BFLOAT16 -> HalfFloat.bfloat16ToFloat(s.get(I16, off));
            case FLOAT32 -> s.get(F32, off);
            case FLOAT64 -> s.get(F64, off);
            case COMPLEX64, COMPLEX128 -> throw complex(t, "getDouble");
        };
    }

    static long readLong(MemorySegment s, long off, DType t) {
        return switch (t) {
            case BOOL -> s.get(I8, off) != 0 ? 1 : 0;
            case INT8 -> s.get(I8, off);
            case UINT8 -> s.get(I8, off) & 0xFF;
            case INT16 -> s.get(I16, off);
            case UINT16 -> s.get(I16, off) & 0xFFFF;
            case INT32 -> s.get(I32, off);
            case UINT32 -> s.get(I32, off) & 0xFFFF_FFFFL;
            case INT64, UINT64 -> s.get(I64, off);
            case FLOAT16, BFLOAT16, FLOAT32, FLOAT64 ->
                    throw new DTypeException("getLong is not defined for " + t.typeName() + "; use getDouble");
            case COMPLEX64, COMPLEX128 -> throw complex(t, "getLong");
        };
    }

    static boolean readBoolean(MemorySegment s, long off, DType t) {
        return switch (t) {
            case BOOL, INT8, UINT8 -> s.get(I8, off) != 0;
            case INT16, UINT16 -> s.get(I16, off) != 0;
            case INT32, UINT32 -> s.get(I32, off) != 0;
            case INT64, UINT64 -> s.get(I64, off) != 0;
            case FLOAT16, BFLOAT16, FLOAT32, FLOAT64 -> readDouble(s, off, t) != 0.0;
            case COMPLEX64 -> s.get(F32, off) != 0 || s.get(F32, off + 4) != 0;
            case COMPLEX128 -> s.get(F64, off) != 0 || s.get(F64, off + 8) != 0;
        };
    }

    static void writeDouble(MemorySegment s, long off, DType t, double v) {
        switch (t) {
            case BOOL -> s.set(I8, off, v != 0.0 ? (byte) 1 : (byte) 0);
            case INT8, UINT8 -> s.set(I8, off, (byte) (long) v);
            case INT16, UINT16 -> s.set(I16, off, (short) (long) v);
            case INT32, UINT32 -> s.set(I32, off, (int) (long) v);
            case INT64 -> s.set(I64, off, (long) v);
            case UINT64 -> s.set(I64, off, Unsigned.fromDouble(v));
            case FLOAT16 -> s.set(I16, off, HalfFloat.doubleToFloat16(v));
            case BFLOAT16 -> s.set(I16, off, HalfFloat.doubleToBFloat16(v));
            case FLOAT32 -> s.set(F32, off, (float) v);
            case FLOAT64 -> s.set(F64, off, v);
            case COMPLEX64, COMPLEX128 -> writeComplex(s, off, t, v, 0.0);
        }
    }

    static void writeLong(MemorySegment s, long off, DType t, long v) {
        switch (t) {
            case BOOL -> s.set(I8, off, v != 0 ? (byte) 1 : (byte) 0);
            case INT8, UINT8 -> s.set(I8, off, (byte) v);
            case INT16, UINT16 -> s.set(I16, off, (short) v);
            case INT32, UINT32 -> s.set(I32, off, (int) v);
            case INT64, UINT64 -> s.set(I64, off, v);
            case FLOAT16 -> s.set(I16, off, Float.floatToFloat16((float) v));
            case BFLOAT16 -> s.set(I16, off, HalfFloat.floatToBFloat16((float) v));
            case FLOAT32 -> s.set(F32, off, (float) v);
            case FLOAT64 -> s.set(F64, off, (double) v);
            case COMPLEX64, COMPLEX128 -> writeComplex(s, off, t, (double) v, 0.0);
        }
    }

    static Complex readComplex(MemorySegment s, long off, DType t) {
        return switch (t) {
            case COMPLEX64 -> new Complex(s.get(F32, off), s.get(F32, off + 4));
            case COMPLEX128 -> new Complex(s.get(F64, off), s.get(F64, off + 8));
            default -> new Complex(readDouble(s, off, t), 0.0);
        };
    }

    static void writeComplex(MemorySegment s, long off, DType t, double re, double im) {
        switch (t) {
            case COMPLEX64 -> {
                s.set(F32, off, (float) re);
                s.set(F32, off + 4, (float) im);
            }
            case COMPLEX128 -> {
                s.set(F64, off, re);
                s.set(F64, off + 8, im);
            }
            default -> throw new DTypeException("setComplex needs a complex dtype, not " + t.typeName()
                    + "; use setDouble for the real part");
        }
    }

    private static DTypeException complex(DType t, String op) {
        return new DTypeException(op + " is not defined for " + t.typeName() + "; use getComplex");
    }
}
