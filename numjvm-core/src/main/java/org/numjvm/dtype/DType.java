package org.numjvm.dtype;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Locale;

import org.numjvm.exception.DTypeException;

/**
 * Element type of an array.
 *
 * <p>A dtype describes how one element is stored: its size, alignment, category and
 * memory layout. Elements are always stored in the platform's native byte order;
 * byte order is a property of external formats (such as {@code .npy}), not of a dtype.
 *
 * <p>All dtypes can be stored, copied, viewed and read from or written to {@code .npy}
 * files (except {@link #BFLOAT16}, which NumPy does not define). Scalar access and
 * conversions are available for all dtypes; complex elements are read and written as
 * {@link Complex} values.
 *
 * <p>Client code should use the query methods ({@link #size()}, {@link #kind()},
 * {@link #isFloatingPoint()}, ...) rather than switching on the constants, so that
 * dtypes added later do not break it.
 */
public enum DType {

    /**
     * Boolean, one byte, 0 or 1.
     */
    BOOL(1, DTypeKind.BOOL, "b1"),
    /**
     * Signed 8-bit integer.
     */
    INT8(1, DTypeKind.SIGNED_INTEGER, "i1"),
    /**
     * Signed 16-bit integer.
     */
    INT16(2, DTypeKind.SIGNED_INTEGER, "i2"),
    /**
     * Signed 32-bit integer.
     */
    INT32(4, DTypeKind.SIGNED_INTEGER, "i4"),
    /**
     * Signed 64-bit integer.
     */
    INT64(8, DTypeKind.SIGNED_INTEGER, "i8"),
    /**
     * Unsigned 8-bit integer.
     */
    UINT8(1, DTypeKind.UNSIGNED_INTEGER, "u1"),
    /**
     * Unsigned 16-bit integer.
     */
    UINT16(2, DTypeKind.UNSIGNED_INTEGER, "u2"),
    /**
     * Unsigned 32-bit integer.
     */
    UINT32(4, DTypeKind.UNSIGNED_INTEGER, "u4"),
    /**
     * Unsigned 64-bit integer. Java has no unsigned 64-bit primitive; values travel as raw bits.
     */
    UINT64(8, DTypeKind.UNSIGNED_INTEGER, "u8"),
    /**
     * IEEE 754 binary16 (half precision).
     */
    FLOAT16(2, DTypeKind.FLOATING, "f2"),
    /**
     * bfloat16: the upper 16 bits of an IEEE 754 binary32. Not a NumPy dtype
     * ({@code NUMJVM_SPECIFIC}); it has no {@code .npy} descriptor.
     */
    BFLOAT16(2, DTypeKind.FLOATING, null),
    /**
     * IEEE 754 binary32 (single precision).
     */
    FLOAT32(4, DTypeKind.FLOATING, "f4"),
    /**
     * IEEE 754 binary64 (double precision). The default floating-point dtype.
     */
    FLOAT64(8, DTypeKind.FLOATING, "f8"),
    /**
     * Complex number with two binary32 components (real, imaginary).
     */
    COMPLEX64(8, DTypeKind.COMPLEX, "c8"),
    /**
     * Complex number with two binary64 components (real, imaginary).
     */
    COMPLEX128(16, DTypeKind.COMPLEX, "c16");

    private static final DType[] VALUES = values();

    private final int size;
    private final DTypeKind kind;
    private final String numpyTypeCode;

    DType(int size, DTypeKind kind, String numpyTypeCode) {
        this.size = size;
        this.kind = kind;
        this.numpyTypeCode = numpyTypeCode;
    }

    /**
     * Returns the size of one element in bytes.
     *
     * @return 1, 2, 4, 8 or 16
     */
    public int size() {
        return size;
    }

    /**
     * Returns the natural alignment of one element in bytes: the element size for
     * real dtypes, the component size for complex dtypes.
     *
     * @return 1, 2, 4 or 8
     */
    public int alignment() {
        return kind == DTypeKind.COMPLEX ? size / 2 : size;
    }

    /**
     * Returns the category of this dtype.
     *
     * @return the kind
     */
    public DTypeKind kind() {
        return kind;
    }

    /**
     * Returns the lower-case name used by NumPy and the Array API, e.g. {@code "float64"}.
     *
     * @return the dtype name
     */
    public String typeName() {
        return name().toLowerCase(Locale.ROOT);
    }

    /**
     * Returns {@code true} for {@link #BOOL}.
     *
     * @return {@code true} for {@link #BOOL}
     */
    public boolean isBool() {
        return kind == DTypeKind.BOOL;
    }

    /**
     * Returns {@code true} for signed and unsigned integer dtypes.
     *
     * @return {@code true} for signed and unsigned integer dtypes
     */
    public boolean isInteger() {
        return kind == DTypeKind.SIGNED_INTEGER || kind == DTypeKind.UNSIGNED_INTEGER;
    }

    /**
     * Returns {@code true} for unsigned integer dtypes.
     *
     * @return {@code true} for unsigned integer dtypes
     */
    public boolean isUnsigned() {
        return kind == DTypeKind.UNSIGNED_INTEGER;
    }

    /**
     * Returns {@code true} for real floating-point dtypes (not complex).
     *
     * @return {@code true} for real floating-point dtypes (not complex)
     */
    public boolean isFloatingPoint() {
        return kind == DTypeKind.FLOATING;
    }

    /**
     * Returns {@code true} for complex dtypes.
     *
     * @return {@code true} for complex dtypes
     */
    public boolean isComplex() {
        return kind == DTypeKind.COMPLEX;
    }

    /**
     * Returns the memory layout of one element in native byte order, without
     * alignment constraints. Complex dtypes are a struct of {@code real} and
     * {@code imag}.
     *
     * @return a layout whose byte size equals {@link #size()}
     */
    public MemoryLayout layout() {
        return switch (this) {
            case BOOL, INT8, UINT8 -> ValueLayout.JAVA_BYTE;
            case INT16, UINT16, FLOAT16, BFLOAT16 -> ValueLayout.JAVA_SHORT_UNALIGNED;
            case INT32, UINT32 -> ValueLayout.JAVA_INT_UNALIGNED;
            case INT64, UINT64 -> ValueLayout.JAVA_LONG_UNALIGNED;
            case FLOAT32 -> ValueLayout.JAVA_FLOAT_UNALIGNED;
            case FLOAT64 -> ValueLayout.JAVA_DOUBLE_UNALIGNED;
            case COMPLEX64 -> MemoryLayout.structLayout(
                    ValueLayout.JAVA_FLOAT_UNALIGNED.withName("real"),
                    ValueLayout.JAVA_FLOAT_UNALIGNED.withName("imag"));
            case COMPLEX128 -> MemoryLayout.structLayout(
                    ValueLayout.JAVA_DOUBLE_UNALIGNED.withName("real"),
                    ValueLayout.JAVA_DOUBLE_UNALIGNED.withName("imag"));
        };
    }

    /**
     * Tells whether NumPy has an equivalent dtype that can be written to {@code .npy}.
     *
     * @return {@code false} only for {@link #BFLOAT16}
     */
    public boolean hasNumpyDescriptor() {
        return numpyTypeCode != null;
    }

    /**
     * Returns NumPy's array-protocol type string for this dtype in native byte order,
     * for example {@code "<f8"} on a little-endian machine or {@code "|u1"} for a
     * one-byte type.
     *
     * @return the descriptor
     * @throws DTypeException for {@link #BFLOAT16}
     */
    public String numpyDescriptor() {
        if (numpyTypeCode == null) {
            throw new DTypeException(typeName() + " has no NumPy equivalent");
        }
        char order = size == 1 ? '|' : ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? '<' : '>';
        return order + numpyTypeCode;
    }

    /**
     * Parses a NumPy array-protocol type string such as {@code "<f8"}, {@code ">i4"},
     * {@code "|b1"} or {@code "f8"}.
     *
     * <p>The byte-order character ({@code '<'}, {@code '>'}, {@code '='}, {@code '|'})
     * is validated but not returned; callers that read external data must handle byte
     * order themselves.
     *
     * @param descriptor the type string
     * @return the matching dtype
     * @throws DTypeException if the string is malformed or names an unsupported type
     *                        (strings, objects, datetimes, structured types, ...)
     */
    public static DType fromNumpyDescriptor(String descriptor) {
        if (descriptor == null || descriptor.isEmpty()) {
            throw new DTypeException("empty NumPy dtype descriptor");
        }
        String code = descriptor;
        char first = descriptor.charAt(0);
        if (first == '<' || first == '>' || first == '=' || first == '|') {
            code = descriptor.substring(1);
        }
        for (DType d : VALUES) {
            if (code.equals(d.numpyTypeCode)) {
                if (first == '|' && d.size > 1) {
                    throw new DTypeException("descriptor '" + descriptor
                            + "' marks a multi-byte type as byte-order independent");
                }
                return d;
            }
        }
        throw new DTypeException("unsupported NumPy dtype descriptor '" + descriptor + "'");
    }

    /**
     * Looks up a dtype by its {@link #typeName()}, e.g. {@code "int32"}.
     *
     * @param name the dtype name, case-insensitive
     * @return the dtype
     * @throws DTypeException if no dtype has that name
     */
    public static DType fromName(String name) {
        for (DType d : VALUES) {
            if (d.typeName().equalsIgnoreCase(name)) {
                return d;
            }
        }
        throw new DTypeException("unknown dtype '" + name + "'");
    }
}
