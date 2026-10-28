package org.numjvm.codegen;

import java.util.List;

/**
 * How generated loops read, compute with and write one dtype.
 *
 * <p>Elements are read into a Java <em>compute type</em> and the result is written back
 * with a conversion: narrow integers compute in {@code int} (unsigned ones masked to
 * their non-negative value), uint32 and int64 in {@code long}, uint64 in {@code long}
 * with unsigned operations, float16 and bfloat16 in {@code float} (as NumPy computes
 * float16 in float32), float32 in {@code float} and float64 in {@code double}.
 *
 * @param dtype   the {@code DType} constant name
 * @param cls     generated class name prefix, e.g. {@code Int8} for {@code Int8Loops}
 * @param family  the group of dtypes sharing operation expressions
 * @param ct      Java compute type
 * @param layout  name of the value-layout constant used for memory access
 * @param size    element size in bytes
 * @param read    read expression; {@code {s}} is the segment, {@code {o}} the byte offset
 * @param write   write statement; {@code {s}}, {@code {o}} and {@code {v}}, the value
 */
record DTypeSpec(String dtype, String cls, Family family, String ct, String layout, int size, String read,
                 String write) {

    /**
     * {@code v}, a value of the compute type, wrapped to this dtype's width as storing and
     * reading it back would: needed where a value stays in a register across steps
     * (reductions) and the operation is not modular (floor division, power, minimum, ...).
     * float16 and bfloat16 are not narrowed: NumPy accumulates them in float32.
     */
    String narrow(String v) {
        return switch (dtype) {
            case "INT8" -> "(byte) (" + v + ")";
            case "UINT8" -> "((" + v + ") & 0xFF)";
            case "INT16" -> "(short) (" + v + ")";
            case "UINT16" -> "((" + v + ") & 0xFFFF)";
            case "UINT32" -> "((" + v + ") & 0xFFFF_FFFFL)";
            default -> v;
        };
    }

    /**
     * Groups of dtypes whose operations are written with the same expressions.
     */
    enum Family {
        /** bool, computed as {@code boolean}. */
        BOOL,
        /** int8, uint8, int16, uint16, int32: {@code int}, unsigned values non-negative. */
        INT,
        /** uint32 and int64: {@code long}, uint32 values non-negative. */
        LONG,
        /** uint64: {@code long} holding the raw bits, unsigned operations. */
        ULONG,
        /** float16, bfloat16, float32: {@code float}. */
        FLOAT,
        /** float64: {@code double}. */
        DOUBLE
    }

    String read(String segment, String offset) {
        return read.replace("{s}", segment).replace("{o}", offset);
    }

    String write(String segment, String offset, String value) {
        return write.replace("{s}", segment).replace("{o}", offset).replace("{v}", value);
    }

    boolean isFloat() {
        return family == Family.FLOAT || family == Family.DOUBLE;
    }

    boolean isInteger() {
        return family == Family.INT || family == Family.LONG || family == Family.ULONG;
    }

    private static final String BOOL_WRITE = "{s}.set(I8, {o}, ({v}) ? (byte) 1 : (byte) 0)";

    /**
     * The real dtypes in NumPy's loop order (bool, then integers by size with signed before
     * unsigned, then floats), with NumJVM's bfloat16 last so that type resolution never
     * prefers it over a NumPy type.
     */
    static final List<DTypeSpec> REAL = List.of(
            new DTypeSpec("BOOL", "Bool", Family.BOOL, "boolean", "I8", 1, "({s}.get(I8, {o}) != 0)", BOOL_WRITE),
            new DTypeSpec("INT8", "Int8", Family.INT, "int", "I8", 1, "{s}.get(I8, {o})",
                    "{s}.set(I8, {o}, (byte) ({v}))"),
            new DTypeSpec("UINT8", "UInt8", Family.INT, "int", "I8", 1, "({s}.get(I8, {o}) & 0xFF)",
                    "{s}.set(I8, {o}, (byte) ({v}))"),
            new DTypeSpec("INT16", "Int16", Family.INT, "int", "I16", 2, "{s}.get(I16, {o})",
                    "{s}.set(I16, {o}, (short) ({v}))"),
            new DTypeSpec("UINT16", "UInt16", Family.INT, "int", "I16", 2, "({s}.get(I16, {o}) & 0xFFFF)",
                    "{s}.set(I16, {o}, (short) ({v}))"),
            new DTypeSpec("INT32", "Int32", Family.INT, "int", "I32", 4, "{s}.get(I32, {o})",
                    "{s}.set(I32, {o}, {v})"),
            new DTypeSpec("UINT32", "UInt32", Family.LONG, "long", "I32", 4, "({s}.get(I32, {o}) & 0xFFFF_FFFFL)",
                    "{s}.set(I32, {o}, (int) ({v}))"),
            new DTypeSpec("INT64", "Int64", Family.LONG, "long", "I64", 8, "{s}.get(I64, {o})",
                    "{s}.set(I64, {o}, {v})"),
            new DTypeSpec("UINT64", "UInt64", Family.ULONG, "long", "I64", 8, "{s}.get(I64, {o})",
                    "{s}.set(I64, {o}, {v})"),
            new DTypeSpec("FLOAT16", "Float16", Family.FLOAT, "float", "I16", 2, "Float.float16ToFloat({s}.get(I16, {o}))",
                    "{s}.set(I16, {o}, Float.floatToFloat16({v}))"),
            new DTypeSpec("FLOAT32", "Float32", Family.FLOAT, "float", "F32", 4, "{s}.get(F32, {o})",
                    "{s}.set(F32, {o}, {v})"),
            new DTypeSpec("FLOAT64", "Float64", Family.DOUBLE, "double", "F64", 8, "{s}.get(F64, {o})",
                    "{s}.set(F64, {o}, {v})"),
            new DTypeSpec("BFLOAT16", "BFloat16", Family.FLOAT, "float", "I16", 2,
                    "HalfFloat.bfloat16ToFloat({s}.get(I16, {o}))",
                    "{s}.set(I16, {o}, HalfFloat.floatToBFloat16({v}))"));

    static DTypeSpec of(String dtype) {
        for (DTypeSpec d : REAL) {
            if (d.dtype.equals(dtype)) {
                return d;
            }
        }
        throw new IllegalArgumentException("unknown dtype " + dtype);
    }

    /**
     * The bool dtype, the output of comparisons and logical operations.
     */
    static DTypeSpec bool() {
        return REAL.getFirst();
    }
}
