package org.numjvm.io;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.ndarray.Shape;

/**
 * The header of a NumPy {@code .npy} file, format versions 1.0, 2.0 and 3.0.
 *
 * <p>Layout: the magic string {@code \x93NUMPY}, a major and minor version byte, the
 * header length (2 bytes little-endian in version 1, 4 bytes in versions 2 and 3), and a
 * Python dict literal such as
 * {@code {'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }}, padded with spaces
 * and a newline so that the data starts at a multiple of 64 bytes.
 *
 * <p>The dict is parsed by a restricted literal parser: strings, {@code True},
 * {@code False} and tuples of non-negative integers only, with exactly the keys
 * {@code descr}, {@code fortran_order} and {@code shape}. Nothing is evaluated, and
 * structured dtypes and Python objects (pickles) are rejected.
 *
 * @param majorVersion format major version, 1 to 3
 * @param minorVersion format minor version, 0
 * @param descr        the dtype descriptor as written, e.g. {@code "<f8"}
 * @param dtype        the parsed dtype
 * @param byteOrder    byte order of the data; native for one-byte types
 * @param fortranOrder {@code true} if the data is in column-major order
 * @param shape        the array shape
 * @param dataOffset   position of the first data byte in the file
 */
public record NpyHeader(
        int majorVersion,
        int minorVersion,
        String descr,
        DType dtype,
        ByteOrder byteOrder,
        boolean fortranOrder,
        Shape shape,
        long dataOffset) {

    /**
     * The magic string at the start of every {@code .npy} file.
     */
    static final byte[] MAGIC = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};

    /**
     * Data alignment of NumPy's writer.
     */
    static final int ARRAY_ALIGN = 64;

    /**
     * Spare header space NumPy reserves so the shape can grow in place.
     */
    static final int GROWTH_AXIS_MAX_DIGITS = 21;

    /**
     * Returns the number of data bytes: {@code shape.size() * dtype.size()}.
     *
     * @return the number of data bytes: {@code shape.size() * dtype.size()}
     */
    public long dataBytes() {
        return Math.multiplyExact(shape.size(), (long) dtype.size());
    }

    // ------------------------------------------------------------------ reading

    /**
     * Reads and validates a header, leaving the stream at the first data byte.
     *
     * @param in     the stream, positioned at the start of the file
     * @param limits limits on header size, rank and data size
     * @return the header
     * @throws NpyFormatException if the header is malformed, unsupported or over a limit
     * @throws IOException        if reading fails
     */
    public static NpyHeader read(InputStream in, NpyLimits limits) throws IOException {
        byte[] prefix = readFully(in, MAGIC.length + 2, "magic string");
        if (!Arrays.equals(Arrays.copyOf(prefix, MAGIC.length), MAGIC)) {
            throw new NpyFormatException("not a .npy file: the magic string \\x93NUMPY is missing");
        }
        int major = prefix[6] & 0xFF;
        int minor = prefix[7] & 0xFF;
        if (major < 1 || major > 3 || minor != 0) {
            throw new NpyFormatException("unsupported .npy format version " + major + "." + minor);
        }
        int lengthBytes = major == 1 ? 2 : 4;
        byte[] len = readFully(in, lengthBytes, "header length");
        long headerLength = 0;
        for (int i = lengthBytes - 1; i >= 0; i--) {
            headerLength = (headerLength << 8) | (len[i] & 0xFF);
        }
        if (headerLength > limits.maxHeaderBytes()) {
            throw new NpyFormatException("header of " + headerLength + " bytes exceeds the limit of "
                    + limits.maxHeaderBytes() + " bytes");
        }
        byte[] header = readFully(in, (int) headerLength, "header");
        String text = new String(header, major == 3 ? StandardCharsets.UTF_8 : StandardCharsets.ISO_8859_1);
        long dataOffset = MAGIC.length + 2L + lengthBytes + headerLength;
        return parse(text, major, minor, dataOffset, limits);
    }

    private static byte[] readFully(InputStream in, int n, String what) throws IOException {
        byte[] b = in.readNBytes(n);
        if (b.length < n) {
            throw new NpyFormatException("truncated .npy file: end of data while reading the " + what,
                    new EOFException());
        }
        return b;
    }

    /**
     * Parses the dict literal of a header.
     *
     * @param dict       the header text, e.g. {@code {'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }}
     * @param major      format major version
     * @param minor      format minor version
     * @param dataOffset position of the first data byte
     * @param limits     limits on rank and data size
     * @return the header
     * @throws NpyFormatException if the text is not a valid header
     */
    public static NpyHeader parse(String dict, int major, int minor, long dataOffset, NpyLimits limits) {
        Map<String, Object> d = new LiteralParser(dict).parseHeaderDict();
        if (!d.keySet().equals(java.util.Set.of("descr", "fortran_order", "shape"))) {
            throw new NpyFormatException("header must have exactly the keys 'descr', 'fortran_order' and 'shape', got "
                    + d.keySet());
        }
        if (!(d.get("descr") instanceof String descr)) {
            throw new NpyFormatException("structured dtypes are not supported (descr is " + d.get("descr") + ")");
        }
        if (!(d.get("fortran_order") instanceof Boolean fortran)) {
            throw new NpyFormatException("'fortran_order' must be True or False, got " + d.get("fortran_order"));
        }
        if (!(d.get("shape") instanceof long[] dims)) {
            throw new NpyFormatException("'shape' must be a tuple of integers, got " + d.get("shape"));
        }
        DType dtype;
        try {
            dtype = DType.fromNumpyDescriptor(descr);
        } catch (DTypeException e) {
            throw new NpyFormatException("unsupported dtype '" + descr + "'", e);
        }
        ByteOrder order = switch (descr.charAt(0)) {
            case '<' -> ByteOrder.LITTLE_ENDIAN;
            case '>' -> ByteOrder.BIG_ENDIAN;
            default -> ByteOrder.nativeOrder();
        };
        if (dims.length > limits.maxRank()) {
            throw new NpyFormatException("rank " + dims.length + " exceeds the limit of " + limits.maxRank());
        }
        Shape shape;
        try {
            shape = Shape.of(dims);
        } catch (RuntimeException e) {
            throw new NpyFormatException("invalid shape " + Arrays.toString(dims), e);
        }
        NpyHeader h = new NpyHeader(major, minor, descr, dtype, order, fortran, shape, dataOffset);
        long bytes;
        try {
            bytes = h.dataBytes();
        } catch (ArithmeticException e) {
            throw new NpyFormatException("data size of shape " + shape + " overflows long", e);
        }
        if (bytes > limits.maxDataBytes()) {
            throw new NpyFormatException("array data of " + bytes + " bytes exceeds the limit of "
                    + limits.maxDataBytes() + " bytes");
        }
        return h;
    }

    // ------------------------------------------------------------------ writing

    /**
     * Encodes a header exactly as NumPy's {@code np.save} does: keys in sorted order, 21
     * spare spaces for growing the shape, padding to a multiple of 64 bytes, version 1.0
     * unless the header needs more than 65,535 bytes.
     *
     * @param descr        dtype descriptor, e.g. {@code "<f8"}
     * @param fortranOrder whether data follows in column-major order
     * @param shape        the shape
     * @return the complete prefix: magic, version, length and padded header
     */
    public static byte[] encode(String descr, boolean fortranOrder, Shape shape) {
        long[] dims = shape.dimensions();
        StringBuilder h = new StringBuilder("{'descr': '").append(descr).append("', 'fortran_order': ")
                .append(fortranOrder ? "True" : "False").append(", 'shape': ").append(tuple(dims)).append(", }");
        if (dims.length > 0) {
            long growth = dims[fortranOrder ? dims.length - 1 : 0];
            h.append(" ".repeat(Math.max(0, GROWTH_AXIS_MAX_DIGITS - Long.toString(growth).length())));
        }
        byte[] v1 = wrap(h.toString(), 1);
        return v1 != null ? v1 : wrap(h.toString(), 2);
    }

    private static byte[] wrap(String header, int major) {
        int lengthBytes = major == 1 ? 2 : 4;
        int hlen = header.length() + 1;
        int padlen = ARRAY_ALIGN - ((MAGIC.length + 2 + lengthBytes + hlen) % ARRAY_ALIGN);
        long total = (long) hlen + padlen;
        if (major == 1 && total > 0xFFFF) {
            return null;
        }
        byte[] text = (header + " ".repeat(padlen) + "\n").getBytes(StandardCharsets.ISO_8859_1);
        byte[] out = new byte[MAGIC.length + 2 + lengthBytes + text.length];
        System.arraycopy(MAGIC, 0, out, 0, MAGIC.length);
        out[6] = (byte) major;
        out[7] = 0;
        for (int i = 0; i < lengthBytes; i++) {
            out[8 + i] = (byte) (total >>> (8 * i));
        }
        System.arraycopy(text, 0, out, 8 + lengthBytes, text.length);
        return out;
    }

    private static String tuple(long[] dims) {
        if (dims.length == 1) {
            return "(" + dims[0] + ",)";
        }
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < dims.length; i++) {
            sb.append(i == 0 ? "" : ", ").append(dims[i]);
        }
        return sb.append(")").toString();
    }

    // ------------------------------------------------------------------ literal parser

    /**
     * Recursive-descent parser for the literal subset used in .npy headers.
     */
    static final class LiteralParser {
        private final String s;
        private int pos;

        LiteralParser(String s) {
            this.s = s;
        }

        Map<String, Object> parseHeaderDict() {
            skipSpace();
            Map<String, Object> d = parseDict();
            skipSpace();
            if (pos != s.length()) {
                throw error("unexpected text after the header dict");
            }
            return d;
        }

        private Map<String, Object> parseDict() {
            expect('{');
            Map<String, Object> d = new LinkedHashMap<>();
            skipSpace();
            while (peek() != '}') {
                String key = parseString();
                skipSpace();
                expect(':');
                skipSpace();
                Object value = parseValue();
                if (d.put(key, value) != null) {
                    throw error("duplicate key '" + key + "'");
                }
                skipSpace();
                if (peek() == ',') {
                    pos++;
                    skipSpace();
                } else if (peek() != '}') {
                    throw error("expected ',' or '}'");
                }
            }
            expect('}');
            return d;
        }

        private Object parseValue() {
            char c = peek();
            if (c == '\'' || c == '"') {
                return parseString();
            }
            if (c == '(') {
                return parseTuple();
            }
            if (c == '[') {
                throw error("structured dtypes (a list as 'descr') are not supported");
            }
            if (s.startsWith("True", pos)) {
                pos += 4;
                return Boolean.TRUE;
            }
            if (s.startsWith("False", pos)) {
                pos += 5;
                return Boolean.FALSE;
            }
            throw error("unsupported value");
        }

        private long[] parseTuple() {
            expect('(');
            List<Long> values = new ArrayList<>();
            skipSpace();
            boolean sawComma = false;
            while (peek() != ')') {
                values.add(parseInt());
                skipSpace();
                if (peek() == ',') {
                    pos++;
                    sawComma = true;
                    skipSpace();
                } else if (peek() != ')') {
                    throw error("expected ',' or ')' in tuple");
                }
            }
            expect(')');
            if (values.size() == 1 && !sawComma) {
                throw error("a one-element tuple needs a trailing comma");
            }
            return values.stream().mapToLong(Long::longValue).toArray();
        }

        private long parseInt() {
            int start = pos;
            while (pos < s.length() && Character.isDigit(s.charAt(pos))) {
                pos++;
            }
            if (start == pos) {
                throw error("expected a non-negative integer");
            }
            if (pos - start > 19) {
                throw error("integer too large");
            }
            long v;
            try {
                v = Long.parseLong(s.substring(start, pos));
            } catch (NumberFormatException e) {
                throw error("integer too large");
            }
            if (pos < s.length() && (s.charAt(pos) == 'L' || s.charAt(pos) == 'l')) {
                pos++;  // Python 2 long suffix, written by old NumPy versions
            }
            return v;
        }

        private String parseString() {
            char quote = peek();
            if (quote != '\'' && quote != '"') {
                throw error("expected a string");
            }
            pos++;
            int start = pos;
            while (pos < s.length() && s.charAt(pos) != quote) {
                if (s.charAt(pos) == '\\') {
                    throw error("escape sequences are not supported");
                }
                pos++;
            }
            if (pos >= s.length()) {
                throw error("unterminated string");
            }
            return s.substring(start, pos++);
        }

        private void expect(char c) {
            if (peek() != c) {
                throw error("expected '" + c + "'");
            }
            pos++;
        }

        private char peek() {
            if (pos >= s.length()) {
                throw error("unexpected end of header");
            }
            return s.charAt(pos);
        }

        private void skipSpace() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
        }

        private NpyFormatException error(String message) {
            String shown = s.length() > 200 ? s.substring(0, 200) + "..." : s;
            return new NpyFormatException("malformed .npy header at position " + pos + ": " + message
                    + " in " + shown.strip());
        }
    }
}
