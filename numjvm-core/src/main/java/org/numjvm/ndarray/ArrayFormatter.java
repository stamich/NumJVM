package org.numjvm.ndarray;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

import org.numjvm.dtype.DType;
import org.numjvm.dtype.Unsigned;
import org.numjvm.internal.HalfFloat;

/**
 * NumPy-style {@code repr} of an array ({@code NUMPY_COMPATIBILITY}, simplified).
 *
 * <p>Follows NumPy's default print options: line width 75, summarisation above 1000
 * elements with 3 edge items per axis, up to 8 significant fractional digits, the
 * shortest representation that round-trips through the element type, and column
 * alignment. Floating-point values switch to scientific notation when the largest
 * magnitude is at least 1e8, the smallest non-zero magnitude is below 1e-4, or their
 * ratio exceeds 1000. The dtype is printed unless it is float64, int64 or bool.
 */
final class ArrayFormatter {

    static final int LINE_WIDTH = 75;
    static final int THRESHOLD = 1000;
    static final int EDGE_ITEMS = 3;
    static final int PRECISION = 8;

    private ArrayFormatter() {
    }

    static String repr(NDArray a) {
        String dtypeSuffix = defaultDType(a.dtype) ? "" : "dtype=" + a.dtype.typeName();
        if (!a.isAlive()) {
            return "array(<inaccessible: " + a.buffer.owner().describe() + " is closed>, shape=" + a.shape
                    + ", dtype=" + a.dtype.typeName() + ")";
        }
        if (a.size() == 0) {
            String shape = a.ndim() == 1 ? "" : "shape=" + tuple(a.shape.dims) + ", ";
            return "array([], " + shape + "dtype=" + a.dtype.typeName() + ")";
        }
        boolean summarise = a.size() > THRESHOLD;
        Formatter f = formatterFor(a, summarise);
        String prefix = "array(";
        StringBuilder sb = new StringBuilder(prefix);
        if (a.ndim() == 0) {
            sb.append(f.format(new long[0]).strip());
        } else {
            long[] index = new long[a.ndim()];
            formatAxis(a, f, index, 0, " ".repeat(prefix.length()), summarise, sb, new StringBuilder("["));
        }
        if (summarise) {
            // NumPy 2 prints the shape of summarised arrays, since it cannot be read off the values.
            dtypeSuffix = "shape=" + tuple(a.shape.dims) + (dtypeSuffix.isEmpty() ? "" : ", " + dtypeSuffix);
        }
        if (!dtypeSuffix.isEmpty()) {
            sb.append(",");
            // NumPy moves the dtype to its own line when it would not fit.
            int lastLine = sb.length() - sb.lastIndexOf("\n") - 1;
            if (lastLine + dtypeSuffix.length() + 2 > LINE_WIDTH) {
                sb.append("\n").append(" ".repeat(prefix.length()));
            } else {
                sb.append(" ");
            }
            sb.append(dtypeSuffix);
        }
        return sb.append(")").toString();
    }

    private static boolean defaultDType(DType t) {
        return t == DType.FLOAT64 || t == DType.INT64 || t == DType.BOOL || t == DType.COMPLEX128;
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

    /** Indices shown along an axis: all, or the edge items with -1 marking the ellipsis. */
    private static long[] shownIndices(long n, boolean summarise) {
        if (!summarise || n <= 2L * EDGE_ITEMS) {
            long[] all = new long[(int) n];
            for (int i = 0; i < n; i++) {
                all[i] = i;
            }
            return all;
        }
        long[] r = new long[2 * EDGE_ITEMS + 1];
        for (int i = 0; i < EDGE_ITEMS; i++) {
            r[i] = i;
            r[EDGE_ITEMS + 1 + i] = n - EDGE_ITEMS + i;
        }
        r[EDGE_ITEMS] = -1;
        return r;
    }

    /**
     * Appends the sub-array at {@code axis}. {@code line} holds the current, unfinished line
     * (starting with its opening brackets), as in NumPy's {@code _formatArray}.
     */
    private static void formatAxis(NDArray a, Formatter f, long[] index, int axis, String indent,
                                   boolean summarise, StringBuilder out, StringBuilder line) {
        int ndim = a.ndim();
        long[] shown = shownIndices(a.shape.dims[axis], summarise);
        if (axis == ndim - 1) {
            String hanging = indent + " ".repeat(axis + 1);
            for (int k = 0; k < shown.length; k++) {
                String word;
                if (shown[k] < 0) {
                    word = "...";
                } else {
                    index[axis] = shown[k];
                    word = f.format(index);
                }
                boolean last = k == shown.length - 1;
                String piece = word + (last ? "]" : ",");
                int column = out.length() - out.lastIndexOf("\n") - 1;
                if (column + line.length() + piece.length() + (last ? axis + 1 : 0) > LINE_WIDTH
                        && line.toString().strip().length() > axis + 1) {
                    out.append(stripTrailing(line)).append("\n");
                    line.setLength(0);
                    line.append(hanging);
                }
                line.append(piece);
                if (!last) {
                    line.append(" ");
                }
            }
            out.append(line);
            line.setLength(0);
            return;
        }
        String nextIndent = indent + " ";
        String separator = "," + "\n".repeat(ndim - axis - 1);
        for (int k = 0; k < shown.length; k++) {
            if (k > 0) {
                out.append(separator).append(nextIndent).append(" ".repeat(axis));
            }
            if (shown[k] < 0) {
                out.append("...");
                continue;
            }
            index[axis] = shown[k];
            StringBuilder child = new StringBuilder(k == 0 ? line.toString() : "").append("[");
            if (k == 0) {
                line.setLength(0);
            }
            formatAxis(a, f, index, axis + 1, indent, summarise, out, child);
        }
        out.append("]");
    }

    private static String stripTrailing(StringBuilder sb) {
        int end = sb.length();
        while (end > 0 && sb.charAt(end - 1) == ' ') {
            end--;
        }
        return sb.substring(0, end);
    }

    // ------------------------------------------------------------------ element formatting

    private interface Formatter {
        String format(long[] index);
    }

    private static Formatter formatterFor(NDArray a, boolean summarise) {
        List<long[]> sample = shownElements(a, summarise);
        DType t = a.dtype;
        if (t.isBool()) {
            return idx -> a.getBoolean(idx) ? " True" : "False";
        }
        if (t.isInteger()) {
            int width = 0;
            for (long[] idx : sample) {
                width = Math.max(width, intString(a, idx).length());
            }
            int w = width;
            return idx -> pad(intString(a, idx), w);
        }
        if (t.isComplex()) {
            return idx -> complexString(a, idx);
        }
        double[] values = new double[sample.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = a.getDouble(sample.get(i));
        }
        FloatFormat ff = FloatFormat.of(values, t);
        return idx -> ff.format(a.getDouble(idx));
    }

    private static String intString(NDArray a, long[] idx) {
        long v = a.getLong(idx);
        return a.dtype == DType.UINT64 ? Unsigned.toString(v) : Long.toString(v);
    }

    private static String pad(String s, int width) {
        return s.length() >= width ? s : " ".repeat(width - s.length()) + s;
    }

    private static String complexString(NDArray a, long[] idx) {
        long off = a.elementOffset(idx);
        boolean single = a.dtype == DType.COMPLEX64;
        double re = single ? a.segment.get(ElementCodec.F32, off) : a.segment.get(ElementCodec.F64, off);
        double im = single ? a.segment.get(ElementCodec.F32, off + 4) : a.segment.get(ElementCodec.F64, off + 8);
        DType component = single ? DType.FLOAT32 : DType.FLOAT64;
        String imag = FloatFormat.of(new double[]{Math.abs(im)}, component).format(Math.abs(im)).strip();
        String sign = im < 0 || (im == 0 && Double.doubleToRawLongBits(im) < 0) ? "-" : "+";
        return FloatFormat.of(new double[]{re}, component).format(re).strip() + sign + imag + "j";
    }

    /** Index tuples of the elements that will be printed. */
    private static List<long[]> shownElements(NDArray a, boolean summarise) {
        List<long[]> out = new ArrayList<>();
        int ndim = a.ndim();
        long[][] shown = new long[ndim][];
        for (int d = 0; d < ndim; d++) {
            shown[d] = shownIndices(a.shape.dims[d], summarise);
        }
        int[] pos = new int[ndim];
        while (true) {
            long[] idx = new long[ndim];
            boolean ellipsis = false;
            for (int d = 0; d < ndim; d++) {
                idx[d] = shown[d][pos[d]];
                ellipsis |= idx[d] < 0;
            }
            if (!ellipsis) {
                out.add(idx);
            }
            int d = ndim - 1;
            while (d >= 0 && ++pos[d] == shown[d].length) {
                pos[d] = 0;
                d--;
            }
            if (d < 0) {
                return out;
            }
        }
    }

    /** Column-aligned floating-point formatting for one array. */
    private static final class FloatFormat {
        private final boolean scientific;
        private final DType type;
        private final int intWidth;
        private final int fracWidth;
        private final int expWidth;

        private FloatFormat(boolean scientific, DType type, int intWidth, int fracWidth, int expWidth) {
            this.scientific = scientific;
            this.type = type;
            this.intWidth = intWidth;
            this.fracWidth = fracWidth;
            this.expWidth = expWidth;
        }

        static FloatFormat of(double[] values, DType type) {
            double maxAbs = 0;
            double minAbs = Double.POSITIVE_INFINITY;
            for (double v : values) {
                if (Double.isFinite(v) && v != 0) {
                    maxAbs = Math.max(maxAbs, Math.abs(v));
                    minAbs = Math.min(minAbs, Math.abs(v));
                }
            }
            boolean sci = maxAbs >= 1e8 || (maxAbs > 0 && (minAbs < 1e-4 || maxAbs / minAbs > 1e3));
            int iw = 0;
            int fw = 0;
            int ew = 0;
            boolean special = false;
            boolean negInf = false;
            for (double v : values) {
                if (!Double.isFinite(v)) {
                    special = true;
                    negInf |= v == Double.NEGATIVE_INFINITY;
                    continue;
                }
                String[] parts = sci ? scientificParts(v, type) : positionalParts(v, type);
                iw = Math.max(iw, parts[0].length());
                fw = Math.max(fw, parts[1].length());
                if (sci) {
                    ew = Math.max(ew, parts[2].length() - 1);
                }
            }
            if (sci) {
                ew = Math.max(ew, 2);
            }
            if (special) {
                // As NumPy: nan/inf are right-aligned in the width of the finite values.
                int offset = fw + 1 + (sci ? 2 + ew : 0);
                iw = Math.max(iw, Math.max(3 - offset, 3 + (negInf ? 1 : 0) - offset));
            }
            return new FloatFormat(sci, type, iw, fw, ew);
        }

        String format(double v) {
            if (!Double.isFinite(v)) {
                int total = intWidth + 1 + fracWidth + (scientific ? 2 + expWidth : 0);
                return pad(special(v), total);
            }
            String[] parts = scientific ? scientificParts(v, type) : positionalParts(v, type);
            StringBuilder sb = new StringBuilder();
            sb.append(" ".repeat(Math.max(0, intWidth - parts[0].length()))).append(parts[0]).append('.');
            if (scientific) {
                // Scientific mantissas are padded with zeros to a common precision, exponents to two digits.
                sb.append(parts[1]).append("0".repeat(fracWidth - parts[1].length()));
                String exp = parts[2];
                sb.append('e').append(exp.charAt(0)).append("0".repeat(Math.max(0, expWidth - (exp.length() - 1))))
                        .append(exp.substring(1));
            } else {
                sb.append(parts[1]).append(" ".repeat(fracWidth - parts[1].length()));
            }
            return sb.toString();
        }

        private static String special(double v) {
            return Double.isNaN(v) ? "nan" : v > 0 ? "inf" : "-inf";
        }

        /** {integer part with sign, fractional digits} of the shortest round-tripping decimal. */
        private static String[] positionalParts(double v, DType type) {
            BigDecimal d = shortest(v, type);
            BigDecimal rounded = d.setScale(Math.min(PRECISION, Math.max(0, d.scale())), RoundingMode.HALF_EVEN);
            String plain = rounded.stripTrailingZeros().abs().toPlainString();
            int dot = plain.indexOf('.');
            String ip = dot < 0 ? plain : plain.substring(0, dot);
            String fp = dot < 0 ? "" : plain.substring(dot + 1);
            boolean negative = v < 0 || (v == 0 && Double.doubleToRawLongBits(v) < 0);
            return new String[]{(negative ? "-" : "") + ip, fp};
        }

        /** {leading digit with sign, mantissa fraction, signed exponent} for scientific notation. */
        private static String[] scientificParts(double v, DType type) {
            if (v == 0) {
                return new String[]{Double.doubleToRawLongBits(v) < 0 ? "-0" : "0", "", "+0"};
            }
            BigDecimal d = shortest(v, type).abs();
            int exponent = d.precision() - d.scale() - 1;
            BigDecimal mantissa = d.movePointLeft(exponent).round(new MathContext(PRECISION + 1, RoundingMode.HALF_EVEN));
            String m = mantissa.stripTrailingZeros().toPlainString();
            int dot = m.indexOf('.');
            String lead = dot < 0 ? m : m.substring(0, dot);
            String frac = dot < 0 ? "" : m.substring(dot + 1);
            String exp = (exponent < 0 ? "-" : "+") + Math.abs(exponent);
            return new String[]{(v < 0 ? "-" : "") + lead, frac, exp};
        }

        /** Shortest decimal that converts back to the same value of {@code type}. */
        private static BigDecimal shortest(double v, DType type) {
            if (type == DType.FLOAT64) {
                return new BigDecimal(Double.toString(v));
            }
            if (type == DType.FLOAT32) {
                return new BigDecimal(Float.toString((float) v));
            }
            BigDecimal exact = new BigDecimal(v);
            for (int p = 1; p < 17; p++) {
                BigDecimal candidate = exact.round(new MathContext(p, RoundingMode.HALF_EVEN));
                if (roundTrip(candidate.doubleValue(), type) == v) {
                    return candidate;
                }
            }
            return exact;
        }

        private static double roundTrip(double x, DType type) {
            return switch (type) {
                case FLOAT16 -> Float.float16ToFloat(Float.floatToFloat16((float) x));
                case BFLOAT16 -> HalfFloat.bfloat16ToFloat(HalfFloat.floatToBFloat16((float) x));
                default -> (float) x;
            };
        }
    }
}
