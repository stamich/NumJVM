package org.numjvm.io.oracle;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import org.numjvm.NJ;
import org.numjvm.dtype.Casting;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.TypePromotion;
import org.numjvm.exception.NumJVMException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.io.Npy;
import org.numjvm.kernels.ValidationKernels;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Slice;
import org.numjvm.shape.Broadcasting;
import org.numjvm.ufunc.BinaryUfunc;
import org.numjvm.ufunc.Reductions;
import org.numjvm.ufunc.Ufunc;
import org.numjvm.ufunc.UnaryUfunc;
import org.numjvm.ufunc.Ufuncs;

/**
 * Replays every case of the NumPy oracle ({@code numjvm-tests-oracle/fixtures/manifest.json})
 * and compares NumJVM's result with NumPy's.
 *
 * <p>Array results are also written to {@code numjvm.oracle.out} so that
 * {@code verify_results.py} can check that NumPy reads NumJVM's files back.
 */
class OracleTest {

    static Path fixtures() {
        String p = System.getProperty("numjvm.oracle.fixtures");
        return p != null ? Path.of(p) : Path.of("..", "numjvm-tests-oracle", "fixtures");
    }

    static Path out() throws IOException {
        String p = System.getProperty("numjvm.oracle.out");
        Path dir = p != null ? Path.of(p) : Path.of("build", "oracle-out");
        return Files.createDirectories(dir);
    }

    @SuppressWarnings("unchecked")
    @TestFactory
    Stream<DynamicTest> numpyOracle() throws IOException {
        Map<String, Object> manifest = (Map<String, Object>) Json.parse(
                Files.readString(fixtures().resolve("manifest.json"), StandardCharsets.UTF_8));
        List<Object> cases = (List<Object>) manifest.get("cases");
        Path out = out();
        List<DynamicTest> tests = new ArrayList<>();
        for (Object o : cases) {
            Map<String, Object> c = (Map<String, Object>) o;
            String name = "[" + c.get("tag") + "] " + c.get("id");
            tests.add(DynamicTest.dynamicTest(name, () -> run(c, out)));
        }
        assertTrue(tests.size() >= 100, "manifest has " + tests.size() + " cases");
        return tests.stream();
    }

    @SuppressWarnings("unchecked")
    private static void run(Map<String, Object> c, Path out) throws IOException {
        String id = (String) c.get("id");
        String op = (String) c.get("op");
        Map<String, Object> params = (Map<String, Object>) c.get("params");
        Map<String, Object> expect = (Map<String, Object>) c.get("expect");
        List<NDArray> in = new ArrayList<>();
        for (Object f : (List<Object>) c.get("inputs")) {
            in.add(Npy.read(fixtures().resolve((String) f)));
        }
        Supplier<Object> action = () -> apply(op, in, params);
        if (Boolean.TRUE.equals(expect.get("error"))) {
            assertThrows(NumJVMException.class, action::get, id);
            return;
        }
        Object result = action.get();
        if (expect.containsKey("file")) {
            NDArray actual = (NDArray) result;
            NDArray expected = Npy.read(fixtures().resolve((String) expect.get("file")));
            assertEquals(expected.dtype(), actual.dtype(), id + " dtype");
            assertEquals(expected.shape(), actual.shape(), id + " shape");
            if (params.containsKey("ulp")) {
                assertWithinUlps(id, expected, actual, ((Number) params.get("ulp")).longValue());
            } else {
                assertSameElements(id, expected, actual);
            }
            if (Boolean.TRUE.equals(expect.get("fortran")) && op.equals("copy")) {
                assertTrue(actual.isFContiguous(), id + " fortran order");
            }
            Npy.write(out.resolve(id + ".npy"), actual);
        } else if (expect.containsKey("value")) {
            double e = ((Number) expect.get("value")).doubleValue();
            double a = (Double) result;
            assertEquals(e, a, Math.abs(e) * 1e-12, id);
        } else if (expect.containsKey("repr")) {
            assertEquals(expect.get("repr"), result.toString(), id);
        } else if (expect.containsKey("flags")) {
            Map<String, Object> flags = (Map<String, Object>) expect.get("flags");
            NDArray a = (NDArray) result;
            assertEquals(flags.get("c"), a.isCContiguous(), id + " C_CONTIGUOUS");
            assertEquals(flags.get("f"), a.isFContiguous(), id + " F_CONTIGUOUS");
        } else if (expect.containsKey("dtype_name")) {
            assertEquals(DType.fromName((String) expect.get("dtype_name")), result, id);
        } else if (expect.containsKey("can_cast")) {
            Map<String, Object> table = (Map<String, Object>) expect.get("can_cast");
            DType from = DType.fromName((String) params.get("from"));
            Casting casting = Casting.valueOf(((String) params.get("casting")).toUpperCase(java.util.Locale.ROOT));
            for (Map.Entry<String, Object> e : table.entrySet()) {
                assertEquals(e.getValue(), TypePromotion.canCast(from, DType.fromName(e.getKey()), casting),
                        id + " -> " + e.getKey());
            }
        } else if (expect.containsKey("shape")) {
            assertArrayEquals(longs((List<Object>) expect.get("shape")), ((Shape) result).dimensions(), id);
        } else {
            throw new AssertionError("unknown expectation in " + id);
        }
    }

    @SuppressWarnings("unchecked")
    private static Object apply(String op, List<NDArray> in, Map<String, Object> p) {
        return switch (op) {
            case "transpose" -> in.get(0).transpose();
            case "permute_dims" -> in.get(0).permuteDims(ints((List<Object>) p.get("axes")));
            case "swap_axes" -> {
                int[] ax = ints((List<Object>) p.get("axes"));
                yield in.get(0).swapAxes(ax[0], ax[1]);
            }
            case "slice", "flags" -> in.get(0).slice(slices((List<Object>) p.get("index")));
            case "reshape" -> in.get(0).reshape(longs((List<Object>) p.get("shape")));
            case "reshape_transposed" -> in.get(0).transpose().reshape(longs((List<Object>) p.get("shape")));
            case "reshape_view_transposed" -> in.get(0).transpose().reshapeView(longs((List<Object>) p.get("shape")));
            case "broadcast_to" -> in.get(0).broadcastTo(longs((List<Object>) p.get("shape")));
            case "broadcast_shapes" -> {
                List<Object> shapes = (List<Object>) p.get("shapes");
                Shape[] s = new Shape[shapes.size()];
                for (int i = 0; i < s.length; i++) {
                    s[i] = Shape.of(longs((List<Object>) shapes.get(i)));
                }
                yield Broadcasting.broadcastShapes(s);
            }
            case "copy" -> {
                NDArray src = "transpose".equals(p.get("view")) ? in.get(0).transpose() : in.get(0);
                yield src.copy("F".equals(p.get("order")) ? Order.F : Order.C);
            }
            case "astype" -> in.get(0).astype(DType.fromNumpyDescriptor((String) p.get("dtype")));
            case "add" -> ValidationKernels.add(in.get(0), in.get(1));
            case "add_transposed" -> ValidationKernels.add(in.get(0).transpose(), in.get(1).transpose());
            case "sum" -> ValidationKernels.sumFloat64(
                    "transpose".equals(p.get("view")) ? in.get(0).transpose() : in.get(0));
            case "sum_generated" -> ValidationKernels.sumFloat64(hashUnit(((Number) p.get("n")).intValue()));
            case "repr" -> in.get(0);
            case "ufunc" -> ufunc(in, p);
            case "reduction" -> reduction(in, p);
            case "reduce" -> {
                BinaryUfunc u = (BinaryUfunc) Ufuncs.byName((String) p.get("ufunc")).orElseThrow();
                List<Object> axis = (List<Object>) p.get("axis");
                yield u.reduce(in.get(0), axis == null ? null : ints(axis), Boolean.TRUE.equals(p.get("keepdims")));
            }
            case "round" -> NJ.round(in.get(0), ((Number) p.get("decimals")).intValue());
            case "result_type" -> TypePromotion.resultType(((List<Object>) p.get("dtypes")).stream()
                    .map(o -> DType.fromName((String) o)).toArray(DType[]::new));
            case "can_cast" -> null;
            default -> throw new AssertionError("unknown op " + op);
        };
    }

    private static NDArray ufunc(List<NDArray> in, Map<String, Object> p) {
        Ufunc u = Ufuncs.byName((String) p.get("ufunc")).orElseThrow(() -> new AssertionError("no ufunc " + p));
        boolean transpose = "transpose".equals(p.get("view"));
        NDArray x = transpose ? in.get(0).transpose() : in.get(0);
        if (p.containsKey("scalar")) {
            BinaryUfunc b = (BinaryUfunc) u;
            Number v = (Number) p.get("scalar");
            boolean right = "right".equals(p.get("scalar_side"));
            if ("long".equals(p.get("scalar_type"))) {
                return right ? b.apply(x, v.longValue()) : b.apply(v.longValue(), x);
            }
            return right ? b.apply(x, v.doubleValue()) : b.apply(v.doubleValue(), x);
        }
        if (u instanceof BinaryUfunc b) {
            NDArray y = transpose ? in.get(1).transpose() : in.get(1);
            return b.apply(x, y);
        }
        return ((UnaryUfunc) u).apply(x);
    }

    @SuppressWarnings("unchecked")
    private static NDArray reduction(List<NDArray> in, Map<String, Object> p) {
        NDArray x;
        if (p.containsKey("generator")) {
            x = hashUnit(((Number) p.get("n")).intValue()).astype(DType.fromName((String) p.get("dtype")));
            if (p.containsKey("shape")) {
                x = x.reshape(longs((List<Object>) p.get("shape")));
            }
        } else {
            x = "transpose".equals(p.get("view")) ? in.get(0).transpose() : in.get(0);
        }
        List<Object> axis = (List<Object>) p.get("axis");
        int[] axes = axis == null ? null : ints(axis);
        boolean keep = Boolean.TRUE.equals(p.get("keepdims"));
        return switch ((String) p.get("reduction")) {
            case "sum" -> Reductions.sum(x, axes, keep);
            case "prod" -> Reductions.prod(x, axes, keep);
            case "max" -> Reductions.max(x, axes, keep);
            case "min" -> Reductions.min(x, axes, keep);
            case "mean" -> Reductions.mean(x, axes, keep);
            case "any" -> Reductions.any(x, axes, keep);
            case "all" -> Reductions.all(x, axes, keep);
            default -> throw new AssertionError("unknown reduction " + p);
        };
    }

    /**
     * Exact comparison of two arrays of the same dtype and shape: integers and bools byte for
     * byte; floats value by value with signed zeros distinguished and any NaN equal to any NaN
     * (NaN payloads are not part of NumPy's or NumJVM's contract).
     */
    static void assertSameElements(String id, NDArray expected, NDArray actual) {
        DType t = expected.dtype();
        if (!t.isFloatingPoint()) {
            assertArrayEquals(bytes(expected), bytes(actual), id + " values");
            return;
        }
        NDArray e = expected.copy().reshape(-1);
        NDArray a = actual.copy().reshape(-1);
        for (long i = 0; i < e.size(); i++) {
            double x = e.getDouble(i);
            double y = a.getDouble(i);
            if (Double.isNaN(x) && Double.isNaN(y)) {
                continue;
            }
            long k = i;
            assertEquals(Double.doubleToLongBits(x), Double.doubleToLongBits(y),
                    () -> id + " element " + k + ": expected " + x + " but was " + y);
        }
    }

    /**
     * Floating-point comparison with a tolerance in units in the last place of the dtype
     * (decision D6, for transcendental functions); NaN equals NaN, and -0.0 equals 0.0.
     */
    static void assertWithinUlps(String id, NDArray expected, NDArray actual, long maxUlps) {
        DType t = expected.dtype();
        if (!t.isFloatingPoint()) {
            assertSameElements(id, expected, actual);
            return;
        }
        NDArray e = expected.copy().reshape(-1);
        NDArray a = actual.copy().reshape(-1);
        for (long i = 0; i < e.size(); i++) {
            double x = e.getDouble(i);
            double y = a.getDouble(i);
            if (Double.isNaN(x) && Double.isNaN(y)) {
                continue;
            }
            long d = Math.abs(line(x, t) - line(y, t));
            long k = i;
            assertTrue(!Double.isNaN(x) && !Double.isNaN(y) && d <= maxUlps,
                    () -> id + " element " + k + ": expected " + x + " but was " + y + " (" + d + " ulp)");
        }
    }

    /**
     * Position of {@code v} on a monotonic integer line of {@code t}'s values: adjacent values
     * differ by 1.
     */
    static long line(double v, DType t) {
        long bits;
        int width;
        switch (t) {
            case FLOAT64 -> {
                bits = Double.doubleToRawLongBits(v);
                width = 64;
            }
            case FLOAT32 -> {
                bits = Float.floatToRawIntBits((float) v) & 0xFFFF_FFFFL;
                width = 32;
            }
            case FLOAT16 -> {
                bits = Float.floatToFloat16((float) v) & 0xFFFFL;
                width = 16;
            }
            default -> {
                bits = (Float.floatToRawIntBits((float) v) >>> 16) & 0xFFFFL;
                width = 16;
            }
        }
        long magnitude = width == 64 ? bits & Long.MAX_VALUE : bits & ((1L << (width - 1)) - 1);
        boolean negative = width == 64 ? bits < 0 : (bits >>> (width - 1)) != 0;
        return negative ? -magnitude : magnitude;
    }

    /**
     * The oracle's generator: (i * 2654435761) mod 2^32 / 2^32.
     */
    static NDArray hashUnit(int n) {
        NDArray a = NJ.zeros(n);
        for (int i = 0; i < n; i++) {
            a.setDouble(i, ((i * 2654435761L) & 0xFFFF_FFFFL) / 4294967296.0);
        }
        return a;
    }

    private static Slice[] slices(List<Object> specs) {
        Slice[] out = new Slice[specs.size()];
        for (int i = 0; i < out.length; i++) {
            String s = (String) specs.get(i);
            if (s.equals("None")) {
                out[i] = Slice.newAxis();
            } else if (s.equals("...")) {
                out[i] = Slice.ellipsis();
            } else if (!s.contains(":")) {
                out[i] = Slice.index(Long.parseLong(s));
            } else {
                String[] parts = s.split(":", -1);
                Long start = parts[0].isEmpty() ? null : Long.valueOf(parts[0]);
                Long stop = parts[1].isEmpty() ? null : Long.valueOf(parts[1]);
                long step = parts.length > 2 && !parts[2].isEmpty() ? Long.parseLong(parts[2]) : 1;
                out[i] = new Slice.Range(start, stop, step);
            }
        }
        return out;
    }

    private static long[] longs(List<Object> l) {
        return l.stream().mapToLong(o -> ((Number) o).longValue()).toArray();
    }

    private static int[] ints(List<Object> l) {
        return l.stream().mapToInt(o -> ((Number) o).intValue()).toArray();
    }

    /**
     * The bytes of an array in C order: exact comparison for every dtype, including complex and NaN.
     */
    static byte[] bytes(NDArray a) {
        NDArray c = a.copy();
        MemorySegment s = SharedSecrets.arrayAccess().buffer(c).segment();
        return s.asSlice(0, c.nbytes()).toArray(ValueLayout.JAVA_BYTE);
    }
}
