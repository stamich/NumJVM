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
import org.numjvm.dtype.DType;
import org.numjvm.exception.NumJVMException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.io.Npy;
import org.numjvm.kernels.ValidationKernels;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Slice;
import org.numjvm.shape.Broadcasting;

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
            assertArrayEquals(bytes(expected), bytes(actual), id + " values");
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
            default -> throw new AssertionError("unknown op " + op);
        };
    }

    /** The oracle's generator: (i * 2654435761) mod 2^32 / 2^32. */
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

    /** The bytes of an array in C order: exact comparison for every dtype, including complex and NaN. */
    static byte[] bytes(NDArray a) {
        NDArray c = a.copy();
        MemorySegment s = SharedSecrets.arrayAccess().buffer(c).segment();
        return s.asSlice(0, c.nbytes()).toArray(ValueLayout.JAVA_BYTE);
    }
}
