package org.numjvm.io.oracle;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ufunc.BinaryUfunc;
import org.numjvm.ufunc.UnaryUfunc;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

/** Replays NumPy-generated periodic values over arrays large enough to use workers. */
class ParallelOracleTest {
    @SuppressWarnings("unchecked")
    @Test void parallelUfuncsAndSumsMatchNumpy() throws Exception {
        var path = OracleTest.fixtures().getParent().resolve("parallel-reference.json");
        var root = (Map<String, Object>) Json.parse(Files.readString(path));
        int size = ((Number) root.get("size")).intValue(), period = ((Number) root.get("period")).intValue();
        var cases = (Map<String, Object>) root.get("cases");
        for (var dtype : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
            var values = (Map<String, Object>) cases.get(dtype == DType.FLOAT32 ? "float32" : "float64");
            var x = (List<Object>) values.get("x"); var y = (List<Object>) values.get("y");
            double[] a = new double[size], b = new double[size];
            for (int i = 0; i < size; i++) { a[i] = ((Number) x.get(i % period)).doubleValue(); b[i] = ((Number) y.get(i % period)).doubleValue(); }
            var left = NJ.array(a).astype(dtype); var right = NJ.array(b).astype(dtype);
            try (var pool = NJ.parallel(3)) {
                for (var item : Map.<String, BinaryUfunc>of("add", Ufuncs.ADD, "subtract", Ufuncs.SUBTRACT,
                        "multiply", Ufuncs.MULTIPLY, "divide", Ufuncs.DIVIDE).entrySet()) {
                    var out = NJ.zeros(dtype, size);
                    pool.apply(item.getValue(), left, right, out);
                    assertEquals("parallel", pool.lastDecision().strategy());
                    check(out, (List<Object>) values.get(item.getKey()), period, dtype);
                }
                for (var item : Map.<String, UnaryUfunc>of("negative", Ufuncs.NEGATIVE, "absolute", Ufuncs.ABSOLUTE,
                        "square", Ufuncs.SQUARE, "sqrt", Ufuncs.SQRT).entrySet()) {
                    var out = NJ.zeros(dtype, size);
                    pool.apply(item.getValue(), left, out);
                    assertEquals("parallel", pool.lastDecision().strategy());
                    check(out, (List<Object>) values.get(item.getKey()), period, dtype);
                }
                double expected = ((Number) values.get("sum")).doubleValue();
                assertEquals(expected, pool.sum(left).getDouble(), dtype == DType.FLOAT32 ? 0.1 : 1e-8);
            }
        }
    }
    private static void check(org.numjvm.ndarray.NDArray actual, List<Object> expected, int period, DType type) {
        for (long i = 0; i < actual.size(); i++) {
            double value = ((Number) expected.get((int) (i % period))).doubleValue();
            assertEquals(value, actual.getDouble(i), type == DType.FLOAT32 ? 1e-6 : 1e-14, "index " + i);
        }
    }
}
