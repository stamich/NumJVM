package org.numjvm.linalg;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.iteration.IndexIterator;
import org.numjvm.ndarray.Shape;
import static org.junit.jupiter.api.Assertions.*;

/** Replays deterministic NumPy shapes and high-precision contraction references. */
class ContractionOracleTest {
    private static long[] shape(String text) {
        return text.equals("-") ? new long[0] : Arrays.stream(text.split(",")).mapToLong(Long::parseLong).toArray();
    }
    private static double[] values(String text) {
        return text.equals("-") ? new double[0] : Arrays.stream(text.split(",")).mapToDouble(Double::parseDouble).toArray();
    }
    @Test void referenceShapesDtypesAndNumericalBounds() throws Exception {
        int count = 0;
        try (var reader = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream("/contraction-oracle.tsv"), StandardCharsets.UTF_8))) {
            for (String line : reader.lines().toList()) {
                if (line.startsWith("#")) continue;
                String[] f = line.split("\t");
                var leftType = f[2].equals("float32") ? DType.FLOAT32 : DType.FLOAT64;
                var rightType = f[3].equals("float32") ? DType.FLOAT32 : DType.FLOAT64;
                try (var scope = NJ.scope()) {
                    var a = scope.array(values(f[7]), shape(f[4])).astype(leftType);
                    var b = scope.array(values(f[8]), shape(f[5])).astype(rightType);
                    if (f[4].equals("-")) a = a.reshapeView();
                    if (f[5].equals("-")) b = b.reshapeView();
                    int axis = Integer.parseInt(f[1]);
                    var actual = f[0].equals("dot") ? Linalg.dot(a, b) : Linalg.vecdot(a, b, axis);
                    assertEquals(Shape.of(shape(f[6])), actual.shape());
                    DType resultType = leftType == DType.FLOAT64 || rightType == DType.FLOAT64 ? DType.FLOAT64 : DType.FLOAT32;
                    assertEquals(resultType, actual.dtype());
                    double epsilon = resultType == DType.FLOAT32 ? Math.ulp(1.0f) : Math.ulp(1.0);
                    long k = f[0].equals("dot") && (a.ndim() == 0 || b.ndim() == 0) ? 1
                            : a.shape().dimension(f[0].equals("dot") ? -1 : axis);
                    double gamma = k * epsilon / (1 - k * epsilon);
                    double[] expected = values(f[9]), scales = values(f[10]);
                    var indices = new IndexIterator(actual.shape()); int i = 0;
                    while (indices.hasNext()) {
                        double floor = 8 * (resultType == DType.FLOAT32 ? Float.MIN_NORMAL : Double.MIN_NORMAL);
                        double bound = 8 * gamma * scales[i] + floor;
                        assertEquals(expected[i], actual.getDouble(indices.next()), bound, f[0] + " case " + count);
                        i++;
                    }
                }
                count++;
            }
        }
        assertEquals(30, count);
    }
}
