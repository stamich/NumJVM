package org.numjvm.vector;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.FusedGraph;
import static org.junit.jupiter.api.Assertions.*;

/** Exercises generated fused bodies in this module's own coverage run. */
class FusedKernelOracleTest {
    private static FusedGraph graph(DType left, DType right) {
        DType result = left == DType.FLOAT64 || right == DType.FLOAT64 ? DType.FLOAT64 : DType.FLOAT32;
        var n = new ArrayList<FusedGraph.Node>();
        n.add(new FusedGraph.Node(FusedGraph.Op.INPUT, left, 0, 0, -1, -1));
        n.add(new FusedGraph.Node(FusedGraph.Op.INPUT, right, 1, 0, -1, -1));
        n.add(new FusedGraph.Node(FusedGraph.Op.CONSTANT, DType.FLOAT64, -1, Double.doubleToRawLongBits(.25), -1, -1));
        n.add(new FusedGraph.Node(FusedGraph.Op.ADD, left, -1, 0, 0, 2));
        n.add(new FusedGraph.Node(FusedGraph.Op.MULTIPLY, result, -1, 0, 3, 1));
        n.add(new FusedGraph.Node(FusedGraph.Op.ADD, result, -1, 0, 4, 3));
        n.add(new FusedGraph.Node(FusedGraph.Op.SUBTRACT, result, -1, 0, 5, 1));
        n.add(new FusedGraph.Node(FusedGraph.Op.DIVIDE, result, -1, 0, 6, 1));
        n.add(new FusedGraph.Node(FusedGraph.Op.NEGATIVE, result, -1, 0, 7, -1));
        n.add(new FusedGraph.Node(FusedGraph.Op.ABSOLUTE, result, -1, 0, 8, -1));
        n.add(new FusedGraph.Node(FusedGraph.Op.SQUARE, result, -1, 0, 9, -1));
        n.add(new FusedGraph.Node(FusedGraph.Op.SQRT, result, -1, 0, 10, -1));
        return new FusedGraph(n, List.of(left, right));
    }
    @Test void generatedGraphsHandleFloatDoubleMixedAndBroadcastTails() {
        var provider = new FusedVectorProvider();
        for (var pair : List.of(new DType[]{DType.FLOAT32, DType.FLOAT32},
                new DType[]{DType.FLOAT64, DType.FLOAT64}, new DType[]{DType.FLOAT32, DType.FLOAT64})) {
            Object left = pair[0] == DType.FLOAT32 ? filledFloat(35, 2) : filledDouble(35, 2);
            Object right = pair[1] == DType.FLOAT32 ? filledFloat(1, 3) : filledDouble(1, 3);
            Object output = pair[0] == DType.FLOAT32 && pair[1] == DType.FLOAT32 ? new float[35] : new double[35];
            var kernel = provider.compile(graph(pair[0], pair[1]));
            assertNotNull(kernel);
            kernel.run(new MemorySegment[]{segment(left), segment(right), segment(output)}, new long[3],
                    new long[]{pair[0].size(), 0, output instanceof float[] ? 4 : 8}, 35);
            for (int i = 0; i < 35; i++) assertEquals(2, output instanceof float[] f ? f[i] : ((double[]) output)[i], 1e-6);
        }
    }
    @Test void unsupportedGraphReturnsNull() {
        var bad = new FusedGraph(List.of(new FusedGraph.Node(FusedGraph.Op.INPUT, DType.FLOAT64, 1, 0, -1, -1)), List.of(DType.FLOAT64));
        assertNull(new FusedVectorProvider().compile(bad));
    }
    private static float[] filledFloat(int size, float value) { float[] a = new float[size]; java.util.Arrays.fill(a, value); return a; }
    private static double[] filledDouble(int size, double value) { double[] a = new double[size]; java.util.Arrays.fill(a, value); return a; }
    private static MemorySegment segment(Object value) {
        return value instanceof float[] f ? MemorySegment.ofArray(f) : MemorySegment.ofArray((double[]) value);
    }
}
