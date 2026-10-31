package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Slice;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class GraphLayoutMatrixTest {
    @Test void strategiesHandleContiguousFortranStridedBroadcastAndOutputViews() {
        for (DType type : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
            for (String layout : new String[]{"C", "F", "STRIDED", "REVERSED"}) {
                var source = NJ.arange(0, 64 * 130, 1).astype(type).reshapeView(64, 130);
                NDArray a = switch (layout) {
                    case "C" -> source.slice(Slice.all(), Slice.range(0, 65)).copy(Order.C);
                    case "F" -> source.slice(Slice.all(), Slice.range(0, 65)).copy(Order.F);
                    case "STRIDED" -> source.slice(Slice.all(), Slice.step(2));
                    default -> source.slice(Slice.all(), Slice.range(0, 65)).slice(Slice.all(), Slice.step(-1));
                };
                NDArray row = NJ.arange(1, 66, 1).astype(type).broadcastTo(64, 65);
                NDArray expected = Ufuncs.ADD.apply(Ufuncs.SQUARE.apply(Ufuncs.ADD.apply(a, row)), row);
                for (Jit.Mode mode : Jit.Mode.values()) {
                    NDArray outStorage = NJ.zeros(type, 64, 130);
                    NDArray out = layout.equals("F") ? NJ.zeros(type, 64, 65).copy(Order.F)
                            : outStorage.slice(Slice.all(), Slice.step(2));
                    if (mode == Jit.Mode.REQUIRED && !GeneratedExecutor.eligible(new NDArray[]{a, row}, out)) continue;
                    try (var context = Jit.context(mode)) {
                        var graph = context.trace(2, i -> {
                            var shared = i.input(0).add(i.input(1));
                            return shared.square().add(i.input(1));
                        });
                        graph.evalInto(out, a, row);
                        for (int r = 0; r < 64; r++) for (int c = 0; c < 65; c++)
                            assertEquals(expected.getDouble(r, c), out.getDouble(r, c), mode + "/" + layout);
                    }
                }
            }
        }
    }

    @Test void partialOverlapReadsSnapshotAcrossBlocksAndVectorRuns() {
        for (Jit.Mode mode : Jit.Mode.values()) {
            NDArray base = NJ.arange(0, 4161, 1).astype(DType.FLOAT64);
            NDArray input = base.slice(Slice.range(0, 4160));
            NDArray output = base.slice(Slice.range(1, 4161));
            try (var context = Jit.context(mode)) {
                var graph = context.trace(1, i -> i.input(0).square().add(i.input(0)));
                graph.evalInto(output, input);
                for (int i = 0; i < 4160; i++) assertEquals((double) i * i + i, output.getDouble(i));
            }
        }
    }

    @Test void emptyAndReadonlyOutputsAreHandledBeforeWrites() {
        for (Jit.Mode mode : Jit.Mode.values()) {
            try (var context = Jit.context(mode)) {
                var graph = context.trace(1, i -> i.input(0).square());
                assertEquals(0, graph.eval(NJ.zeros(0)).size());
                var readonly = NJ.ones(1).broadcastTo(2);
                assertThrows(ReadOnlyArrayException.class, () -> graph.evalInto(readonly, NJ.ones(2)));
            }
        }
    }
}
