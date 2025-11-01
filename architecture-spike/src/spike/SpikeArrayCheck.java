package spike;

import java.util.Arrays;

/**
 * S2 — verifies {@link SpikeArray} and {@link Odometer} against hand-computed values
 * and checks that element access does not allocate.
 *
 * <p>Run: {@code java -cp out spike.SpikeArrayCheck}. Exits non-zero on failure.
 */
public final class SpikeArrayCheck {

    private static int failures;

    private SpikeArrayCheck() {
    }

    public static void main(String[] args) {
        for (boolean nat : new boolean[]{false, true}) {
            String mem = nat ? "native" : "heap";
            // a = arange(12).reshape(3, 4)
            SpikeArray a = SpikeArray.arange(12, nat).reshape(3, 4);
            expect(mem + " contiguous", a.toArray(), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
            expect(mem + " get(i,j)", new double[]{a.get(1, 2), a.get(2, 3)}, 6, 11);
            expect(mem + " strides", toDouble(a.strides), 32, 8);

            SpikeArray t = a.transpose();
            expect(mem + " transpose shape", toDouble(t.shape), 4, 3);
            expect(mem + " transpose strides", toDouble(t.strides), 8, 32);
            expect(mem + " transpose values", t.toArray(), 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11);
            check(mem + " transpose not C-contiguous", !t.isCContiguous());

            // a[::2] and a[::-1]
            expect(mem + " step 2", a.stepAxis0(0, 2, 2).toArray(), 0, 1, 2, 3, 8, 9, 10, 11);
            expect(mem + " step -1", a.stepAxis0(2, 3, -1).toArray(), 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);

            // broadcast a row (1, 4) to (2, 4) and a column of t
            SpikeArray row = SpikeArray.arange(4, nat).reshape(1, 4);
            SpikeArray b = row.broadcastTo(2, 4);
            expect(mem + " broadcast strides", toDouble(b.strides), 0, 8);
            expect(mem + " broadcast values", b.toArray(), 0, 1, 2, 3, 0, 1, 2, 3);
            expect(mem + " broadcast 1-D to 3-D", SpikeArray.arange(2, nat).broadcastTo(2, 2, 2).toArray(),
                    0, 1, 0, 1, 0, 1, 0, 1);

            // 0-D and empty
            SpikeArray scalar = SpikeArray.arange(1, nat).reshape();
            expect(mem + " 0-D", scalar.toArray(), 0);
            expect(mem + " empty", SpikeArray.zeros(nat, 0, 5).toArray());
        }
        allocationCheck();
        if (failures > 0) {
            System.out.println(failures + " check(s) FAILED");
            System.exit(1);
        }
        System.out.println("all checks passed");
    }

    /** Element access must be allocation-free once compiled; varargs access is not. */
    private static void allocationCheck() {
        SpikeArray a = SpikeArray.arange(1 << 16, false).reshape(256, 256);
        Bench.Result fixed = Bench.measure(2_000_000, i -> Bench.sinkDouble = a.get(i & 255, (i >>> 8) & 255));
        Bench.Result varargs = Bench.measure(2_000_000, i -> Bench.sinkDouble = a.getVarargs(i & 255, (i >>> 8) & 255));
        Bench.Result iterate = Bench.measure(200, i -> Bench.sinkDouble = sumViaOdometer(a.transpose()));
        System.out.println("get(i, j)        " + fixed);
        System.out.println("getVarargs(i, j) " + varargs);
        System.out.println("odometer sum     " + iterate + "  (65,536 elements per op, transposed)");
        check("get(i, j) allocation-free", fixed.bytesPerOp() < 1);
    }

    static double sumViaOdometer(SpikeArray a) {
        Odometer it = new Odometer(a.shape, new long[][]{a.strides}, new long[]{a.offset});
        double acc = 0;
        do {
            long p = it.offset(0);
            long s = it.innerStride(0);
            for (long k = 0, n = it.innerSize(); k < n; k++, p += s) {
                acc += a.segment.get(SpikeArray.F64, p);
            }
        } while (it.advance());
        return acc;
    }

    private static double[] toDouble(long[] v) {
        return Arrays.stream(v).asDoubleStream().toArray();
    }

    private static void expect(String name, double[] actual, double... expected) {
        check(name + " " + Arrays.toString(actual), Arrays.equals(actual, expected));
    }

    private static void check(String name, boolean ok) {
        System.out.println((ok ? "ok    " : "FAIL  ") + name);
        if (!ok) {
            failures++;
        }
    }
}
