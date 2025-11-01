package spike;

import java.lang.foreign.ValueLayout;

/**
 * S3 — correctness and speed of the strided add and the pairwise sum.
 *
 * <p>Run: {@code java -cp out spike.KernelSpike}. Exits non-zero on a failed check.
 */
public final class KernelSpike {

    private KernelSpike() {
    }

    public static void main(String[] args) {
        int failures = correctness();
        speed();
        if (failures > 0) {
            System.out.println(failures + " check(s) FAILED");
            System.exit(1);
        }
    }

    static int correctness() {
        int failures = 0;
        // (3, 4) + (4,) broadcast, and transposed + contiguous
        SpikeArray a = SpikeArray.arange(12, false).reshape(3, 4);
        SpikeArray row = SpikeArray.arange(4, true);
        double[] r1 = StridedAdd.add(a, row, false).toArray();
        double[] e1 = {0, 2, 4, 6, 4, 6, 8, 10, 8, 10, 12, 14};
        failures += report("add (3,4)+(4,)", java.util.Arrays.equals(r1, e1));

        SpikeArray t = a.transpose();
        SpikeArray tc = SpikeArray.zeros(false, 4, 3);
        double[] tv = t.toArray();
        for (int i = 0; i < tv.length; i++) {
            tc.segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, tv[i]);
        }
        double[] r2 = StridedAdd.add(t, tc, true).toArray();
        boolean ok2 = true;
        for (int i = 0; i < r2.length; i++) {
            ok2 &= r2[i] == 2 * tv[i];
        }
        failures += report("add transposed + contiguous", ok2);

        double[] r3 = StridedAdd.add(a.stepAxis0(2, 3, -1), SpikeArray.arange(3, false).reshape(3, 1), false).toArray();
        double[] e3 = {8, 9, 10, 11, 5, 6, 7, 8, 2, 3, 4, 5};
        failures += report("add reversed + column", java.util.Arrays.equals(r3, e3));

        // sum: exact for integer-valued data, closed form n(n-1)/2
        for (long n : new long[]{0, 1, 7, 8, 9, 127, 128, 129, 1000, 65_536, 1_048_577}) {
            SpikeArray v = SpikeArray.arange(n, n > 4096);
            double expected = (double) n * (n - 1) / 2;
            failures += report("pairwise sum n=" + n, SumReduction.sum(v) == expected);
        }
        failures += report("sum transposed", SumReduction.sum(t) == 66.0);
        failures += report("sum broadcast", SumReduction.sum(row.broadcastTo(5, 4)) == 30.0);

        // rounding: 1e8 copies of 0.1 would take too long; use 2^24 values of 0.1
        long n = 1 << 24;
        SpikeArray tenth = SpikeArray.zeros(true, n);
        tenth.segment.fill((byte) 0);
        for (long i = 0; i < n; i++) {
            tenth.segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, 0.1);
        }
        double exact = n * 0.1;
        double pw = SumReduction.sum(tenth);
        double nv = SumReduction.naive(tenth.segment, 0, n, 8);
        System.out.printf("      2^24 x 0.1: pairwise rel. error %.2e, naive rel. error %.2e%n",
                Math.abs(pw - exact) / exact, Math.abs(nv - exact) / exact);
        failures += report("pairwise more accurate than naive", Math.abs(pw - exact) < Math.abs(nv - exact));
        return failures;
    }

    static void speed() {
        System.out.println();
        System.out.println("== S3 speed, ns per element (manual harness)");
        System.out.printf("%-10s %-12s %-12s %-12s %-12s %-12s%n",
                "elements", "add fast", "add generic", "add transp.", "sum pairwise", "sum naive");
        for (long n : new long[]{1_024, 65_536, 1_048_576, 16_777_216}) {
            boolean nat = n > 4096;
            long side = (long) Math.sqrt(n);
            SpikeArray a = SpikeArray.arange(n, nat).reshape(side, n / side);
            SpikeArray b = SpikeArray.arange(n, nat).reshape(side, n / side);
            SpikeArray bt = SpikeArray.arange(n, nat).reshape(n / side, side).transpose();
            long ops = Math.max(5, 20_000_000L / n);
            double fast = Bench.measure(ops, i -> Bench.sinkObject = StridedAdd.add(a, b, nat)).nsPerOp() / n;
            double generic = Bench.measure(ops, i -> Bench.sinkObject = StridedAdd.addGeneric(a, b, nat)).nsPerOp() / n;
            double transposed = Bench.measure(ops, i -> Bench.sinkObject = StridedAdd.add(a, bt, nat)).nsPerOp() / n;
            double pw = Bench.measure(ops, i -> Bench.sinkDouble = SumReduction.sum(a)).nsPerOp() / n;
            double nv = Bench.measure(ops, i -> Bench.sinkDouble = SumReduction.naive(a.segment, 0, n, 8)).nsPerOp() / n;
            System.out.printf("%-10d %-12.3f %-12.3f %-12.3f %-12.3f %-12.3f%n", n, fast, generic, transposed, pw, nv);
            System.gc();
        }
    }

    private static int report(String name, boolean ok) {
        System.out.println((ok ? "ok    " : "FAIL  ") + name);
        return ok ? 0 : 1;
    }
}
