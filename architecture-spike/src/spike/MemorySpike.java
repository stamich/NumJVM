package spike;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * S1 — heap vs native allocation prototype.
 *
 * <p>Answers three questions for ADR-001:
 * <ol>
 *   <li>Can a {@code byte[]}-backed heap segment be read with aligned {@code double} access?
 *       (Expected: no. Heap segments only guarantee the alignment of their array element.)</li>
 *   <li>What does allocation cost for a {@code long[]}-backed heap segment versus the
 *       confined, shared and automatic arenas, from 16 to 16M float64 elements?</li>
 *   <li>Is there a measurable difference between aligned and unaligned value layouts
 *       when the memory is in fact aligned?</li>
 * </ol>
 *
 * <p>Run: {@code java -cp out spike.MemorySpike}
 */
public final class MemorySpike {

    static final long[] SIZES = {16, 128, 1_024, 65_536, 1_048_576, 16_777_216};

    private MemorySpike() {
    }

    public static void main(String[] args) {
        boolean all = args.length == 0;
        if (all || args[0].equals("alignment")) {
            alignmentCheck();
        }
        if (all || args[0].equals("allocation")) {
            allocationCost();
        }
        if (all || args[0].equals("access")) {
            accessCost();
        }
        if (all || args[0].equals("crossover")) {
            crossover();
        }
    }

    /** Locates the size where automatic native allocation becomes cheaper than heap. */
    static void crossover() {
        System.out.println("== S1.4 heap vs auto crossover (bytes per array)");
        System.out.printf("%-10s %-14s %-14s %-14s%n", "bytes", "heap ns/op", "auto ns/op", "confined ns/op");
        for (long bytes = 8 * 1024; bytes <= 8 * 1024 * 1024; bytes *= 2) {
            long b = bytes;
            long ops = Math.max(20, 20_000_000L / bytes);
            double heap = Bench.measure(ops, i -> Bench.sinkObject = MemorySegment.ofArray(new long[(int) (b / 8)])).nsPerOp();
            double auto = Bench.measure(ops, i -> Bench.sinkObject = Arena.ofAuto().allocate(b, 64)).nsPerOp();
            double confined = Bench.measure(ops, i -> {
                try (Arena a = Arena.ofConfined()) {
                    Bench.sinkObject = a.allocate(b, 64);
                }
            }).nsPerOp();
            System.out.printf("%-10d %,14.0f %,14.0f %,14.0f%n", bytes, heap, auto, confined);
        }
        System.out.println();
    }

    // ------------------------------------------------------------------ 1. alignment

    static void alignmentCheck() {
        System.out.println("== S1.1 alignment of heap segments");
        MemorySegment bytes = MemorySegment.ofArray(new byte[64]);
        MemorySegment longs = MemorySegment.ofArray(new long[8]);
        System.out.println("byte[] segment, JAVA_DOUBLE get:           " + tryRead(bytes, ValueLayout.JAVA_DOUBLE));
        System.out.println("byte[] segment, JAVA_DOUBLE_UNALIGNED get: " + tryRead(bytes, ValueLayout.JAVA_DOUBLE_UNALIGNED));
        System.out.println("long[] segment, JAVA_DOUBLE get:           " + tryRead(longs, ValueLayout.JAVA_DOUBLE));
        System.out.println("long[] segment, JAVA_DOUBLE get at +4:     " + tryReadAt(longs, ValueLayout.JAVA_DOUBLE, 4));
        System.out.println("long[] segment maxByteAlignment:           " + longs.maxByteAlignment());
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment n = arena.allocate(64, 64);
            System.out.println("native segment maxByteAlignment:           " + n.maxByteAlignment());
        }
        System.out.println();
    }

    private static String tryRead(MemorySegment s, ValueLayout.OfDouble layout) {
        return tryReadAt(s, layout, 0);
    }

    private static String tryReadAt(MemorySegment s, ValueLayout.OfDouble layout, long offset) {
        try {
            return "ok (" + s.get(layout, offset) + ")";
        } catch (IllegalArgumentException e) {
            return "FAILS: " + e.getClass().getSimpleName() + ": " + e.getMessage();
        }
    }

    // ------------------------------------------------------------------ 2. allocation

    static void allocationCost() {
        System.out.println("== S1.2 allocation cost (allocate + release, float64 elements)");
        System.out.printf("%-12s %-10s %s%n", "elements", "strategy", "time / heap bytes per op");
        for (long n : SIZES) {
            long bytes = n * Double.BYTES;
            long ops = Bench.opsFor(n) / 4 + 3;
            report(n, "heap", Bench.measure(ops, i -> {
                MemorySegment s = MemorySegment.ofArray(new long[(int) n]);
                Bench.sinkObject = s;
            }));
            report(n, "confined", Bench.measure(ops, i -> {
                try (Arena a = Arena.ofConfined()) {
                    MemorySegment s = a.allocate(bytes, 64);
                    Bench.sinkDouble = s.get(ValueLayout.JAVA_DOUBLE, 0);
                }
            }));
            report(n, "shared", Bench.measure(Math.max(3, ops / 4), i -> {
                try (Arena a = Arena.ofShared()) {
                    MemorySegment s = a.allocate(bytes, 64);
                    Bench.sinkDouble = s.get(ValueLayout.JAVA_DOUBLE, 0);
                }
            }));
            report(n, "auto", Bench.measure(Math.max(3, ops / 4), i -> {
                MemorySegment s = Arena.ofAuto().allocate(bytes, 64);
                Bench.sinkDouble = s.get(ValueLayout.JAVA_DOUBLE, 0);
            }));
            System.gc();
        }
        System.out.println();
    }

    private static void report(long n, String strategy, Bench.Result r) {
        System.out.printf("%-12d %-10s %s%n", n, strategy, r);
    }

    // ------------------------------------------------------------------ 3. access

    static void accessCost() {
        System.out.println("== S1.3 sequential read, aligned vs unaligned layout (ns per element)");
        long n = 1_048_576;
        MemorySegment heap = MemorySegment.ofArray(new long[(int) n]);
        MemorySegment nat = Arena.ofAuto().allocate(n * 8, 64);
        for (long i = 0; i < n; i++) {
            heap.setAtIndex(ValueLayout.JAVA_DOUBLE, i, i);
            nat.setAtIndex(ValueLayout.JAVA_DOUBLE, i, i);
        }
        for (String name : new String[]{"heap", "native"}) {
            MemorySegment s = name.equals("heap") ? heap : nat;
            Bench.Result aligned = Bench.measure(50, it -> Bench.sinkDouble = sumAligned(s, n));
            Bench.Result unaligned = Bench.measure(50, it -> Bench.sinkDouble = sumUnaligned(s, n));
            System.out.printf("%-7s aligned   %.3f ns/elem%n", name, aligned.nsPerOp() / n);
            System.out.printf("%-7s unaligned %.3f ns/elem%n", name, unaligned.nsPerOp() / n);
        }
        System.out.println();
    }

    // The layout must be a compile-time constant for the JIT to intrinsify the access;
    // passing it as a parameter measures VarHandle dispatch instead of memory access.
    static double sumAligned(MemorySegment s, long n) {
        double acc = 0;
        for (long i = 0; i < n; i++) {
            acc += s.get(ValueLayout.JAVA_DOUBLE, i * 8);
        }
        return acc;
    }

    static double sumUnaligned(MemorySegment s, long n) {
        double acc = 0;
        for (long i = 0; i < n; i++) {
            acc += s.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, i * 8);
        }
        return acc;
    }
}
