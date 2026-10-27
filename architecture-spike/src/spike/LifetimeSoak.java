package spike;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.ref.Cleaner;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

/**
 * S4 — 100,000 short-lived off-heap arrays under {@code Arena.ofAuto()}.
 *
 * <p>Automatic arenas free native memory only when the GC finds the arena unreachable,
 * and the GC is driven by Java heap pressure. A workload that allocates many native
 * arrays but little Java heap may therefore never trigger a GC, and native memory
 * grows without bound. This spike measures that and tests one mitigation: a live
 * native-bytes counter with a soft limit that triggers {@code System.gc()}.
 *
 * <p>Run each mode in its own JVM so RSS numbers are independent:
 * <pre>
 * java -Xmx512m -cp out spike.LifetimeSoak heap
 * java -Xmx512m -cp out spike.LifetimeSoak auto
 * java -Xmx512m -cp out spike.LifetimeSoak auto-limit
 * java -Xmx512m -cp out spike.LifetimeSoak confined
 * java -Xmx512m -cp out spike.LifetimeSoak auto 1048576   # optional array size in bytes
 * </pre>
 */
public final class LifetimeSoak {

    static final int ARRAYS = 100_000;
    static long BYTES = 32 * 1024;                 // default: 4,096 float64, at the 32 KiB threshold
    static final long SOFT_LIMIT = 256L << 20;     // 256 MiB of live native memory
    static final long RSS_ABORT = 3_500L << 20;    // protect the machine

    static final AtomicLong LIVE_NATIVE = new AtomicLong();
    static final Cleaner CLEANER = Cleaner.create();

    /**
     * Stands in for the future Buffer: owns the segment, carries the accounting hook.
     */
    record Holder(MemorySegment segment) {
    }

    private LifetimeSoak() {
    }

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "auto";
        if (args.length > 1) {
            BYTES = Long.parseLong(args[1]);
        }
        Holder[] window = new Holder[8];          // a few arrays alive at any time
        long peakRss = rssBytes();
        long gcBefore = gcCount();
        long forcedGcs = 0;
        long t0 = System.nanoTime();
        double checksum = 0;

        for (int i = 0; i < ARRAYS; i++) {
            Holder h;
            switch (mode) {
                case "heap" -> h = new Holder(MemorySegment.ofArray(new long[(int) (BYTES / 8)]));
                case "auto", "auto-limit" -> {
                    if (mode.equals("auto-limit") && LIVE_NATIVE.get() + BYTES > SOFT_LIMIT) {
                        forcedGcs++;
                        System.gc();
                        long deadline = System.nanoTime() + 100_000_000L;
                        while (LIVE_NATIVE.get() + BYTES > SOFT_LIMIT && System.nanoTime() < deadline) {
                            Thread.sleep(1);
                        }
                    }
                    MemorySegment s = Arena.ofAuto().allocate(BYTES, 64);
                    h = new Holder(s);
                    LIVE_NATIVE.addAndGet(BYTES);
                    long size = BYTES;
                    CLEANER.register(h, () -> LIVE_NATIVE.addAndGet(-size));
                }
                case "confined" -> {
                    try (Arena arena = Arena.ofConfined()) {
                        MemorySegment s = arena.allocate(BYTES, 64);
                        s.set(ValueLayout.JAVA_DOUBLE, 0, i);
                        checksum += s.get(ValueLayout.JAVA_DOUBLE, 0);
                    }
                    h = null;
                }
                default -> throw new IllegalArgumentException(mode);
            }
            if (h != null) {
                h.segment().set(ValueLayout.JAVA_DOUBLE, 0, i);
                checksum += h.segment().get(ValueLayout.JAVA_DOUBLE, 0);
                window[i & 7] = h;
            }
            if ((i + 1) % 10_000 == 0) {
                long rss = rssBytes();
                peakRss = Math.max(peakRss, rss);
                System.out.printf("%-10s %,7d arrays  RSS %,6d MiB  tracked native %,6d MiB  GCs %d%n",
                        mode, i + 1, rss >> 20, LIVE_NATIVE.get() >> 20, gcCount() - gcBefore);
                if (rss > RSS_ABORT) {
                    System.out.println("ABORT: RSS above " + (RSS_ABORT >> 20) + " MiB");
                    break;
                }
            }
        }
        long t1 = System.nanoTime();
        System.out.printf("RESULT mode=%s time=%.0f ms peakRss=%d MiB gcs=%d forcedGcs=%d checksum=%.0f%n",
                mode, (t1 - t0) / 1e6, peakRss >> 20, gcCount() - gcBefore, forcedGcs, checksum);
    }

    static long rssBytes() throws IOException {
        for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
            if (line.startsWith("VmRSS:")) {
                return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024;
            }
        }
        return -1;
    }

    static long gcCount() {
        long n = 0;
        for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
            n += Math.max(0, gc.getCollectionCount());
        }
        return n;
    }
}
