package spike;

import java.lang.management.ManagementFactory;

/**
 * Minimal measurement harness for the spike.
 *
 * <p>This is deliberately simple: warm-up, then the median of several timed rounds.
 * It exists so that the spike can produce numbers before the JMH harness (S5) is in
 * place. Numbers from this harness are indicative only; JMH numbers in
 * {@code results.md} take precedence where both exist.
 */
final class Bench {

    /**
     * Sink that prevents the JIT from eliminating measured work.
     */
    static volatile double sinkDouble;
    static volatile Object sinkObject;

    private static final com.sun.management.ThreadMXBean THREADS =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();

    private Bench() {
    }

    /**
     * Operation measured by the harness.
     */
    @FunctionalInterface
    interface Op {
        void run(int iteration);
    }

    /**
     * Result of one measurement: nanoseconds and heap bytes per operation.
     */
    record Result(double nsPerOp, double bytesPerOp) {
        @Override
        public String toString() {
            return String.format("%,12.1f ns/op %,12.1f B/op", nsPerOp, bytesPerOp);
        }
    }

    /**
     * Runs {@code op} {@code ops} times per round, five warm-up rounds and seven
     * measured rounds, and reports the median.
     */
    static Result measure(long ops, Op op) {
        for (int w = 0; w < 5; w++) {
            for (int i = 0; i < ops; i++) {
                op.run(i);
            }
        }
        int rounds = 7;
        double[] ns = new double[rounds];
        double[] bytes = new double[rounds];
        long tid = Thread.currentThread().threadId();
        for (int r = 0; r < rounds; r++) {
            long b0 = THREADS.getThreadAllocatedBytes(tid);
            long t0 = System.nanoTime();
            for (int i = 0; i < ops; i++) {
                op.run(i);
            }
            long t1 = System.nanoTime();
            long b1 = THREADS.getThreadAllocatedBytes(tid);
            ns[r] = (double) (t1 - t0) / ops;
            bytes[r] = (double) (b1 - b0) / ops;
        }
        java.util.Arrays.sort(ns);
        java.util.Arrays.sort(bytes);
        return new Result(ns[rounds / 2], bytes[rounds / 2]);
    }

    /**
     * Picks an operation count so that one round takes roughly the same time.
     */
    static long opsFor(long elements) {
        long ops = 50_000_000L / Math.max(1, elements);
        return Math.max(3, Math.min(200_000, ops));
    }
}
