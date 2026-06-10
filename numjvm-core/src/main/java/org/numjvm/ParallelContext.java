package org.numjvm;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * An opt-in, bounded CPU execution context. Close it after its operations finish.
 * Existing NumJVM operations continue to execute on the calling thread.
 */
@Experimental
public final class ParallelContext implements AutoCloseable {
    private static final AtomicInteger CONTEXT_IDS = new AtomicInteger();
    private final int workers;
    private final ThreadPoolExecutor executor;
    private volatile boolean closed;

    ParallelContext(int workers) {
        if (workers < 1 || workers > 256) throw new IllegalArgumentException("workers must be between 1 and 256");
        this.workers = workers;
        int id = CONTEXT_IDS.incrementAndGet();
        var sequence = new AtomicInteger();
        executor = new ThreadPoolExecutor(workers, workers, 0, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(workers), task -> {
                    var thread = new Thread(task, "numjvm-" + id + "-worker-" + sequence.incrementAndGet());
                    thread.setDaemon(true);
                    return thread;
                }, new ThreadPoolExecutor.AbortPolicy());
    }

    /** Returns the configured maximum number of workers.
     * @return worker limit
     */
    public int workers() { return workers; }

    /** Whether this context has been closed.
     * @return true after closure starts
     */
    public boolean isClosed() { return closed; }

    /** Stops accepting work and waits for all submitted workers to terminate. */
    @Override public void close() {
        synchronized (this) {
            if (!closed) {
                closed = true;
                executor.shutdown();
            }
        }
        boolean interrupted = false;
        for (;;) {
            try {
                if (executor.awaitTermination(1, TimeUnit.DAYS)) break;
            } catch (InterruptedException ex) { interrupted = true; }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }
}
