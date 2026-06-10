package org.numjvm;

import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
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
    private static final ThreadLocal<ParallelContext> CURRENT = new ThreadLocal<>();
    private final Object operationLock = new Object();
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

    @FunctionalInterface
    interface ChunkAction { void run(long index, long start, long end); }

    /** Submits at most one task per worker; tasks claim fixed chunks dynamically. */
    void runChunks(ChunkPlan plan, ChunkAction action) {
        if (CURRENT.get() == this) {
            if (closed) throw new IllegalStateException("parallel context is closed");
            for (long i = 0; i < plan.count; i++) action.run(i, plan.start(i), plan.end(i));
            return;
        }
        synchronized (operationLock) {
            var futures = new ArrayList<Future<?>>();
            var next = new AtomicLong();
            synchronized (this) {
                if (closed) throw new IllegalStateException("parallel context is closed");
                int tasks = (int) Math.min(plan.count, workers);
                for (int worker = 0; worker < tasks; worker++) futures.add(executor.submit(() -> {
                    CURRENT.set(this);
                    try {
                        for (;;) {
                            long index = next.getAndIncrement();
                            if (index < 0 || index >= plan.count) break;
                            action.run(index, plan.start(index), plan.end(index));
                        }
                    } finally { CURRENT.remove(); }
                }));
            }
            boolean interrupted = false;
            Throwable failure = null;
            for (var future : futures) {
                for (;;) {
                    try { future.get(); break; }
                    catch (InterruptedException ex) { interrupted = true; }
                    catch (ExecutionException ex) { if (failure == null) failure = ex.getCause(); break; }
                }
            }
            if (interrupted) Thread.currentThread().interrupt();
            if (failure instanceof Error error) throw error;
            if (failure instanceof RuntimeException error) throw error;
            if (failure != null) throw new IllegalStateException("parallel task failed", failure);
            if (interrupted) throw new IllegalStateException("parallel operation interrupted");
        }
    }

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
