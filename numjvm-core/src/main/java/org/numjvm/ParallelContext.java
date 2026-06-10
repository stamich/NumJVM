package org.numjvm;

import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.numjvm.internal.ExpressionSemantics;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;

/**
 * An opt-in, bounded CPU execution context. Close it after its operations finish.
 * Existing NumJVM operations continue to execute on the calling thread.
 */
@Experimental
public final class ParallelContext implements AutoCloseable {
    private static final AtomicInteger CONTEXT_IDS = new AtomicInteger();
    // A distinct thread identifies confined FFM segments without starting a worker.
    private static final Thread ACCESS_PROBE = new Thread(() -> { }, "numjvm-access-probe");
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

    boolean workerAccessible(NDArray... arrays) {
        ExpressionSemantics.accessible(arrays);
        for (var array : arrays) {
            var buffer = SharedSecrets.arrayAccess().buffer(array);
            buffer.checkAlive();
            if (!buffer.segment().isAccessibleBy(ACCESS_PROBE)) return false;
        }
        return true;
    }

    void runOrElse(NDArray[] arrays, ChunkPlan plan, ChunkAction action, Runnable fallback) {
        if (CURRENT.get() == this) {
            if (closed) throw new IllegalStateException("parallel context is closed");
            if (workerAccessible(arrays)) runChunks(plan, action);
            else fallback.run();
            return;
        }
        synchronized (operationLock) {
            if (closed) throw new IllegalStateException("parallel context is closed");
            if (workerAccessible(arrays)) runChunks(plan, action);
            else fallback.run();
        }
    }

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
            Throwable submissionFailure = null;
            synchronized (this) {
                if (closed) throw new IllegalStateException("parallel context is closed");
                int tasks = (int) Math.min(plan.count, workers);
                for (int worker = 0; worker < tasks; worker++) try {
                    futures.add(executor.submit(() -> {
                    CURRENT.set(this);
                    try {
                        for (;;) {
                            long index = next.getAndIncrement();
                            if (index < 0 || index >= plan.count) break;
                            action.run(index, plan.start(index), plan.end(index));
                        }
                    } finally { CURRENT.remove(); }
                    }));
                } catch (RuntimeException | Error ex) {
                    submissionFailure = ex;
                    break;
                }
            }
            boolean interrupted = false;
            Throwable failure = submissionFailure;
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
        if (CURRENT.get() == this) throw new IllegalStateException("cannot close a context from its worker");
        synchronized (operationLock) {
            synchronized (this) {
                if (!closed) {
                    closed = true;
                    executor.shutdown();
                }
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
