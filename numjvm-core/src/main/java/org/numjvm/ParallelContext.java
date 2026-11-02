package org.numjvm;

import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.numjvm.internal.ExpressionSemantics;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Order;
import java.lang.foreign.ValueLayout;
import org.numjvm.dtype.DType;
import org.numjvm.ufunc.BinaryUfunc;
import org.numjvm.ufunc.UnaryUfunc;
import org.numjvm.ufunc.Ufuncs;
import org.numjvm.ufunc.Reductions;

/**
 * An opt-in, bounded CPU execution context. Close it after its operations finish.
 * Existing NumJVM operations continue to execute on the calling thread.
 */
@Experimental
public final class ParallelContext implements AutoCloseable {
    private static final long CHUNK_SIZE = 32768;
    static {
        SharedSecrets.setParallelAccess((context, arrays, size, eligible, task, fallback) -> {
            if (!eligible || context.workers == 1 || size < CHUNK_SIZE) {
                context.sequential(() -> { fallback.run(); return null; }, "small or unsupported expression");
            } else {
                context.runOrElse(arrays, new ChunkPlan(size, CHUNK_SIZE),
                        (index, start, end) -> task.run(start, end), fallback);
            }
        });
    }


    private static final AtomicInteger CONTEXT_IDS = new AtomicInteger();
    // A distinct thread identifies confined FFM segments without starting a worker.
    private static final Thread ACCESS_PROBE = new Thread(() -> { }, "numjvm-access-probe");
    private static final ThreadLocal<ParallelContext> CURRENT = new ThreadLocal<>();
    private final Object operationLock = new Object();
    private final LongAdder parallelRuns = new LongAdder();
    private final LongAdder fallbackRuns = new LongAdder();
    private final ThreadLocal<Decision> decision = ThreadLocal.withInitial(() -> new Decision("none", "no operation"));
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

    /** The strategy most recently selected on the calling thread.
     * @param strategy execution strategy
     * @param reason selection reason
     */
    public record Decision(String strategy, String reason) { }

    /** Returns the calling thread's last selection.
     * @return last decision, initially {@code none}
     */
    public Decision lastDecision() { return decision.get(); }

    /** Returns the number of completed parallel dispatches.
     * @return completed parallel dispatches
     */
    public long parallelRuns() { return parallelRuns.sum(); }

    /** Returns the number of completed single-worker fallbacks.
     * @return completed fallbacks
     */
    public long fallbackRuns() { return fallbackRuns.sum(); }

    private void dispatch(NDArray[] arrays, ChunkPlan plan, ChunkAction action, Runnable fallback) {
        if (workerAccessible(arrays)) {
            boolean nested = CURRENT.get() == this;
            runChunks(plan, action);
            if (nested) {
                fallbackRuns.increment();
                decision.set(new Decision("sequential", "nested worker"));
            } else {
                parallelRuns.increment();
                decision.set(new Decision("parallel", "worker-accessible memory"));
            }
        } else {
            fallback.run();
            fallbackRuns.increment();
            decision.set(new Decision("sequential", "confined memory"));
        }
    }

    private static boolean supported(DType dtype) {
        return dtype == DType.FLOAT32 || dtype == DType.FLOAT64;
    }

    private boolean compatible(NDArray input, NDArray output) {
        return workers > 1 && output.size() >= CHUNK_SIZE && input.isCContiguous() && output.isCContiguous()
                && input.shape().equals(output.shape()) && input.dtype() == output.dtype()
                && supported(input.dtype()) && output.isWriteable() && safeOverlap(input, output);
    }

    private static boolean safeOverlap(NDArray input, NDArray output) {
        if (!input.sharesMemoryWith(output)) return true;
        var access = SharedSecrets.arrayAccess();
        return access.buffer(input).segment() == access.buffer(output).segment()
                && access.offset(input) == access.offset(output)
                && java.util.Arrays.equals(input.strides().values(), output.strides().values());
    }

    /** Applies a supported unary ufunc into an output, using workers when eligible.
     * @param op unary ufunc
     * @param input input array
     * @param output destination
     * @return output
     */
    public NDArray apply(UnaryUfunc op, NDArray input, NDArray output) {
        java.util.Objects.requireNonNull(op, "op");
        ExpressionSemantics.accessible(input, output);
        boolean eligible = (op == Ufuncs.NEGATIVE || op == Ufuncs.ABSOLUTE || op == Ufuncs.SQUARE || op == Ufuncs.SQRT)
                && compatible(input, output);
        if (!eligible) return sequential(() -> op.apply(input, output), "small or unsupported operation");
        var source = input.reshapeView(input.size());
        var destination = output.reshapeView(output.size());
        runOrElse(new NDArray[]{input, output}, new ChunkPlan(output.size(), CHUNK_SIZE),
                (i, start, end) -> op.apply(source.slice(Slice.range(start, end)), destination.slice(Slice.range(start, end))),
                () -> op.apply(input, output));
        return output;
    }

    /** Applies a supported binary ufunc into an output, using workers when eligible.
     * @param op binary ufunc
     * @param left first input
     * @param right second input
     * @param output destination
     * @return output
     */
    public NDArray apply(BinaryUfunc op, NDArray left, NDArray right, NDArray output) {
        java.util.Objects.requireNonNull(op, "op");
        ExpressionSemantics.accessible(left, right, output);
        boolean eligible = (op == Ufuncs.ADD || op == Ufuncs.SUBTRACT || op == Ufuncs.MULTIPLY || op == Ufuncs.DIVIDE)
                && compatible(left, output) && compatible(right, output);
        if (!eligible) return sequential(() -> op.apply(left, right, output), "small or unsupported operation");
        var a = left.reshapeView(left.size());
        var b = right.reshapeView(right.size());
        var destination = output.reshapeView(output.size());
        runOrElse(new NDArray[]{left, right, output}, new ChunkPlan(output.size(), CHUNK_SIZE),
                (i, start, end) -> op.apply(a.slice(Slice.range(start, end)), b.slice(Slice.range(start, end)),
                        destination.slice(Slice.range(start, end))),
                () -> op.apply(left, right, output));
        return output;
    }

    /**
     * Sums a contiguous real array using fixed logical chunks and a fixed merge order.
     * The parallel result is reproducible across worker counts for the same inputs,
     * dtype and runtime, but can differ from a sequential floating-point sum.
     * Unsupported or confined arrays use the existing sequential reduction.
     * @param input array to reduce
     * @return scalar result with the existing sum dtype and ownership rules
     */
    public NDArray sum(NDArray input) {
        rejectOtherContext();
        if (CURRENT.get() == this) return sumNow(input);
        synchronized (operationLock) { return sumNow(input); }
    }

    private NDArray sumNow(NDArray input) {
        if (closed) throw new IllegalStateException("parallel context is closed");
        ExpressionSemantics.accessible(input);
        var plan = new ChunkPlan(input.size(), CHUNK_SIZE);
        if (!input.isCContiguous() || !supported(input.dtype()) || input.size() < CHUNK_SIZE
                || plan.count > Integer.MAX_VALUE)
            return sequential(() -> Reductions.sum(input), "small or unsupported reduction");
        var result = SharedSecrets.arrayAccess().allocate(ExpressionSemantics.allocator(input),
                input.dtype(), Shape.of(), Order.C);
        var segment = SharedSecrets.arrayAccess().buffer(input).segment();
        long base = SharedSecrets.arrayAccess().offset(input);
        // Each index identifies a fixed input interval regardless of worker count.
        var partials = new double[(int) plan.count];
        var computed = new java.util.concurrent.atomic.AtomicBoolean();
        runOrElse(new NDArray[]{input, result}, plan, (chunk, start, end) -> {
            if (input.dtype() == DType.FLOAT32) {
                float value = 0;
                for (long i = start; i < end; i++) value += segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, base + i * Float.BYTES);
                partials[(int) chunk] = value;
            } else {
                double value = 0;
                for (long i = start; i < end; i++) value += segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, base + i * Double.BYTES);
                partials[(int) chunk] = value;
            }
            computed.set(true);
        }, () -> {
            var sequentialResult = Reductions.sum(input);
            result.setDouble(sequentialResult.getDouble());
        });
        if (computed.get()) {
            if (input.dtype() == DType.FLOAT32) {
                float value = 0;
                for (double part : partials) value += (float) part;
                result.setDouble(value);
            } else {
                double value = 0;
                for (double part : partials) value += part;
                result.setDouble(value);
            }
        }
        return result;
    }

    private NDArray sequential(java.util.function.Supplier<NDArray> action, String reason) {
        rejectOtherContext();
        if (CURRENT.get() == this) return sequentialNow(action, reason);
        synchronized (operationLock) { return sequentialNow(action, reason); }
    }

    private NDArray sequentialNow(java.util.function.Supplier<NDArray> action, String reason) {
        if (closed) throw new IllegalStateException("parallel context is closed");
        NDArray result = action.get();
        fallbackRuns.increment();
        decision.set(new Decision("sequential", reason));
        return result;
    }

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
        rejectOtherContext();
        if (CURRENT.get() == this) {
            if (closed) throw new IllegalStateException("parallel context is closed");
            dispatch(arrays, plan, action, fallback);
            return;
        }
        synchronized (operationLock) {
            if (closed) throw new IllegalStateException("parallel context is closed");
            dispatch(arrays, plan, action, fallback);
        }
    }

    /** Submits at most one task per worker; tasks claim fixed chunks dynamically. */
    void runChunks(ChunkPlan plan, ChunkAction action) {
        rejectOtherContext();
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
                    } finally { decision.remove(); CURRENT.remove(); }
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
                    // Keep the interrupt recorded while allowing every worker to be joined.
                    if (interrupted) Thread.interrupted();
                    try { future.get(); break; }
                    catch (InterruptedException ex) { interrupted = true; Thread.currentThread().interrupt(); }
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

    private void rejectOtherContext() {
        var current = CURRENT.get();
        if (current != null && current != this)
            throw new IllegalStateException("nested execution across parallel contexts is unsupported");
    }

    /** Stops accepting work and waits for all submitted workers to terminate. */
    @Override public void close() {
        if (CURRENT.get() != null) throw new IllegalStateException("cannot close a context from a NumJVM worker");
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
            if (interrupted) Thread.interrupted();
            try {
                if (executor.awaitTermination(1, TimeUnit.DAYS)) break;
            } catch (InterruptedException ex) { interrupted = true; Thread.currentThread().interrupt(); }
        }
        decision.remove();
        if (interrupted) Thread.currentThread().interrupt();
    }
}
