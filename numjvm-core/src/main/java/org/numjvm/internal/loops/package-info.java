/**
 * Inner loops and their execution over iteration plans (ADR-004). Internal: not exported.
 *
 * <p>An inner loop processes one run of elements: per operand a memory segment, the byte
 * offset of the run's first element and the byte stride between elements, plus the run
 * length. {@link org.numjvm.internal.loops.LoopExecutor} builds the
 * {@link org.numjvm.iteration.IterationPlan} for the operands and calls the loop once per
 * run; the loop itself selects a contiguous, broadcast-scalar or strided path.
 */
package org.numjvm.internal.loops;
