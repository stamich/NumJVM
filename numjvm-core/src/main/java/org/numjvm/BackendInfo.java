package org.numjvm;

import org.numjvm.internal.backend.Accelerators;

/** Experimental backend diagnostics. Configuration is read once, on first backend use. */
@org.numjvm.Experimental
public final class BackendInfo {
    private BackendInfo() { }
    /** Returns usable SIMD provider name, or scalar.
     * @return usable SIMD provider name, or scalar
     */
    public static String simdProvider() { return Accelerators.available(); }
    /** Returns discovery outcome; does not imply a specific operation was accelerated.
     * @return discovery outcome; does not imply a specific operation was accelerated
     */
    public static String simdReason() { return Accelerators.reason(); }
    /** Returns completed/entered vector runs; compare before and after an operation in isolated tests.
     * @return completed/entered vector runs; compare before and after an operation in isolated tests
     */
    public static long vectorRuns() { return Accelerators.executions(); }
}
