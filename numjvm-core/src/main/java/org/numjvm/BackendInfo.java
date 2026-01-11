package org.numjvm;

import org.numjvm.internal.backend.Accelerators;

/** Experimental backend diagnostics. Configuration is read once, on first backend use. */
public final class BackendInfo {
    private BackendInfo() { }
    /** @return usable SIMD provider name, or scalar */
    public static String simdProvider() { return Accelerators.available(); }
    /** @return discovery outcome; does not imply a specific operation was accelerated */
    public static String simdReason() { return Accelerators.reason(); }
    /** @return completed/entered vector runs; compare before and after an operation in isolated tests */
    public static long vectorRuns() { return Accelerators.executions(); }
}
