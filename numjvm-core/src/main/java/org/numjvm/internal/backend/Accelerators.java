package org.numjvm.internal.backend;

import java.util.ServiceLoader;
import java.util.ServiceConfigurationError;
import java.util.concurrent.atomic.LongAdder;
import org.numjvm.dtype.DType;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.UnaryLoop;

/** Discovers optional providers without a static dependency on incubator modules. */
public final class Accelerators {
    private static final String MODE = System.getProperty("numjvm.simd", "auto");
    private static String discovery = "not installed";
    private static final LoopProvider PROVIDER = load();
    private static final LongAdder EXECUTIONS = new LongAdder();
    private Accelerators() { }

    private static LoopProvider load() {
        if (!java.util.Set.of("auto", "off", "required").contains(MODE)) {
            throw new IllegalArgumentException("numjvm.simd must be auto, off or required: " + MODE);
        }
        if (MODE.equals("off")) { discovery = "disabled"; return null; }
        LoopProvider provider = null;
        try {
            provider = ServiceLoader.load(LoopProvider.class).findFirst().orElse(null);
        } catch (ServiceConfigurationError | LinkageError e) {
            discovery = e.getClass().getSimpleName() + ": " + e.getMessage();
        }
        if (provider == null && MODE.equals("required")) {
            throw new IllegalStateException("SIMD provider required but unavailable: " + discovery
                    + "; install numjvm-vector and resolve jdk.incubator.vector");
        }
        if (provider != null) { discovery = "available"; }
        return provider;
    }

    /** Records an actually executed vector run, not merely provider discovery. */
    public static void executed() { EXECUTIONS.increment(); }

    /** @return actual accelerated run count */
    public static long executions() { return EXECUTIONS.sum(); }

    /** @return provider discovery outcome or fallback reason */
    public static String reason() { return discovery; }

    /** @return provider name, or scalar when no extension is installed */
    public static String available() { return PROVIDER == null ? "scalar" : PROVIDER.name(); }

    /** @return the selected unary implementation */
    public static UnaryLoop unary(String op, DType type, UnaryLoop scalar) {
        return PROVIDER == null ? scalar : PROVIDER.unary(op, type, scalar);
    }

    /** @return the selected binary implementation */
    public static BinaryLoop binary(String op, DType type, BinaryLoop scalar) {
        return PROVIDER == null ? scalar : PROVIDER.binary(op, type, scalar);
    }
}
