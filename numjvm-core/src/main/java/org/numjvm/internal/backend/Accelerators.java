package org.numjvm.internal.backend;

import java.util.ServiceLoader;
import org.numjvm.dtype.DType;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.UnaryLoop;

/** Discovers optional providers without a static dependency on incubator modules. */
public final class Accelerators {
    private static final LoopProvider PROVIDER = load();
    private Accelerators() { }

    private static LoopProvider load() {
        return ServiceLoader.load(LoopProvider.class).findFirst().orElse(null);
    }

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
