package org.numjvm;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Properties;

import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.NativeMemoryTracker;
import org.numjvm.memory.MemoryStats;
import org.numjvm.memory.NJScope;

/**
 * Entry point to NumJVM.
 *
 * <p>Array factories, scopes and memory statistics are reached through static methods
 * on this class.
 */
public final class NJ {

    private static final String VERSION = loadVersion();

    private NJ() {
    }

    /**
     * Returns the NumJVM version, for example {@code 0.1.0-alpha}.
     *
     * @return the library version, or {@code "unknown"} when the build did not record it
     */
    public static String version() {
        return VERSION;
    }

    /**
     * Opens a memory scope confined to the current thread. Arrays created through the
     * scope are freed when it is closed.
     *
     * @return a new confined scope; close it with try-with-resources
     * @see NJScope
     */
    public static NJScope scope() {
        return NJScope.confined();
    }

    /**
     * Opens a memory scope confined to the current thread; same as {@link #scope()}.
     *
     * @return a new confined scope
     */
    public static NJScope scopeConfined() {
        return NJScope.confined();
    }

    /**
     * Opens a memory scope whose arrays any thread may access.
     *
     * @return a new shared scope
     */
    public static NJScope scopeShared() {
        return NJScope.shared();
    }

    /**
     * Returns a snapshot of NumJVM's native memory usage and memory settings.
     *
     * <p>Settings are read from system properties:
     * {@code -Dnumjvm.memory.heapThreshold=<bytes>} (default 1 MiB) and
     * {@code -Dnumjvm.memory.nativeSoftLimit=<bytes>} (default: none). The hard bound on
     * native memory is the JVM's {@code -XX:MaxDirectMemorySize}.
     *
     * @return current statistics
     */
    public static MemoryStats memoryStats() {
        HybridAllocator allocator = HybridAllocator.defaultInstance();
        NativeMemoryTracker t = NativeMemoryTracker.GLOBAL;
        return new MemoryStats(t.liveBytes(), t.peakBytes(), t.allocationCount(),
                allocator.heapThreshold(), allocator.nativeSoftLimit());
    }

    private static String loadVersion() {
        try (InputStream in = NJ.class.getResourceAsStream("version.properties")) {
            if (in == null) {
                return "unknown";
            }
            Properties p = new Properties();
            p.load(in);
            String v = p.getProperty("version", "unknown");
            return v.startsWith("${") ? "unknown" : v;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
