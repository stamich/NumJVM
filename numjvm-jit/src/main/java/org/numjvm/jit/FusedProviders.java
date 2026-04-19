package org.numjvm.jit;

import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import org.numjvm.internal.backend.FusedProvider;

/** Optional service discovery; missing or unresolved vector modules remain usable. */
final class FusedProviders {
    private static final FusedProvider PROVIDER = load();
    private FusedProviders() { }

    private static FusedProvider load() {
        try { return ServiceLoader.load(FusedProvider.class).findFirst().orElse(null); }
        catch (ServiceConfigurationError | LinkageError unavailable) { return null; }
    }
    static FusedProvider provider() { return PROVIDER; }
    static String name() { return PROVIDER == null ? "unavailable" : PROVIDER.name(); }
}
