package org.numjvm.linalg;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

/** One process-lifetime Linux x86-64 LP64 CBLAS provider. */
final class OpenBlas {
    static final String MODE=mode();
    private static String unavailable="not configured";
    static final OpenBlas INSTANCE=load();
    private final Arena library;
    final SymbolLookup symbols;
    final String configuration;
    private OpenBlas(Path path) {
        if(!OpenBlas.class.getModule().isNativeAccessEnabled()) {
            throw new IllegalArgumentException("enable native access for org.numjvm.linalg (ALL-UNNAMED on classpath)");
        }
        Arena arena=Arena.ofShared();
        try {
            symbols=SymbolLookup.libraryLookup(path,arena);
            var linker=Linker.nativeLinker();
            MethodHandle config=linker.downcallHandle(required("openblas_get_config"),FunctionDescriptor.of(ValueLayout.ADDRESS));
            MemorySegment address;
            try { address=(MemorySegment)config.invokeExact(); }
            catch(Throwable e) { throw new IllegalStateException("cannot read OpenBLAS configuration",e); }
            if(address.equals(MemorySegment.NULL)) throw new IllegalArgumentException("OpenBLAS returned null configuration");
            configuration=address.reinterpret(1024).getString(0);
            if(!configuration.startsWith("OpenBLAS ") || configuration.contains("USE64BITINT")) {
                throw new IllegalArgumentException("only OpenBLAS LP64 CBLAS is supported: "+configuration);
            }
            required("cblas_sgemm");required("cblas_dgemm");
            library=arena; // Retain symbol lifetime for every synchronous call in this JVM.
        } catch(RuntimeException|Error e) { arena.close();throw e; }
    }
    MemorySegment required(String name) {
        return symbols.find(name).orElseThrow(()->new IllegalArgumentException("missing OpenBLAS symbol "+name));
    }
    private static String mode() {
        String value=System.getProperty("numjvm.linalg","auto");
        if(!java.util.Set.of("auto","java","openblas").contains(value)) {
            throw new IllegalArgumentException("numjvm.linalg must be auto, java or openblas: "+value);
        }
        return value;
    }
    private static OpenBlas load() {
        if(MODE.equals("java")) { unavailable="forced Java";return null; }
        if(!System.getProperty("os.name").equals("Linux") || !System.getProperty("os.arch").equals("amd64")) {
            return missing("native baseline is Linux x86-64");
        }
        String explicit=System.getProperty("numjvm.openblas.path");
        try {
            if(explicit!=null) return new OpenBlas(Path.of(explicit).toAbsolutePath());
            for(String candidate:new String[]{"/usr/lib/x86_64-linux-gnu/libopenblas.so.0","/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0"}) {
                if(Files.isRegularFile(Path.of(candidate))) return new OpenBlas(Path.of(candidate));
            }
            return missing("no OpenBLAS installation found; set numjvm.openblas.path");
        } catch(IllegalArgumentException|IllegalStateException|UnsatisfiedLinkError|IllegalCallerException e) {
            return missing(e.getMessage());
        }
    }
    private static OpenBlas missing(String reason) {
        unavailable=reason;
        if(MODE.equals("openblas")) throw new IllegalStateException("OpenBLAS required but unavailable: "+reason);
        return null;
    }
    static String availability() { return INSTANCE==null?unavailable:INSTANCE.configuration; }
}
