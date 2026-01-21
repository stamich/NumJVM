/** Optional Vector API implementation; no incubator types enter the core API. */
module org.numjvm.vector {
    requires org.numjvm.core;
    requires jdk.incubator.vector;
    provides org.numjvm.internal.backend.LoopProvider with org.numjvm.vector.VectorProvider;
}
