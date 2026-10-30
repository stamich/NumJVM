package org.numjvm.backend;

import org.junit.jupiter.api.Test;
import org.numjvm.BackendInfo;
import org.numjvm.NJ;
import static org.junit.jupiter.api.Assertions.*;

class BackendTest {
    @Test void coreOnlyExecutesWithoutOptionalModules() {
        boolean installed = Boolean.getBoolean("numjvm.test.simdInstalled") && !System.getProperty("numjvm.simd", "auto").equals("off");
        assertEquals(installed ? "jdk-vector" : "scalar", BackendInfo.simdProvider());
        long before = BackendInfo.vectorRuns();
        assertEquals(3.0, NJ.add(NJ.array(new double[]{1}), NJ.array(new double[]{2})).getDouble(0));
        assertEquals(before, BackendInfo.vectorRuns());
    }
}
