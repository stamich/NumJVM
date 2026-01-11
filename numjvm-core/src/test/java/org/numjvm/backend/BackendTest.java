package org.numjvm.backend;

import org.junit.jupiter.api.Test;
import org.numjvm.BackendInfo;
import org.numjvm.NJ;
import static org.junit.jupiter.api.Assertions.*;

class BackendTest {
    @Test void coreOnlyExecutesWithoutOptionalModules() {
        assertEquals("scalar", BackendInfo.simdProvider());
        assertEquals(3.0, NJ.add(NJ.array(new double[]{1}), NJ.array(new double[]{2})).getDouble(0));
        assertEquals(0, BackendInfo.vectorRuns());
    }
}
