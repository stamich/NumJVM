package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class OpenBlasLoadingTest {
    @Test void absentOptionalProviderIsExplainedWithoutBreakingJava() {
        assertNotNull(OpenBlas.availability());
        if(OpenBlas.MODE.equals("openblas"))assertNotNull(OpenBlas.INSTANCE);
    }
}
