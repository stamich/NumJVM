package org.numjvm.backend;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import static org.junit.jupiter.api.Assertions.*;
class BackendConfigurationTest {
    @Test void missingRequiredAndInvalidModesFailInFreshJvms() throws Exception {
        String cp=java.nio.file.Path.of(NJ.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                + java.io.File.pathSeparator + java.nio.file.Path.of(BackendProbe.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        for(String mode:new String[]{"required","invalid"}) {
            Process p=new ProcessBuilder(java.nio.file.Path.of(System.getProperty("java.home"),"bin","java").toString(),
                    "-cp",cp,"-Dnumjvm.simd="+mode,BackendProbe.class.getName()).redirectErrorStream(true).start();
            String output=new String(p.getInputStream().readAllBytes(),java.nio.charset.StandardCharsets.UTF_8);
            assertNotEquals(0,p.waitFor());assertTrue(output.contains(mode.equals("required")?"SIMD provider required":"numjvm.simd must be"),output);
        }
    }
}
