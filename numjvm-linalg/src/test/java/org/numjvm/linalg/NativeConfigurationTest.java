package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import static org.junit.jupiter.api.Assertions.*;
class NativeConfigurationTest {
    @Test void forceModesAndMissingLibraryHaveExplicitBehavior() throws Exception {
        String cp=java.nio.file.Path.of(NJ.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                + java.io.File.pathSeparator + java.nio.file.Path.of(Linalg.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                + java.io.File.pathSeparator + java.nio.file.Path.of(NativeConfigurationProbe.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        for(String mode:new String[]{"java","auto","openblas","invalid"}) {
            Process p=new ProcessBuilder(java.nio.file.Path.of(System.getProperty("java.home"),"bin","java").toString(),
                "--enable-native-access=ALL-UNNAMED","-cp",cp,"-Dnumjvm.linalg="+mode,
                "-Dnumjvm.openblas.path=/nonexistent/numjvm-test-openblas.so",NativeConfigurationProbe.class.getName())
                .redirectErrorStream(true).start();
            String output=new String(p.getInputStream().readAllBytes(),java.nio.charset.StandardCharsets.UTF_8);
            int exit=p.waitFor();
            if(mode.equals("java") || mode.equals("auto")) { assertEquals(0,exit,output);assertTrue(output.contains("java:")); }
            else { assertNotEquals(0,exit);assertTrue(output.contains(mode.equals("openblas")?"OpenBLAS required but unavailable":"numjvm.linalg must be"),output); }
        }
    }
}
