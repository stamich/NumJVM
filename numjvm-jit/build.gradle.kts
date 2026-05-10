plugins { `java-library` }

description = "Experimental expression fusion for NumJVM"

dependencies {
    api(project(":numjvm-core"))
    testRuntimeOnly(project(":numjvm-vector"))
}

tasks.register<Test>("fusedProviderTest") {
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    jvmArgs("--add-modules", "jdk.incubator.vector")
    systemProperty("numjvm.expectFusedProvider", "jdk-vector")
    include("**/FusedProviderTest.class")
    include("**/FusedDoubleTest.class")
    include("**/FusedFloatTest.class")
    include("**/FusedLayoutTest.class")
    include("**/FusedMixedTest.class")
    include("**/FusedDispatchTest.class")
    include("**/FusedPropertyStressTest.class")
}
