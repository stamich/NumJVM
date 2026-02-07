plugins { `java-library` }
description = "Portable real matrix multiplication with optional LP64 OpenBLAS"
dependencies { api(project(":numjvm-core")) }
tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("numjvm.linalg", providers.gradleProperty("linalgBackend").orElse("java").get())
    providers.gradleProperty("openblasPath").orNull?.let { systemProperty("numjvm.openblas.path", it) }
}
tasks.register<Test>("nativeTest") {
    description = "Runs the full linalg contract with OpenBLAS required (never silently skips native tests)."
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    systemProperty("numjvm.linalg", "openblas")
    shouldRunAfter(tasks.test)
}

// Native lines count toward coverage; they are not excluded to make Java-only coverage pass.
tasks.named<JacocoReport>("jacocoTestReport") {
    executionData(fileTree(layout.buildDirectory.dir("jacoco")) { include("*.exec") })
    mustRunAfter(tasks.named("nativeTest"))
}
tasks.named<JacocoCoverageVerification>("jacocoTestCoverageVerification") {
    dependsOn(tasks.test, tasks.named("nativeTest"))
    executionData(fileTree(layout.buildDirectory.dir("jacoco")) { include("*.exec") })
}
tasks.register("backendCheck") {
    group = "verification"
    description = "Validates Java and required OpenBLAS execution with combined 80% line coverage."
    dependsOn(tasks.named("jacocoTestCoverageVerification"), tasks.named("jacocoTestReport"))
}
