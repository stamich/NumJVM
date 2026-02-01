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
