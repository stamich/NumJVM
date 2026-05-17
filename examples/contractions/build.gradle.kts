plugins { application }
dependencies {
    implementation(project(":numjvm-linalg"))
    implementation(project(":numjvm-jit"))
    if (providers.gradleProperty("withVector").isPresent) runtimeOnly(project(":numjvm-vector"))
}
application { mainClass = "examples.Contractions" }
tasks.named<JavaExec>("run") {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("numjvm.linalg", providers.gradleProperty("linalgBackend").orElse("java").get())
    if (providers.gradleProperty("withVector").isPresent) {
        jvmArgs("--add-modules", "jdk.incubator.vector")
        systemProperty("numjvm.simd", "required")
    }
}
