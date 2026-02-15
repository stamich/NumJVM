plugins { application }
dependencies { implementation(project(":numjvm-linalg")) }
application { mainClass = "examples.MatrixMultiplication" }
tasks.named<JavaExec>("run") {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("numjvm.linalg", providers.gradleProperty("linalgBackend").orElse("java").get())
    providers.gradleProperty("openblasPath").orNull?.let { systemProperty("numjvm.openblas.path", it) }
}
