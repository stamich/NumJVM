plugins { application }
dependencies {
    implementation(project(":numjvm-core"))
    if (!providers.gradleProperty("withoutVector").isPresent) { runtimeOnly(project(":numjvm-vector")) }
}
application { mainClass = "examples.AcceleratedArrays" }
tasks.named<JavaExec>("run") {
    if (!providers.gradleProperty("withoutVector").isPresent) { jvmArgs("--add-modules", "jdk.incubator.vector") }
}
