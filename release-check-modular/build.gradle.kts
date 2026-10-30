plugins {
    application
}

val numjvmVersion = providers.gradleProperty("numjvmVersion").orElse("0.3.0-alpha")

dependencies {
    implementation(numjvmVersion.map { "io.github.stamich:numjvm-io:$it" })
    implementation(numjvmVersion.map { "io.github.stamich:numjvm-linalg:$it" })
    if (providers.gradleProperty("withVector").isPresent || providers.gradleProperty("unresolvedVector").isPresent) {
        runtimeOnly(numjvmVersion.map { "io.github.stamich:numjvm-vector:$it" })
    }
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

application {
    mainClass = "smoke.ReleaseCheck"
}

if (providers.gradleProperty("withVector").isPresent) {
    tasks.named<JavaExec>("run") {
        jvmArgs("--add-modules", "jdk.incubator.vector", "-Dnumjvm.simd=required", "-DexpectVector=true")
    }
}
sourceSets.main { java.srcDir("../release-check/src/main/java") }
application { mainModule = "numjvm.releasecheck" }
if (providers.gradleProperty("withVector").isPresent) {
    tasks.named<JavaExec>("run") { jvmArgs("--add-modules", "org.numjvm.vector") }
}
