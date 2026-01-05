plugins {
    application
}

val numjvmVersion = providers.gradleProperty("numjvmVersion").orElse("0.2.0-alpha")

dependencies {
    implementation(numjvmVersion.map { "io.github.stamich:numjvm-io:$it" })
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

application {
    mainClass = "smoke.ReleaseCheck"
}
