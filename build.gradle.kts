// Shared configuration for all NumJVM modules.
//
// Library modules (numjvm-core, numjvm-io) are JPMS modules compiled for Java 25.
// Tests have no module-info and run on the class path, which gives them access to
// internal packages (white-box tests).

plugins {
    base
}

val javaVersion = 25

subprojects {
    apply(plugin = "java")

    extensions.configure<JavaPluginExtension> {
        toolchain {
            languageVersion = JavaLanguageVersion.of(javaVersion)
        }
    }

    tasks.withType<JavaCompile>().configureEach {
        options.encoding = "UTF-8"
        options.release = javaVersion
        options.compilerArgs.add("-parameters")
    }

    // Library code is held to a warning-free standard; tests and benchmarks are not.
    tasks.named<JavaCompile>("compileJava") {
        options.compilerArgs.addAll(listOf("-Xlint:all,-requires-transitive-automatic", "-Werror"))
    }

    dependencies {
        "testImplementation"(platform(rootProject.libs.junit.bom))
        "testImplementation"(rootProject.libs.junit.jupiter)
        "testImplementation"(rootProject.libs.jqwik)
        "testRuntimeOnly"(rootProject.libs.junit.platform.launcher)
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform {
            includeEngines("junit-jupiter", "jqwik")
        }
        maxHeapSize = "1g"
        testLogging {
            events("failed", "skipped")
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        }
    }
}
