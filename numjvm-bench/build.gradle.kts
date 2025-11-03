// JMH benchmarks. Not published.
//
// Run all:       ./gradlew :numjvm-bench:jmh
// Run a subset:  ./gradlew :numjvm-bench:jmh -PjmhArgs="ViewBenchmark -p size=1024 -prof gc"

plugins {
    java
}

description = "NumJVM JMH benchmarks"

val jmh: SourceSet by sourceSets.creating

dependencies {
    "jmhImplementation"(project(":numjvm-core"))
    "jmhImplementation"(project(":numjvm-io"))
    "jmhImplementation"(libs.jmh.core)
    "jmhAnnotationProcessor"(libs.jmh.generator.annprocess)
}

tasks.register<JavaExec>("jmh") {
    group = "benchmark"
    description = "Runs the JMH benchmarks; pass JMH options with -PjmhArgs=\"...\"."
    classpath = jmh.runtimeClasspath
    mainClass = "org.openjdk.jmh.Main"
    val extra = providers.gradleProperty("jmhArgs").orElse("")
    argumentProviders.add(CommandLineArgumentProvider {
        extra.get().split(" ").filter { it.isNotBlank() }
    })
}

tasks.named("assemble") {
    dependsOn(tasks.named("jmhClasses"))
}
