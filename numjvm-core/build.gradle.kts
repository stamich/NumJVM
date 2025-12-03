plugins {
    `java-library`
}

description = "NumJVM core: dtypes, memory management, NDArray, views and broadcasting"

tasks.processResources {
    val projectVersion = project.version.toString()
    inputs.property("version", projectVersion)
    filesMatching("org/numjvm/version.properties") {
        expand("version" to projectVersion)
    }
}

// ---------------------------------------------------------------------- generated loops
//
// The scalar inner loops (org.numjvm.internal.loops.*Loops) are generated from the
// templates in src/codegen/templates by the generator in src/codegen/java (decision D3).
// The output is a source directory of the main source set, so compileJava, javadoc and
// sourcesJar pick it up and depend on the generator automatically.

val codegen: SourceSet by sourceSets.creating

val generateKernels by tasks.registering(JavaExec::class) {
    group = "build"
    description = "Generates the scalar inner loops from src/codegen/templates."
    val templates = layout.projectDirectory.dir("src/codegen/templates")
    val output = layout.buildDirectory.dir("generated/sources/kernels/java/main")
    classpath = codegen.runtimeClasspath
    mainClass = "org.numjvm.codegen.KernelGenerator"
    inputs.dir(templates).withPathSensitivity(PathSensitivity.RELATIVE)
    outputs.dir(output)
    argumentProviders.add(CommandLineArgumentProvider {
        listOf(templates.asFile.absolutePath, output.get().asFile.absolutePath)
    })
}

sourceSets.main {
    java.srcDir(generateKernels)
}
