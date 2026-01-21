plugins { `java-library` }
description = "Optional incubating Vector API CPU kernels"
dependencies { implementation(project(":numjvm-core")) }
tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
    // javac reports use of incubator modules even for a correctly configured module.
    options.compilerArgs.remove("-Werror")
}
tasks.withType<Test>().configureEach { jvmArgs("--add-modules", "jdk.incubator.vector") }
val codegen: SourceSet by sourceSets.creating
val generateVectorKernels by tasks.registering(JavaExec::class) {
    classpath = codegen.runtimeClasspath
    mainClass = "org.numjvm.vector.codegen.KernelGenerator"
    val templates = layout.projectDirectory.dir("src/codegen/templates")
    val output = layout.buildDirectory.dir("generated/sources/vector/java/main")
    inputs.dir(templates)
    outputs.dir(output)
    argumentProviders.add(CommandLineArgumentProvider { listOf(templates.asFile.absolutePath, output.get().asFile.absolutePath) })
}
sourceSets.main { java.srcDir(generateVectorKernels) }
