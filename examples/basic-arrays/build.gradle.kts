plugins {
    application
}

description = "NumJVM example: creating, viewing, broadcasting and saving arrays"

dependencies {
    implementation(project(":numjvm-io"))
}

application {
    mainClass = "org.numjvm.examples.BasicArrays"
}

tasks.named<JavaExec>("run") {
    workingDir = layout.buildDirectory.get().asFile.also { it.mkdirs() }
}
