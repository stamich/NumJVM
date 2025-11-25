plugins {
    application
}

description = "NumJVM example: scoped memory, keep() and memory statistics"

dependencies {
    implementation(project(":numjvm-io"))
}

application {
    mainClass = "org.numjvm.examples.MemoryScopes"
}
