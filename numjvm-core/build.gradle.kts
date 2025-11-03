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
