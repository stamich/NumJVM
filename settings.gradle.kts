plugins {
    // Lets Gradle download a JDK 25 toolchain when none is installed locally.
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

rootProject.name = "numjvm"

dependencyResolutionManagement {
    repositoriesMode = RepositoriesMode.FAIL_ON_PROJECT_REPOS
    repositories {
        mavenCentral()
    }
}

include(
    "numjvm-core",
    "numjvm-jit",
    "numjvm-linalg",
    "numjvm-vector",
    "numjvm-io",
    "numjvm-bench",
    "examples:basic-arrays",
    "examples:accelerated-arrays",
    "examples:matmul",
    "examples:fused-expressions",
    "examples:memory-scopes",
)
