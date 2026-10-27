// A separate build that consumes the published artifacts the way a user project would.
// Run after ./gradlew publishToMavenLocal:  ./gradlew -p release-check run

rootProject.name = "numjvm-release-check"

dependencyResolutionManagement {
    repositories {
        mavenLocal()
        mavenCentral()
    }
}
