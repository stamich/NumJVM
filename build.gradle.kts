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

// ---------------------------------------------------------------------------- publishing
//
// numjvm-core and numjvm-io are published with sources and Javadoc. Releases are first
// staged in build/staging-repo (./gradlew publish) and can be installed locally with
// ./gradlew publishToMavenLocal; release-check/ consumes them like a user project would.

val publishedModules = setOf("numjvm-core", "numjvm-io")

configure(subprojects.filter { it.name in publishedModules }) {
    apply(plugin = "maven-publish")

    extensions.configure<JavaPluginExtension> {
        withSourcesJar()
        withJavadocJar()
    }

    tasks.named<Jar>("jar") {
        manifest {
            attributes(
                "Implementation-Title" to project.name,
                "Implementation-Version" to project.version,
            )
        }
    }

    extensions.configure<PublishingExtension> {
        publications {
            create<MavenPublication>("maven") {
                from(components["java"])
                pom {
                    name = project.name
                    description = provider { project.description }
                    url = "https://github.com/stamich/NumJVM"
                    licenses {
                        license {
                            name = "Apache License, Version 2.0"
                            url = "https://www.apache.org/licenses/LICENSE-2.0"
                        }
                    }
                    developers {
                        developer {
                            id = "stamich"
                            name = "Michal Stawarski"
                        }
                    }
                    scm {
                        connection = "scm:git:https://github.com/stamich/NumJVM.git"
                        url = "https://github.com/stamich/NumJVM"
                    }
                }
            }
        }
        repositories {
            maven {
                name = "staging"
                url = uri(rootProject.layout.buildDirectory.dir("staging-repo"))
            }
        }
    }
}
