// Shared configuration for all NumJVM modules.
//
// Library modules (numjvm-core, numjvm-io) are JPMS modules compiled for Java 25.
// Tests have no module-info and run on the class path, which gives them access to
// internal packages (white-box tests).

plugins {
    base
    alias(libs.plugins.sonarqube)
}

val javaVersion = 25

subprojects {
    apply(plugin = "java")
    apply(plugin = "jacoco")

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

    // Coverage: every test run produces build/reports/jacoco/test/jacocoTestReport.xml,
    // which the SonarQube plugin picks up automatically.
    extensions.configure<JacocoPluginExtension> {
        toolVersion = rootProject.libs.versions.jacoco.get()
    }
    val jacocoTestReport = tasks.named<JacocoReport>("jacocoTestReport") {
        reports {
            xml.required = true
            csv.required = true
            html.required = true
        }
    }
    tasks.named<Test>("test") {
        finalizedBy(jacocoTestReport)
    }
}

// --------------------------------------------------------------------------- SonarQube
//
// Analysis runs on SonarCloud from .github/workflows/sonar.yml:
//
//   SONAR_TOKEN=... ./gradlew build sonar
//
// The defaults below point at SonarCloud. For a self-hosted SonarQube server, set
// SONAR_HOST_URL and override the keys, e.g. -Dsonar.projectKey=numjvm.
//
// Scope (decision D1, docs/quality/README.md): the main code of the published libraries,
// numjvm-core and numjvm-io, with their tests analysed as test code. Benchmarks, examples,
// the architecture spike, release-check and the oracle fixtures are not analysed.

sonar {
    properties {
        property("sonar.projectKey", "stamich_NumJVM")
        property("sonar.organization", "stamich")
        property("sonar.host.url", "https://sonarcloud.io")
        property("sonar.projectName", "NumJVM")
        property("sonar.gradle.scanAll", "false")
        property("sonar.sourceEncoding", "UTF-8")
        property("sonar.links.scm", "https://github.com/stamich/NumJVM")
        property("sonar.links.ci", "https://github.com/stamich/NumJVM/actions")
        // Decision D3: generated inner loops are analysed, but not counted as duplication.
        property("sonar.cpd.exclusions", "**/build/generated/**")
        // Decision D2: the analysis step fails when the quality gate fails.
        property("sonar.qualitygate.wait", "true")
        property("sonar.qualitygate.timeout", "300")
    }
}

val sonarSkipped = setOf("numjvm-bench", "examples", "basic-arrays", "memory-scopes", "accelerated-arrays", "matmul")

configure(subprojects.filter { it.name in sonarSkipped }) {
    extensions.configure<org.sonarqube.gradle.SonarExtension> {
        isSkipProject = true
    }
}

// ---------------------------------------------------------------------------- publishing
//
// numjvm-core and numjvm-io are published with sources and Javadoc. Releases are first
// staged in build/staging-repo (./gradlew publish) and can be installed locally with
// ./gradlew publishToMavenLocal; release-check/ consumes them like a user project would.

val publishedModules = setOf("numjvm-core", "numjvm-io", "numjvm-vector", "numjvm-linalg")

configure(subprojects.filter { it.name in publishedModules }) {
    apply(plugin = "maven-publish")

    extensions.configure<JavaPluginExtension> {
        withSourcesJar()
        withJavadocJar()
    }

    // Decision D2: at least 80 % line coverage for the published libraries, checked by
    // `./gradlew check` (and so by `build`) as well as by the SonarCloud quality gate.
    val coverageVerification = tasks.named<JacocoCoverageVerification>("jacocoTestCoverageVerification") {
        violationRules {
            rule {
                limit {
                    counter = "LINE"
                    minimum = "0.80".toBigDecimal()
                }
            }
        }
    }
    tasks.named("check") {
        // Linalg's coverage gate combines Java and required-native test jobs. A
        // Java-only build must remain possible without installing a BLAS library.
        if (project.name != "numjvm-linalg") { dependsOn(coverageVerification) }
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

// Optional providers are test runtime dependencies only: production core stays independent.
configure(subprojects.filter { it.name in setOf("numjvm-core", "numjvm-io") }) {
    if (providers.gradleProperty("simdEnabled").orElse("false").get().toBoolean()) {
        dependencies.add("testRuntimeOnly", project(":numjvm-vector"))
        tasks.withType<Test>().configureEach {
            jvmArgs("--add-modules", "jdk.incubator.vector")
            systemProperty("numjvm.test.simdInstalled", "true")
            systemProperty("numjvm.simd", providers.gradleProperty("simdMode").orElse("auto").get())
        }
    }
}
