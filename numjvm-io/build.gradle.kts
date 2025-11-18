plugins {
    `java-library`
}

description = "NumJVM I/O: NumPy .npy reading and writing"

dependencies {
    api(project(":numjvm-core"))
}

// The NumPy oracle (numjvm-tests-oracle) is replayed by OracleTest. Its results are
// written to build/oracle-out for verify_results.py, which runs in the oracle workflow.
val oracleFixtures = rootProject.file("numjvm-tests-oracle/fixtures")
tasks.test {
    inputs.dir(oracleFixtures).withPathSensitivity(PathSensitivity.RELATIVE)
    systemProperty("numjvm.oracle.fixtures", oracleFixtures.absolutePath)
    systemProperty("numjvm.oracle.out", layout.buildDirectory.dir("oracle-out").get().asFile.absolutePath)
}
