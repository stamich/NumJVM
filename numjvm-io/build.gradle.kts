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
val oracleResults = layout.buildDirectory.dir("oracle-out")
tasks.test {
    inputs.dir(oracleFixtures).withPathSensitivity(PathSensitivity.RELATIVE)
    // NumPy verifies these files in a later CI step, including on a cache hit.
    outputs.dir(oracleResults)
    systemProperty("numjvm.oracle.fixtures", oracleFixtures.absolutePath)
    systemProperty("numjvm.oracle.out", oracleResults.get().asFile.absolutePath)
}
