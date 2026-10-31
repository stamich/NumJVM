plugins { application }
dependencies { implementation(project(":numjvm-jit")) }
application { mainClass = "examples.FusedExpressions" }
