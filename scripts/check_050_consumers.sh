#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
version="${1:-$(sed -n 's/^version=//p' gradle.properties)}"
repo="$(pwd)/build/staging-repo/io/github/stamich"
core="$repo/numjvm-core/$version/numjvm-core-$version.jar"
jit="$repo/numjvm-jit/$version/numjvm-jit-$version.jar"
linalg="$repo/numjvm-linalg/$version/numjvm-linalg-$version.jar"
vector="$repo/numjvm-vector/$version/numjvm-vector-$version.jar"
for jar in "$core" "$jit" "$linalg" "$vector"; do test -f "$jar"; done
workdir="$(mktemp -d "${TMPDIR:-/tmp}/numjvm-050-consumer.XXXXXXXX")"
trap 'rm -rf "$workdir"' EXIT
mkdir -p "$workdir/classpath" "$workdir/modules/numjvm.releasecheck.fusion"
javac --release 25 -cp "$core:$jit:$linalg" -d "$workdir/classpath" release-check-050/src/smoke/Consumer.java
javac --release 25 --module-path "$core:$jit:$linalg" -d "$workdir/modules/numjvm.releasecheck.fusion" release-check-050/module/module-info.java release-check-050/src/smoke/Consumer.java
classpath="$core:$jit:$linalg:$workdir/classpath"
modules="$core:$jit:$linalg:$workdir/modules"
java --enable-native-access=ALL-UNNAMED -Dnumjvm.linalg=java -cp "$classpath" smoke.Consumer
# An installed but unresolved vector provider must not break scalar consumers.
java --enable-native-access=ALL-UNNAMED -Dnumjvm.linalg=java -cp "$classpath:$vector" smoke.Consumer
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector -Dnumjvm.linalg=java -Dnumjvm.simd=required -Dnumjvm.expect.fused=true -cp "$classpath:$vector" smoke.Consumer
java --enable-native-access=org.numjvm.linalg -Dnumjvm.linalg=java --module-path "$modules" -m numjvm.releasecheck.fusion/smoke.Consumer
java --enable-native-access=org.numjvm.linalg --add-modules org.numjvm.vector -Dnumjvm.linalg=java -Dnumjvm.simd=required -Dnumjvm.expect.fused=true --module-path "$modules:$vector" -m numjvm.releasecheck.fusion/smoke.Consumer
if [[ -n "${NUMJVM_OPENBLAS_PATH:-}" ]]; then
  java --enable-native-access=ALL-UNNAMED -Dnumjvm.linalg=openblas -Dnumjvm.openblas.path="$NUMJVM_OPENBLAS_PATH" -Dnumjvm.expect.native=true -cp "$classpath" smoke.Consumer
  java --enable-native-access=org.numjvm.linalg -Dnumjvm.linalg=openblas -Dnumjvm.openblas.path="$NUMJVM_OPENBLAS_PATH" -Dnumjvm.expect.native=true --module-path "$modules" -m numjvm.releasecheck.fusion/smoke.Consumer
else
  printf '%s\n' 'OpenBLAS consumer checks skipped: NUMJVM_OPENBLAS_PATH is not set.'
fi
