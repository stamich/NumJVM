#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
version="${1:-0.3.0-alpha}"
repo="$(pwd)/build/staging-repo/io/github/stamich"
core="$repo/numjvm-core/$version/numjvm-core-$version.jar"
jit="$repo/numjvm-jit/$version/numjvm-jit-$version.jar"
vector="$repo/numjvm-vector/$version/numjvm-vector-$version.jar"
test -f "$core"
test -f "$jit"
workdir="$(mktemp -d "${TMPDIR:-/tmp}/numjvm-jit-consumer.XXXXXXXX")"
trap 'rm -rf "$workdir"' EXIT
mkdir -p "$workdir/classpath" "$workdir/modules/numjvm.releasecheck.jit"
javac --release 25 -cp "$core:$jit" -d "$workdir/classpath" release-check-jit/src/smoke/JitConsumer.java
java -cp "$core:$jit:$workdir/classpath" smoke.JitConsumer
# An installed but unresolved Vector provider must preserve automatic fallback.
java -cp "$core:$jit:$vector:$workdir/classpath" smoke.JitConsumer
java --add-modules jdk.incubator.vector -Dnumjvm.simd=required -cp "$core:$jit:$vector:$workdir/classpath" smoke.JitConsumer
javac --release 25 --module-path "$core:$jit" -d "$workdir/modules/numjvm.releasecheck.jit" release-check-jit/module/module-info.java release-check-jit/src/smoke/JitConsumer.java
java --module-path "$core:$jit:$workdir/modules" -m numjvm.releasecheck.jit/smoke.JitConsumer
java --module-path "$core:$jit:$vector:$workdir/modules" --add-modules org.numjvm.vector -Dnumjvm.simd=required -m numjvm.releasecheck.jit/smoke.JitConsumer
