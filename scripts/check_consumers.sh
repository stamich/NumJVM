#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./gradlew publishToMavenLocal --max-workers=4 --console=plain
for consumer in release-check release-check-modular; do
  ./gradlew -p "$consumer" run --console=plain
  ./gradlew -p "$consumer" run -PwithVector --console=plain
done
# Classpath auto mode must survive an installed provider whose incubator module is unresolved.
./gradlew -p release-check run -PunresolvedVector --console=plain
