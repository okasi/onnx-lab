#!/usr/bin/env bash
# Build ONNX Runtime main with Node.js bindings.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_DIR="${ORT_DIR:-$ROOT/vendor/onnxruntime}"
ORT_LOG="${ORT_LOG:-$ROOT/vendor/ort-build.log}"

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4\n'
  fi
}

JOBS="${JOBS:-$(detect_jobs)}"
export CC="${CC:-cc}"
export CXX="${CXX:-c++}"

echo "==> ORT source: $ORT_DIR"
echo "==> Compiler: $CC / $CXX"

if ! command -v cmake >/dev/null; then
  echo "cmake is required" >&2
  exit 1
fi

LINK_TEST="$(mktemp "${TMPDIR:-/tmp}/ort-cxx-test.XXXXXX")"
trap 'rm -f "$LINK_TEST"' EXIT
if ! "$CXX" -x c++ - -o "$LINK_TEST" <<< 'int main(){}' 2>/dev/null; then
  echo "C++ linker failed; set CC and CXX to a working toolchain." >&2
  exit 1
fi

mkdir -p "$(dirname "$ORT_DIR")"

if [[ ! -d "$ORT_DIR/.git" ]]; then
  echo "==> Cloning onnxruntime (shallow)..."
  git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git "$ORT_DIR"
fi

cd "$ORT_DIR"
echo "==> ORT version: $(cat VERSION_NUMBER)"

echo "==> Building Release + nodejs binding (log: $ORT_LOG)..."
./build.sh \
  --config Release \
  --build_shared_lib \
  --parallel "$JOBS" \
  --build_nodejs \
  --skip_tests \
  --skip_nodejs_tests \
  2>&1 | tee "$ORT_LOG"

echo "==> Installing js workspace deps..."
cd "$ORT_DIR/js" && npm ci
cd "$ORT_DIR/js/common" && npm ci
cd "$ORT_DIR/js/node" && ONNXRUNTIME_NODE_INSTALL=skip npm ci

PLATFORM="$(node -p 'process.platform')"
ARCH="$(node -p 'process.arch')"
BINDING="$ORT_DIR/js/node/bin/napi-v6/$PLATFORM/$ARCH/onnxruntime_binding.node"
if [[ ! -f "$BINDING" ]]; then
  echo "Build finished but binding not found: $BINDING" >&2
  exit 1
fi

echo "==> Done. Binding: $BINDING"
echo "==> Reinstall onnx-lab deps: cd $ROOT && npm install"
