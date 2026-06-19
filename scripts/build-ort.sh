#!/usr/bin/env bash
# Build ONNX Runtime (main / 1.28+) with Node.js bindings for 2-bit GatherBlockQuantized.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_DIR="${ORT_DIR:-$ROOT/vendor/onnxruntime}"
ORT_LOG="${ORT_LOG:-$ROOT/vendor/ort-build.log}"
JOBS="${JOBS:-$(nproc)}"

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

echo "==> ORT source: $ORT_DIR"
echo "==> Compiler: $CC / $CXX"

if ! command -v cmake >/dev/null; then
  echo "cmake is required" >&2
  exit 1
fi

if ! "$CXX" -x c++ - -o /tmp/ort-cxx-test <<< 'int main(){}' 2>/dev/null; then
  echo "C++ linker failed — ensure g++ is default (not clang without libstdc++):" >&2
  echo "  export CC=gcc CXX=g++" >&2
  exit 1
fi
rm -f /tmp/ort-cxx-test

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

BINDING="$ORT_DIR/js/node/bin/napi-v6/linux/x64/onnxruntime_binding.node"
if [[ ! -f "$BINDING" ]]; then
  echo "Build finished but binding not found: $BINDING" >&2
  exit 1
fi

echo "==> Done. Binding: $BINDING"
echo "==> Reinstall onnx-lab deps: cd $ROOT && npm install"
