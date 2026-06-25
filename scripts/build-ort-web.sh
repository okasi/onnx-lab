#!/usr/bin/env bash
# Build onnxruntime-web WASM artifacts + JS bundles (ORT main / 1.28+).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_DIR="${ORT_DIR:-$ROOT/vendor/onnxruntime}"
ORT_WEB_LOG="${ORT_WEB_LOG:-$ROOT/vendor/ort-web-build.log}"
JOBS="${JOBS:-$(nproc)}"
DIST="$ORT_DIR/js/web/dist"

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

COMMON_FLAGS=(
  --config Release
  --parallel "$JOBS"
  --skip_tests
  --skip_submodule_sync
  --build_wasm
  --target onnxruntime_webassembly
  --enable_wasm_simd
  --enable_wasm_threads
  --disable_wasm_exception_catching
  --disable_rtti
)

setup_emsdk() {
  local emsdk_root="$ORT_DIR/cmake/external/emsdk"
  if [[ ! -f "$emsdk_root/emsdk" ]]; then
    echo "emsdk submodule missing; updating submodules..."
    git -C "$ORT_DIR" submodule update --init --recursive cmake/external/emsdk
  fi
  pushd "$emsdk_root" >/dev/null
  if [[ ! -d "$emsdk_root/emsdk/upstream" ]]; then
    echo "==> Installing emsdk (this may take several minutes)..."
    ./emsdk install latest
  fi
  ./emsdk activate latest
  # shellcheck disable=SC1091
  source ./emsdk_env.sh
  popd >/dev/null
  echo "==> emcc: $(which emcc) ($(emcc --version | head -1))"
}

copy_wasm_artifact() {
  local build_dir="$1"
  local name="$2"
  local src="$ORT_DIR/$build_dir/Release/$name"
  if [[ ! -f "$src" ]]; then
    echo "Missing artifact: $src" >&2
    exit 1
  fi
  mkdir -p "$DIST"
  cp -f "$src" "$DIST/"
  # .wasm shares basename with .mjs
  if [[ "$name" == *.mjs ]]; then
    local wasm="${name%.mjs}.wasm"
    cp -f "$ORT_DIR/$build_dir/Release/$wasm" "$DIST/"
  fi
  echo "  copied $name"
}

build_wasm_variant() {
  local label="$1"
  local build_dir="$2"
  shift 2
  echo "==> WASM build: $label (dir=$build_dir)"
  ./build.sh \
    "${COMMON_FLAGS[@]}" \
    --build_dir "$build_dir" \
    "$@" \
    2>&1 | tee -a "$ORT_WEB_LOG"
}

main() {
  if [[ ! -d "$ORT_DIR/.git" ]]; then
    echo "ORT source not found at $ORT_DIR — run scripts/build-ort.sh first" >&2
    exit 1
  fi

  cd "$ORT_DIR"
  : >"$ORT_WEB_LOG"
  echo "==> ORT web build log: $ORT_WEB_LOG"

  # Prevent /workspace/package.json type:module from breaking ORT's CommonJS wasm_post_build.js
  if [[ ! -f "$ORT_DIR/package.json" ]]; then
    printf '%s\n' '{"private":true,"type":"commonjs"}' >"$ORT_DIR/package.json"
  fi

  setup_emsdk

  # Three WASM variants consumed by transformers.js / onnxruntime-web
  build_wasm_variant "simd-threaded (base)" "build_wasm_base"
  build_wasm_variant "simd-threaded.jsep" "build_wasm_jsep" --use_jsep
  build_wasm_variant "simd-threaded.asyncify (webgpu)" "build_wasm_webgpu" --use_webgpu

  echo "==> Copying WASM artifacts to js/web/dist..."
  copy_wasm_artifact build_wasm_base ort-wasm-simd-threaded.mjs
  copy_wasm_artifact build_wasm_jsep ort-wasm-simd-threaded.jsep.mjs
  copy_wasm_artifact build_wasm_webgpu ort-wasm-simd-threaded.asyncify.mjs

  # jspi bundles expect these files; reuse asyncify artifacts (transformers.js uses jsep/asyncify)
  cp -f "$DIST/ort-wasm-simd-threaded.asyncify.mjs" "$DIST/ort-wasm-simd-threaded.jspi.mjs"
  cp -f "$DIST/ort-wasm-simd-threaded.asyncify.wasm" "$DIST/ort-wasm-simd-threaded.jspi.wasm"
  echo "  stubbed jspi wasm from asyncify"

  echo "==> Installing js/web deps and building bundles..."
  cd "$ORT_DIR/js" && npm ci
  cd "$ORT_DIR/js/common" && npm ci
  cd "$ORT_DIR/js/web" && npm ci
  npm run build

  local marker="$DIST/ort.min.mjs"
  if [[ ! -f "$marker" ]]; then
    echo "onnxruntime-web bundle missing: $marker" >&2
    exit 1
  fi

  echo "==> onnxruntime-web build complete."
  echo "    dist: $DIST"
  echo "==> Reinstall onnx-lab: cd $ROOT && npm install"
}

main "$@"
