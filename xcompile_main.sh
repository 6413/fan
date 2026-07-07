#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
set -e
MODE=""
REBUILD=false
MAIN_FILE=""
COMPILER="clang"
WASM=false
XMAKE_ARGS=()
FEATURE_ARGS=()
BUILDLIB=false

declare -A FEATURE_DEFAULTS=(
  [FAN_WINDOW]=false [FAN_2D]=false [FAN_GUI]=false
  [FAN_PHYSICS_2D]=false [FAN_JSON]=false
  [FAN_3D]=false [FAN_VULKAN]=false [FAN_FMT]=false
  [FAN_WAYLAND_SCREEN]=false [FAN_NETWORK]=false [FAN_AUDIO]=false
  [FAN_VIDEO]=false [FAN_REFLECTION]=false
)
declare -A FEATURES=()
PRESET_USED=false

print_usage() {
  cat <<'EOF'
Usage: ./xcompile_main.sh [mode] [preset] [features] [xmake args...]

Modes:
  --debug | --release | --release-minsize | --asan
  --rebuild
  --main <file>
  --clang | --gcc
  --wasm
  --buildlib

Presets:
  --core
      Build the base fan modules only.
  --headless
      Build core + JSON.
  --only-network, --network-only
      Build core + JSON + network.
  --window
      Build headless + window + OpenGL.
  --2d
      Build window + 2D + GUI + physics 2D.

Feature toggles:
  --enable-<feature>
  --disable-<feature>

Features:
  window, 2d, gui, physics-2d, json, opengl, 3d, vulkan, fmt,
  wayland-screen, network, audio, video, reflection

Legacy enable aliases:
  --audio --network --3d --video --fmt --reflection --wayland-screen --vulkan

Examples:
  ./xcompile_main.sh --only-network --main examples/network/network_socket.cpp
  ./xcompile_main.sh --core --enable-network --enable-json
  ./xcompile_main.sh --2d --disable-audio --release
EOF
}

enable_features() {
  for f in "$@"; do FEATURES[$f]=true; done
}

feature_name_to_key() {
  local name="$1"
  name="${name#--}"
  name="${name#enable-}"
  name="${name#disable-}"
  name="${name//-/_}"
  name="${name^^}"
  if [[ "$name" != FAN_* ]]; then
    name="FAN_${name}"
  fi
  echo "$name"
}

set_feature_from_arg() {
  local key
  key=$(feature_name_to_key "$1")
  local value="$2"
  if [[ -z "${FEATURE_DEFAULTS[$key]+x}" ]]; then
    echo -e "${RED}Error:${NC} Unknown feature '$1' (${key})" >&2
    exit 1
  fi
  FEATURES[$key]="$value"
  if [[ "$key" == "FAN_NETWORK" && "$value" == true ]]; then
    enable_features FAN_JSON
  fi
}

apply_preset_core()     { : ; }
apply_preset_headless() { enable_features FAN_JSON; }
apply_preset_network()  { enable_features FAN_JSON FAN_NETWORK; }
apply_preset_window()   { apply_preset_headless; enable_features FAN_WINDOW; }
apply_preset_2d()       { apply_preset_window;  enable_features FAN_2D FAN_GUI FAN_PHYSICS_2D; }
while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)
      print_usage
      exit 0
      ;;
    --debug)
      MODE="debug"
      shift
      ;;
    --release)
      MODE="release"
      shift
      ;;
    --release-minsize)
      MODE="release-minsize"
      shift
      ;;
    --asan)
      MODE="asan"
      shift
      ;;
    --rebuild)
      REBUILD=true
      shift
      ;;
    --main)
      MAIN_FILE="$2"
      shift 2
      ;;
    --gcc)
      COMPILER="gcc"
      shift
      ;;
    --clang)
      COMPILER="clang"
      shift
      ;;
    --wasm)
      WASM=true
      shift
      ;;
    --buildlib)
      BUILDLIB=true
      FEATURE_ARGS+=("--buildlib=y")
      shift
      ;;
    --core)        PRESET_USED=true; apply_preset_core;     shift ;;
    --headless)    PRESET_USED=true; apply_preset_headless; shift ;;
    --only-network|--network-only)
      PRESET_USED=true
      apply_preset_network
      shift
      ;;
    --window)      PRESET_USED=true; apply_preset_window;   shift ;;
    --2d)          PRESET_USED=true; apply_preset_2d;       shift ;;
    --enable-*)
      set_feature_from_arg "$1" true
      shift
      ;;
    --disable-*)
      set_feature_from_arg "$1" false
      shift
      ;;
    --audio)       enable_features FAN_AUDIO;               shift ;;
    --network)     apply_preset_network;                    shift ;;
    --3d)          enable_features FAN_3D;                  shift ;;
    --video)       enable_features FAN_VIDEO;               shift ;;
    --fmt)         enable_features FAN_FMT;                 shift ;;
    --reflection)  enable_features FAN_REFLECTION;          shift ;;
    --wayland-screen) enable_features FAN_WAYLAND_SCREEN;   shift ;;
    --vulkan)      enable_features FAN_VULKAN;              shift ;;
    *)
      XMAKE_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$PRESET_USED" == true ]]; then
  for flag in "${!FEATURE_DEFAULTS[@]}"; do
    val=${FEATURES[$flag]:-false}
    if [[ "$val" == true ]]; then
      FEATURE_ARGS+=("--${flag}=y")
    else
      FEATURE_ARGS+=("--${flag}=n")
    fi
  done
elif [[ ${#FEATURES[@]} -gt 0 ]]; then
  for flag in "${!FEATURES[@]}"; do
    if [[ "${FEATURES[$flag]}" == true ]]; then
      FEATURE_ARGS+=("--${flag}=y")
    else
      FEATURE_ARGS+=("--${flag}=n")
    fi
  done
fi

find_compiler() {
  local binary="$1"
  local min_version="${2:-0}"
  local best_bin=""
  local best_version=0

  IFS=':' read -ra path_dirs <<< "$PATH"
  for dir in "${path_dirs[@]}"; do
    for bin in "$dir"/${binary}-*; do
      [[ -x "$bin" ]] || continue
      local name=$(basename "$bin")
      if [[ $name =~ ^(.*)-([0-9]+)$ ]] && [[ "${BASH_REMATCH[1]}" == "$binary" ]]; then
        local ver="${BASH_REMATCH[2]}"
        if (( ver >= min_version && ver > best_version )); then
          best_version=$ver
          best_bin=$bin
        fi
      fi
    done
  done

  if [[ -z "$best_bin" ]] && command -v "$binary" >/dev/null 2>&1; then
    best_bin=$(command -v "$binary")
  fi

  if [[ -z "$best_bin" ]]; then
    printf "\033[0;31mError: No suitable %s found (>= %d)\033[0m\n" "$binary" "$min_version" >&2
    return 1
  fi
  echo "$best_bin"
}

if [[ "$WASM" == true ]]; then
  # Verify emscripten is available
  if ! command -v emcc &> /dev/null; then
    echo -e "${RED}Error: emcc not found. Source your emsdk_env.sh first:${NC}"
    echo "  source /path/to/emsdk/emsdk_env.sh"
    exit 1
  fi
  EMSDK_ROOT=$(dirname $(dirname $(which emcc)))
  CONFIG_ARGS=("-p" "wasm" "--sdk=${EMSDK_ROOT}/upstream/emscripten")
  if [[ -n "$MAIN_FILE" ]]; then
    CONFIG_ARGS+=("--main=$MAIN_FILE")
  fi
  if [[ -n "$MODE" ]]; then
    CONFIG_ARGS+=("-m" "$MODE")
  fi
  echo -e "${CYAN}Wasm build using:${NC} $(emcc --version | head -1)"
else
  if [[ "$COMPILER" == "gcc" ]]; then
    CXX=$(find_compiler "g++" 0)
    CC=$(find_compiler "gcc" 0)
    TOOLCHAIN="gcc"
    BIN_CXX="g++"
  else
    CXX=$(find_compiler "clang++" 20)
    CC=$(find_compiler "clang" 20)
    SCAN=$(find_compiler "clang-scan-deps" 20 || true)
    TOOLCHAIN="clang"
    BIN_CXX="clang++"
  fi

  mkdir -p .xmake/bin
  ln -sf "$CC" .xmake/bin/$COMPILER
  ln -sf "$CXX" .xmake/bin/$BIN_CXX
  if [[ -n "$SCAN" ]]; then
    ln -sf "$SCAN" .xmake/bin/clang-scan-deps
  fi
  export PATH="$PWD/.xmake/bin:$PATH"

  RESOURCE_DIR=$($CXX -print-resource-dir 2>/dev/null) || true
  if [[ -n "$RESOURCE_DIR" ]]; then
    CONFIG_ARGS=("--compiler=$COMPILER" "--toolchain=$TOOLCHAIN" "--cxxflags=-resource-dir=$RESOURCE_DIR")
  else
    CONFIG_ARGS=("--compiler=$COMPILER" "--toolchain=$TOOLCHAIN")
  fi
  if [[ -n "$MAIN_FILE" ]]; then
    CONFIG_ARGS+=("--main=$MAIN_FILE")
  fi
  if [[ -n "$MODE" ]]; then
    CONFIG_ARGS+=("-m" "$MODE")
  fi
fi

if [[ "$REBUILD" == true ]]; then
  echo -e "${BLUE}[1/3]${NC} Cleaning build directory..."
  rm -rf build .xmake

  mkdir -p .xmake/bin
  ln -sf "$CC" .xmake/bin/$COMPILER
  ln -sf "$CXX" .xmake/bin/$BIN_CXX
  if [[ -n "$SCAN" ]]; then
    ln -sf "$SCAN" .xmake/bin/clang-scan-deps
  fi

  echo ""
  echo -e "${BLUE}[2/3]${NC} Configuring..."
  if ! xmake f -c "${CONFIG_ARGS[@]}" "${FEATURE_ARGS[@]}" "${XMAKE_ARGS[@]}"; then
    echo -e "${RED}✗ XMake configuration failed!${NC}"
    exit 1
  fi
  echo ""
  echo -e "${BLUE}[3/3]${NC} Building..."
else
  CFG_STR="${CONFIG_ARGS[*]} ${FEATURE_ARGS[*]} ${XMAKE_ARGS[*]}"
  if [[ ! -f .xmake/cfg_cache ]] || [[ "$(cat .xmake/cfg_cache 2>/dev/null)" != "$CFG_STR" ]]; then
    echo -e "${BLUE}Configuring...${NC}"
    if ! xmake f -c "${CONFIG_ARGS[@]}" "${FEATURE_ARGS[@]}" "${XMAKE_ARGS[@]}"; then
      echo -e "${RED}✗ XMake configuration failed!${NC}"
      exit 1
    fi
    mkdir -p .xmake
    echo "$CFG_STR" > .xmake/cfg_cache
  fi
fi

if [[ "$WASM" == false ]]; then
  echo -e "${CYAN}Compiler:${NC} $($CXX --version | head -1)"
fi

if ! xmake -j$(nproc) "${XMAKE_ARGS[@]}"; then
  echo -e "${RED}✗ XMake build failed!${NC}"
  exit 1
fi

if [[ "$WASM" == true ]]; then
  # Find the .html output
  html_path=$(find build -type f -name "*.html" | head -n1)
  if [ -z "$html_path" ]; then
    echo -e "${RED}Error:${NC} Built .html not found in build/"
    exit 1
  fi
  # Copy html + wasm + js together
  base=$(basename "$html_path" .html)
  dir=$(dirname "$html_path")
  for ext in html wasm js; do
    f="$dir/$base.$ext"
    if [ -f "$f" ]; then
      cp "$f" .
      echo -e "${GREEN}✓ Copied:${NC} $f → ./$base.$ext"
    fi
  done
else
  if [[ "$BUILDLIB" == true ]]; then
    exit 0
  fi

  target_name="a"
  if [[ -n "$MODE" ]]; then
    mode_dir="${MODE}"
  else
    mode_dir="mode_none"
  fi
  exe_path=$(find build -path "*/${mode_dir}/*" \( -name "$target_name" -o -name "$target_name.exe" \) -perm -111 | head -n1)
  if [ -z "$exe_path" ]; then
    exe_path=$(find build -path "*/${mode_dir}/*" \( -name "$target_name" -o -name "$target_name.exe" \) | head -n1)
  fi
  if [ -z "$exe_path" ]; then
    echo -e "${RED}Error:${NC} Built executable '${target_name}.exe' not found"
    exit 1
  fi

  out_path="./${target_name}.exe"
  cp "$exe_path" "$out_path"
  echo -e "${GREEN}✓ Copied:${NC} ${exe_path} → ${out_path}"
fi
