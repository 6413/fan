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

declare -A FEATURE_DEFAULTS=(
  [FAN_WINDOW]=false [FAN_2D]=false [FAN_GUI]=false
  [FAN_PHYSICS_2D]=false [FAN_JSON]=false [FAN_OPENGL]=false
  [FAN_3D]=false [FAN_VULKAN]=false [FAN_FMT]=false
  [FAN_WAYLAND_SCREEN]=false [FAN_NETWORK]=false [FAN_AUDIO]=false
  [FAN_VIDEO]=false [FAN_REFLECTION]=false
)
declare -A FEATURES=()
PRESET_USED=false

enable_features() {
  for f in "$@"; do FEATURES[$f]=true; done
}

apply_preset_core()     { : ; }
apply_preset_headless() { enable_features FAN_JSON; }
apply_preset_window()   { apply_preset_headless; enable_features FAN_WINDOW FAN_OPENGL; }
apply_preset_2d()       { apply_preset_window;  enable_features FAN_2D FAN_GUI FAN_PHYSICS_2D; }
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      MODE="debug"
      shift
      ;;
    --release)
      MODE="release"
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
      XMAKE_ARGS+=("--buildlib=y")
      shift
      ;;
    --core)        PRESET_USED=true; apply_preset_core;     shift ;;
    --headless)    PRESET_USED=true; apply_preset_headless; shift ;;
    --window)      PRESET_USED=true; apply_preset_window;   shift ;;
    --2d)          PRESET_USED=true; apply_preset_2d;       shift ;;
    --audio)       enable_features FAN_AUDIO;               shift ;;
    --network)     enable_features FAN_NETWORK;             shift ;;
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
    FEATURE_ARGS+=("--${flag}=y")
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
      if [[ $name =~ ^${binary}-([0-9]+)$ ]]; then
        local ver="${BASH_REMATCH[1]}"
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
  else
    CXX=$(find_compiler "clang++" 20)
    CC=$(find_compiler "clang" 20)
    TOOLCHAIN="clang"
  fi
  CONFIG_ARGS=("--compiler=$COMPILER" "--toolchain=$TOOLCHAIN" "--cc=$CC" "--cxx=$CXX")
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
  echo ""
  echo -e "${BLUE}[2/3]${NC} Configuring..."
  if ! xmake f -c "${CONFIG_ARGS[@]}" "${FEATURE_ARGS[@]}" "${XMAKE_ARGS[@]}"; then
    echo -e "${RED}✗ XMake configuration failed!${NC}"
    exit 1
  fi
  echo ""
  echo -e "${BLUE}[3/3]${NC} Building..."
else
  echo -e "${BLUE}Configuring & Building...${NC}"
  if ! xmake f -c "${CONFIG_ARGS[@]}" "${FEATURE_ARGS[@]}" "${XMAKE_ARGS[@]}"; then
    echo -e "${RED}✗ XMake configuration failed!${NC}"
    exit 1
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
  target_name=$(grep -E 'target\("([^"]+\.exe)"' xmake.lua | sed -E 's/target\("([^"]+\.exe)".*/\1/' | head -n1)
  if [ -z "$target_name" ]; then
    echo -e "${RED}Error:${NC} Could not find target name ending with .exe in xmake.lua"
    exit 1
  fi
  exe_path=$(find build -type f -name "${target_name}" | head -n1)
  if [ -z "$exe_path" ]; then
    echo -e "${RED}Error:${NC} Built executable for target '${target_name}' not found"
    exit 1
  fi
  cp "$exe_path" .
  echo -e "${GREEN}✓ Copied:${NC} ${exe_path} → ./$(basename "$exe_path")"
fi