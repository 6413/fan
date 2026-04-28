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
XMAKE_ARGS=()

while [[ $

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
    *)
      XMAKE_ARGS+=("$1")
      shift
      ;;
  esac
done

find_clang() {
  local min_version=20
  local best_bin=""
  local best_version=0

  while IFS= read -r path; do
    bin=$(basename "$path")

    if [[ $bin =~ ^clang\+\+-([0-9]+)$ ]]; then
      ver="${BASH_REMATCH[1]}"

      if (( ver >= min_version && ver > best_version )); then
        best_version=$ver
        best_bin=$path
      fi
    fi
  done < <(command -v -a clang++ 2>/dev/null)

  if [[ -z "$best_bin" ]]; then
    if command -v clang++ >/dev/null 2>&1; then
      best_bin=$(command -v clang++)
    fi
  fi

  if [[ -z "$best_bin" ]]; then
    printf "\033[0;31mError: No suitable clang++ found (>= %d)\033[0m\n" "$min_version" >&2
    return 1
  fi

  echo "$best_bin"
}

if [[ "$COMPILER" == "gcc" ]]; then
  CC="gcc-15"
  CXX="g++-15"
  TOOLCHAIN="gcc"
else
  CXX=$(find_clang)
  CC="${CXX/clang++/clang}"
  TOOLCHAIN="clang"
fi

CONFIG_ARGS=("--compiler=$COMPILER" "--toolchain=$TOOLCHAIN" "--cc=$CC" "--cxx=$CXX")

if [[ -n "$MAIN_FILE" ]]; then
  CONFIG_ARGS+=("--main=$MAIN_FILE")
fi

if [[ -n "$MODE" ]]; then
  CONFIG_ARGS+=("-m" "$MODE")
fi

if [[ "$REBUILD" == true ]]; then
  echo -e "${BLUE}[1/3]${NC} Cleaning build directory..."
  rm -rf build .xmake

  echo ""
  echo -e "${BLUE}[2/3]${NC} Configuring..."
  if ! xmake f -c "${CONFIG_ARGS[@]}" "${XMAKE_ARGS[@]}"; then
    echo -e "${RED}✗ XMake configuration failed!${NC}"
    exit 1
  fi

  echo ""
  echo -e "${BLUE}[3/3]${NC} Building..."
else
  echo -e "${BLUE}Configuring & Building...${NC}"
  if ! xmake f "${CONFIG_ARGS[@]}" "${XMAKE_ARGS[@]}"; then
    echo -e "${RED}✗ XMake configuration failed!${NC}"
    exit 1
  fi
fi

echo -e "${CYAN}Compiler:${NC} $($CXX --version | head -1)"
if ! xmake -j$(nproc) "${XMAKE_ARGS[@]}"; then
  echo -e "${RED}✗ XMake build failed!${NC}"
  exit 1
fi

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