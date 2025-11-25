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
XMAKE_ARGS=()

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
    --rebuild)
      REBUILD=true
      shift
      ;;
    --main)
      MAIN_FILE="$2"
      shift 2
      ;;
    *)
      XMAKE_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$MAIN_FILE" ]]; then
  echo -e "${CYAN}Setting main file:${NC} ${MAIN_FILE}"
  if ! xmake f --main="$MAIN_FILE" "${XMAKE_ARGS[@]}"; then
    echo -e "${RED}✗ XMake configuration failed!${NC}"
    exit 1
  fi
fi

if [[ "$REBUILD" == true ]]; then
  echo -e "${BLUE}[1/4]${NC} Cleaning build directory..."
  rm -rf build .xmake
  
  echo ""
  echo -e "${BLUE}[2/4]${NC} Initial configuration..."
  CONFIG_ARGS=("${XMAKE_ARGS[@]}")
  if [[ -n "$MAIN_FILE" ]]; then
    CONFIG_ARGS+=("--main=$MAIN_FILE")
  fi
  if ! xmake f -c "${CONFIG_ARGS[@]}"; then
    echo -e "${RED}✗ XMake clean configuration failed!${NC}"
    exit 1
  fi
  
  echo ""
  echo -e "${BLUE}[3/4]${NC} Configuring toolchain..."
  EXTRA_FLAGS=()
  if [[ -n "$MODE" ]]; then
    EXTRA_FLAGS+=("-m" "$MODE")
  fi
  
  if ! xmake f --toolchain=clang --cc=clang-20 --cxx=clang++-20 "${EXTRA_FLAGS[@]}" "${CONFIG_ARGS[@]}"; then
    echo -e "${RED}✗ XMake configuration failed!${NC}"
    exit 1
  fi
  
  echo ""
  echo -e "${BLUE}[4/4]${NC} Building..."
  if ! xmake -vD "${XMAKE_ARGS[@]}"; then
    echo -e "${RED}✗ XMake build failed!${NC}"
    exit 1
  fi
else
  echo -e "${BLUE}Building...${NC}"
  
  if [[ -n "$MODE" ]]; then
    if ! xmake f -m "$MODE" "${XMAKE_ARGS[@]}"; then
      echo -e "${RED}✗ XMake mode configuration failed!${NC}"
      exit 1
    fi
  fi
  
  if ! xmake "${XMAKE_ARGS[@]}"; then
    echo -e "${RED}✗ XMake build failed!${NC}"
    exit 1
  fi
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