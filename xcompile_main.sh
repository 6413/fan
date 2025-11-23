#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

set -e

echo ""
read -p "Delete build folder? (Y/n) " ANSWER
ANSWER=${ANSWER,,}

if [[ "$ANSWER" != "n" && "$ANSWER" != "no" ]]; then
    echo -e "${BLUE}[1/3]${NC} Cleaning build directory..."
    rm -rf build .xmake
else
    echo -e "${YELLOW}Skipping build cleanup.${NC}"
fi

echo ""
echo -e "${BLUE}[2/3]${NC} Configuring XMake..."
if ! xmake f -c; then
    echo -e "${RED}✗ XMake clean configuration failed!${NC}"
    exit 1
fi

if ! xmake f --toolchain=clang --cc=clang-20 --cxx=clang++-20; then
    echo -e "${RED}✗ XMake configuration failed!${NC}"
    exit 1
fi

echo ""
xmake -vD

target_name=$(grep -E 'target\("([^"]+\.exe)"' xmake.lua | sed -E 's/target\("([^"]+\.exe)".*/\1/' | head -n1)
if [ -z "$target_name" ]; then
    echo -e "${RED}Error: Could not find target name ending with .exe in xmake.lua${NC}"
    exit 1
fi

exe_path=$(find build -type f -name "${target_name}" | head -n1)
if [ -z "$exe_path" ]; then
    echo -e "${RED}Error: Built executable for target '${target_name}' not found${NC}"
    exit 1
fi

cp "$exe_path" .
echo -e "${GREEN} Copied: $exe_path → ./$(basename "$exe_path")${NC}"