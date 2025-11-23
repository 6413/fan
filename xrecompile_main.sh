#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

set -e

echo -e "${BLUE}Starting build…${NC}"

xmake

target_name=$(grep -E 'target\("([^"]+\.exe)"' xmake.lua | sed -E 's/target\("([^"]+\.exe)".*/\1/' | head -n1)
if [ -z "$target_name" ]; then
  echo -e "${RED}Error:${NC} Could not find target name ending with .exe in xmake.lua"
  exit 1
fi
echo -e "${CYAN}Detected target:${NC} ${target_name}"

exe_path=$(find build -type f -name "${target_name}" | head -n1)
if [ -z "$exe_path" ]; then
  echo -e "${RED}Error:${NC} Built executable for target '${target_name}' not found"
  exit 1
fi

cp "$exe_path" .
echo -e "${GREEN} Copied:${NC} ${exe_path} → ./$(basename "$exe_path")"
