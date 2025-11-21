#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p build
set -e
cd build

echo ""
echo -e "${BLUE}[1/3]${NC} Configuring CMake..."
if ! cmake .. -G Ninja "$@"; then
    echo -e "${RED}✗ CMake configuration failed!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}[2/3]${NC} Compiling sources..."
START_TIME=$(date +%s)

ninja 2>&1 | while IFS= read -r line; do
    if [[ "$line" =~ \[([0-9]+)/([0-9]+)\]\ Building\ (.*)\ object\ (.*)$ ]]; then
        CURRENT="${BASH_REMATCH[1]}"
        TOTAL="${BASH_REMATCH[2]}"
        TYPE="${BASH_REMATCH[3]}"
        FILE="${BASH_REMATCH[4]}"
        PERCENT=$((CURRENT * 100 / TOTAL))
        
        echo -e "${CYAN}[${PERCENT}%]${NC} ${GREEN}${CURRENT}/${TOTAL}${NC} ${FILE}"
    elif [[ "$line" =~ \[([0-9]+)/([0-9]+)\]\ Linking ]]; then
        CURRENT="${BASH_REMATCH[1]}"
        TOTAL="${BASH_REMATCH[2]}"
        echo -e "${YELLOW}[$(( CURRENT * 100 / TOTAL ))%]${NC} Linking executable..."
    elif [[ "$line" =~ ^FAILED || "$line" =~ error: ]]; then
        echo -e "${RED}${line}${NC}"
    fi
done

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))

echo ""
if [ -f a.exe ]; then
    echo -e "${BLUE}[3/3]${NC} Moving executable..."
    mv a.exe ..
    FILE_SIZE=$(ls -lh ../a.exe | awk '{print $5}')
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo -e "${GREEN}✓ Time:       ${BUILD_TIME}s${NC}"
    echo -e "${GREEN}✓ Executable: ../a.exe (${FILE_SIZE})${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo -e "${YELLOW}⚠ Warning: a.exe not found${NC}"
    exit 1
fi