#!/bin/bash
INSTALL_DIR="$(pwd)/thirdparty/fan"
INCLUDE_DIR="$INSTALL_DIR/include"
LIB_DIR="$INSTALL_DIR/lib"
mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

# Check for --force flag to rebuild everything
FORCE_REBUILD=false
if [[ "$1" == "--force" ]]; then
    FORCE_REBUILD=true
    echo "Force rebuild enabled - will rebuild all libraries"
fi

move_and_pull() {
	REPO_URL=$1
	DIR_NAME=$2
	REPO_DIR="$INSTALL_DIR/repos/$DIR_NAME"
	TARGET_DIR="$INCLUDE_DIR/$DIR_NAME"
	
	mkdir -p "$INSTALL_DIR/repos"
	
	if [ -d "$REPO_DIR/.git" ]; then
		echo "Updating $DIR_NAME..."
		cd "$REPO_DIR"
		git pull || { echo "failed to update $DIR_NAME"; exit 1; }
		cd - > /dev/null
	else
		echo "Cloning $DIR_NAME..."
		git clone --depth 1 "$REPO_URL" "$REPO_DIR"
	fi
	
	rm -rf "$TARGET_DIR"
	mkdir -p "$TARGET_DIR"
	
	if [ -d "$REPO_DIR/$DIR_NAME" ]; then
		echo "Found nested structure $DIR_NAME, copying from $DIR_NAME/$DIR_NAME/"
		cp -r "$REPO_DIR/$DIR_NAME"/* "$TARGET_DIR/"
	else
		find "$REPO_DIR" -maxdepth 1 -type f -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" | xargs -I {} cp {} "$TARGET_DIR/" 2>/dev/null || true
		find "$REPO_DIR" -maxdepth 1 -type d ! -name ".git" ! -path "$REPO_DIR" | xargs -I {} cp -r {} "$TARGET_DIR/" 2>/dev/null || true
	fi
}

# Always update header-only libraries (they're quick)
move_and_pull "https://github.com/7244/WITCH.git" "WITCH"
move_and_pull "https://github.com/7244/BCOL.git" "BCOL"
move_and_pull "https://github.com/7244/BLL.git" "BLL"
move_and_pull "https://github.com/7244/BVEC.git" "BVEC"
move_and_pull "https://github.com/7244/BDBT.git" "BDBT"
move_and_pull "https://github.com/7244/bcontainer.git" "bcontainer"
move_and_pull "https://github.com/7244/pixfconv.git" "pixfconv"
move_and_pull "https://github.com/6413/PIXF.git" "PIXF"

# Build Box2D (only if not exists or force rebuild)
if [[ "$FORCE_REBUILD" == "true" ]] || [[ ! -f "$LIB_DIR/libbox2d.a" ]]; then
    echo "Building Box2D..."
    REPO_DIR="$INSTALL_DIR/box2d"
    
    # Remove old repo if it exists
    rm -rf "$REPO_DIR"
    
    git clone https://github.com/erincatto/box2d.git "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout v3.1.1
    mkdir build
    cd build
    cmake -DBOX2D_SAMPLES=OFF \
          -DBOX2D_BENCHMARKS=OFF \
          -DBOX2D_DOCS=OFF \
          -DBOX2D_PROFILE=OFF \
          -DBOX2D_VALIDATE=OFF \
          -DBOX2D_UNIT_TESTS=OFF \
          -DUSE_SIMD=OFF \
          -DBOX2D_AVX2=OFF \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" ..
    make -j$(nproc)
    make install
    cd ..
    if [ -d "include/box2d" ]; then
        cp -r include/box2d "$INCLUDE_DIR/"
    else
        echo "failed to find includes for box2d"
    fi
    cd ..
    rm -rf "$REPO_DIR"
    echo "✓ Box2D built successfully"
else
    echo "✓ Box2D already exists, skipping build"
fi

# Build FreeType (only if not exists or force rebuild)
if [[ "$FORCE_REBUILD" == "true" ]] || [[ ! -f "$LIB_DIR/libfreetype.a" ]]; then
    echo "Building FreeType..."
    FREETYPE_DIR="$INSTALL_DIR/freetype"

    rm -rf "$FREETYPE_DIR"
    rm -f freetype-2.13.2.tar.xz

    wget -O freetype-2.13.2.tar.xz https://download.savannah.gnu.org/releases/freetype/freetype-2.13.2.tar.xz
    tar -xf freetype-2.13.2.tar.xz
    mv freetype-2.13.2 "$FREETYPE_DIR"
    cd "$FREETYPE_DIR"
    mkdir build
    cd build

    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DFT_DISABLE_HARFBUZZ=ON \
        -DFT_DISABLE_BROTLI=ON \
        -DFT_DISABLE_PNG=OFF \
        -DFT_DISABLE_ZLIB=OFF \
        -DFT_DISABLE_BZIP2=ON \
        -DBUILD_SHARED_LIBS=OFF

    make -j$(nproc)
    make install
    cd ../..

    rm -f freetype-2.13.2.tar.xz
    echo "✓ FreeType built and installed (kept in $INSTALL_DIR)"
else
    echo "✓ FreeType already exists, skipping build"
fi

# Build LunaSVG (only if not exists or force rebuild)
if [[ "$FORCE_REBUILD" == "true" ]] || [[ ! -f "$LIB_DIR/liblunasvg.a" ]]; then
    echo "Building LunaSVG..."
    LUNASVG_DIR="$INSTALL_DIR/lunasvg"
    
    # Remove old sources if they exist
    rm -rf "$LUNASVG_DIR"
    rm -f lunasvg-2.4.1.tar.gz
    
    wget -O lunasvg-2.4.1.tar.gz https://github.com/sammycage/lunasvg/archive/refs/tags/v2.4.1.tar.gz
    tar -xf lunasvg-2.4.1.tar.gz
    mv lunasvg-2.4.1 "$LUNASVG_DIR"
    cd "$LUNASVG_DIR"
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DBUILD_SHARED_LIBS=OFF \
          -DLUNASVG_BUILD_EXAMPLES=OFF \
          -DCMAKE_C_COMPILER=/usr/bin/clang \
          -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 \
          -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
          -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi" \
          -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -lc++abi" ..
    make -j$(nproc)
    make install
    cd ../..
    rm -rf "$LUNASVG_DIR"
    rm -f lunasvg-2.4.1.tar.gz
    echo "✓ LunaSVG built successfully"
else
    echo "✓ LunaSVG already exists, skipping build"
fi

echo ""
echo "All dependencies processed successfully!"
echo "Built libraries:"
if [[ -f "$LIB_DIR/libfreetype.a" ]]; then
    echo "✓ FreeType: $(ls -la $LIB_DIR/libfreetype.a | awk '{print $5, $9}')"
else
    echo "✗ FreeType: not found"
fi

if [[ -f "$LIB_DIR/libbox2d.a" ]]; then
    echo "✓ Box2D: $(ls -la $LIB_DIR/libbox2d.a | awk '{print $5, $9}')"
else
    echo "✗ Box2D: not found"
fi

if [[ -f "$LIB_DIR/liblunasvg.a" ]]; then
    echo "✓ LunaSVG: $(ls -la $LIB_DIR/liblunasvg.a | awk '{print $5, $9}')"
else
    echo "✗ LunaSVG: not found"
fi

echo ""
echo "Usage:"
echo "  ./install.sh         # Build only missing libraries"
echo "  ./install.sh --force # Rebuild all libraries"