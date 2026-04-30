#!/bin/bash
INSTALL_DIR="$(pwd)/third_party/fan"
INCLUDE_DIR="$INSTALL_DIR/include"
LIB_DIR="$INSTALL_DIR/lib"
mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

FORCE_REBUILD=false
WASM_BUILD=false

for arg in "$@"; do
    case $arg in
        --force) FORCE_REBUILD=true; echo "Force rebuild enabled" ;;
        --wasm)  WASM_BUILD=true;    echo "Wasm build enabled" ;;
    esac
done

if $WASM_BUILD; then
    if ! command -v emcc &> /dev/null; then
        echo "Error: emcc not found. Please source your emsdk_env.sh first:"
        echo "  source /path/to/emsdk/emsdk_env.sh"
        exit 1
    fi
    echo "Using emcc: $(emcc --version | head -1)"
    LIB_DIR="$INSTALL_DIR/lib/wasm"
    mkdir -p "$LIB_DIR"
fi

move_and_pull() {
    local REPO_URL=$1 DIR_NAME=$2
    local REPO_DIR="$INSTALL_DIR/repos/$DIR_NAME"
    local TARGET_DIR="$INCLUDE_DIR/$DIR_NAME"
    mkdir -p "$INSTALL_DIR/repos"

    if [ -d "$REPO_DIR/.git" ]; then
        echo "Updating $DIR_NAME..."
        git -C "$REPO_DIR" pull || { echo "failed to update $DIR_NAME"; exit 1; }
    else
        echo "Cloning $DIR_NAME..."
        git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    fi

    rm -rf "$TARGET_DIR" && mkdir -p "$TARGET_DIR"

    if [ -d "$REPO_DIR/$DIR_NAME" ]; then
        echo "Found nested structure $DIR_NAME, copying from $DIR_NAME/$DIR_NAME/"
        cp -r "$REPO_DIR/$DIR_NAME"/* "$TARGET_DIR/"
    else
        find "$REPO_DIR" -maxdepth 1 -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" \) \
            | xargs -I {} cp {} "$TARGET_DIR/" 2>/dev/null || true
        find "$REPO_DIR" -maxdepth 1 -type d ! -name ".git" ! -path "$REPO_DIR" \
            | xargs -I {} cp -r {} "$TARGET_DIR/" 2>/dev/null || true
    fi
}

check() { [[ -e "$1" ]] && echo "✓ $2" || echo "✗ $2"; }

cmake_flags() {
    echo "-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_FLAGS=-stdlib=libstdc++ -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR"
}

# ─── Header-only libs ─────────────────────────────────────────────────────────
move_and_pull "https://github.com/7244/WITCH.git"       "WITCH"
move_and_pull "https://github.com/7244/BCOL.git"        "BCOL"
move_and_pull "https://github.com/7244/BLL.git"         "BLL"
move_and_pull "https://github.com/7244/BVEC.git"        "BVEC"
move_and_pull "https://github.com/7244/BDBT.git"        "BDBT"
move_and_pull "https://github.com/7244/bcontainer.git"  "bcontainer"
move_and_pull "https://github.com/7244/pixfconv.git"    "pixfconv"
move_and_pull "https://github.com/6413/PIXF.git"        "PIXF"

# ─── glad ─────────────────────────────────────────────────────────────────────
mkdir -p "$INCLUDE_DIR/glad" "$INCLUDE_DIR/KHR"

if $FORCE_REBUILD || [[ ! -f "$INCLUDE_DIR/glad/gl_native.h" ]]; then
    echo "Generating glad (native)..."
    pip install glad2 --quiet
    GLAD_DIR="$INSTALL_DIR/repos/glad"
    glad --api gl:core --out-path "$GLAD_DIR"
    cp "$GLAD_DIR/include/glad/gl.h"         "$INCLUDE_DIR/glad/gl_native.h"
    cp "$GLAD_DIR/include/KHR/khrplatform.h" "$INCLUDE_DIR/KHR/khrplatform.h"
    cp "$GLAD_DIR/src/gl.c"                  "$INSTALL_DIR/glad.c"
    echo "✓ glad native generated"
else
    echo "✓ glad native already exists, skipping"
fi

cat > "$INCLUDE_DIR/glad/gl.h" << 'EOF'
#pragma once
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
static inline int gladLoadGL(void) { return 1; }
static inline int gladLoaderLoadGL(void) { return 1; }
#else
#include "gl_native.h"
#endif
EOF
echo "✓ glad wrapper gl.h written"

# ─── libuv (wasm) ─────────────────────────────────────────────────────────────
if $WASM_BUILD; then
    LIBUV_OUT="$LIB_DIR/libuv_wasm.a"
    if $FORCE_REBUILD || [[ ! -f "$LIBUV_OUT" ]]; then
        echo "Building libuv for wasm..."
        LIBUV_SRC="/tmp/libuv_wasm_build"
        rm -rf "$LIBUV_SRC"
        git clone https://github.com/libuv/libuv.git "$LIBUV_SRC" --depth=1

        sed -i 's/return UV__ERR(pthread_setname_np(pthread_self(), namebuf));/return 0;/' \
            "$LIBUV_SRC/src/unix/thread.c"
        sed -i 's/r = pthread_getname_np(\*tid, thread_name, sizeof(thread_name));/r = 0; thread_name[0] = 0;/' \
            "$LIBUV_SRC/src/unix/thread.c"
        sed -i 's/#if defined(__linux__)/#if defined(__EMSCRIPTEN__)\n# include "uv\/posix.h"\n#elif defined(__linux__)/' \
            "$LIBUV_SRC/include/uv/unix.h"

        cat << 'EOF' >> "$LIBUV_SRC/CMakeLists.txt"

if(TARGET uv_a)
  target_sources(uv_a PRIVATE
    src/unix/posix-hrtime.c
    src/unix/posix-poll.c
    src/unix/no-fsevents.c
    src/unix/no-proctitle.c
  )
endif()
EOF

        mkdir -p "$LIBUV_SRC/build_wasm"
        cd "$LIBUV_SRC/build_wasm"
        emcmake cmake .. \
            --log-level=WARNING -Wno-dev \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_FLAGS="-pthread" \
            -DLIBUV_BUILD_SHARED=OFF \
            -DBUILD_TESTING=OFF
        emmake make -j$(nproc)

        cp "$LIBUV_SRC/build_wasm/libuv.a" "$LIBUV_OUT"
        cp "$LIBUV_SRC/include/uv.h"       "$INCLUDE_DIR/uv.h"
        cp -r "$LIBUV_SRC/include/uv"      "$INCLUDE_DIR/uv"
        cd - > /dev/null
        rm -rf "$LIBUV_SRC"
        echo "✓ libuv built for wasm"
    else
        echo "✓ libuv wasm already exists, skipping"
    fi

# ─── libwebp (wasm) ───────────────────────────────────────────────────────────
    WEBP_OUT="$LIB_DIR/libwebp_wasm.a"
    if $FORCE_REBUILD || [[ ! -f "$WEBP_OUT" ]]; then
        echo "Building libwebp for wasm..."
        WEBP_SRC="/tmp/libwebp_wasm_build"
        rm -rf "$WEBP_SRC"
        git clone https://chromium.googlesource.com/webm/libwebp "$WEBP_SRC" --depth=1

        mkdir -p "$WEBP_SRC/build_wasm"
        cd "$WEBP_SRC/build_wasm"
        emcmake cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_FLAGS="-pthread" \
            -DBUILD_SHARED_LIBS=OFF \
            -DWEBP_BUILD_ANIM_UTILS=OFF \
            -DWEBP_BUILD_CWEBP=OFF \
            -DWEBP_BUILD_DWEBP=OFF \
            -DWEBP_BUILD_EXTRAS=OFF \
            -DWEBP_BUILD_WEBPINFO=OFF \
            -DWEBP_BUILD_WEBPMUX=OFF
        emmake make -j$(nproc)

        cp "$WEBP_SRC/build_wasm/libwebp.a" "$WEBP_OUT"
        mkdir -p "$INCLUDE_DIR/webp"
        cp "$WEBP_SRC/src/webp/"*.h "$INCLUDE_DIR/webp/"
        cd - > /dev/null
        rm -rf "$WEBP_SRC"
        echo "✓ libwebp built for wasm"
    else
        echo "✓ libwebp wasm already exists, skipping"
    fi

# ─── Native-only libraries ────────────────────────────────────────────────────
else
    if $FORCE_REBUILD || [[ ! -f "$LIB_DIR/libbox2d.a" ]]; then
        echo "Building Box2D..."
        REPO_DIR="$INSTALL_DIR/box2d"
        rm -rf "$REPO_DIR"
        git clone https://github.com/erincatto/box2d.git "$REPO_DIR"
        cd "$REPO_DIR" && git checkout v3.1.1
        mkdir build && cd build
        cmake $(cmake_flags) \
              -DBOX2D_SAMPLES=OFF -DBOX2D_BENCHMARKS=OFF -DBOX2D_DOCS=OFF \
              -DBOX2D_PROFILE=OFF -DBOX2D_VALIDATE=OFF -DBOX2D_UNIT_TESTS=OFF \
              -DUSE_SIMD=OFF -DBOX2D_AVX2=OFF ..
        make -j$(nproc) && make install
        cd ..
        [ -d "include/box2d" ] && cp -r include/box2d "$INCLUDE_DIR/" || echo "failed to find includes for box2d"
        cd .. && rm -rf "$REPO_DIR"
        echo "✓ Box2D built successfully"
    else
        echo "✓ Box2D already exists, skipping build"
    fi

    if $FORCE_REBUILD || [[ ! -f "$LIB_DIR/libfreetype.a" ]]; then
        echo "Building FreeType..."
        FREETYPE_DIR="$INSTALL_DIR/freetype"
        rm -rf "$FREETYPE_DIR" freetype-2.13.2.tar.xz
        wget -O freetype-2.13.2.tar.xz https://download.savannah.gnu.org/releases/freetype/freetype-2.13.2.tar.xz
        tar -xf freetype-2.13.2.tar.xz
        mv freetype-2.13.2 "$FREETYPE_DIR"
        cd "$FREETYPE_DIR" && mkdir build && cd build
        cmake $(cmake_flags) \
              -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
              -DFT_DISABLE_HARFBUZZ=ON -DFT_DISABLE_BROTLI=ON \
              -DFT_DISABLE_PNG=OFF -DFT_DISABLE_ZLIB=OFF \
              -DFT_DISABLE_BZIP2=ON -DBUILD_SHARED_LIBS=OFF ..
        make -j$(nproc) && make install
        cd ../.. && rm -f freetype-2.13.2.tar.xz
        echo "✓ FreeType built successfully"
    else
        echo "✓ FreeType already exists, skipping build"
    fi

    if $FORCE_REBUILD || [[ ! -f "$LIB_DIR/liblunasvg.a" ]]; then
        echo "Building LunaSVG..."
        LUNASVG_DIR="$INSTALL_DIR/lunasvg"
        rm -rf "$LUNASVG_DIR" lunasvg-2.4.1.tar.gz
        wget -O lunasvg-2.4.1.tar.gz https://github.com/sammycage/lunasvg/archive/refs/tags/v2.4.1.tar.gz
        tar -xf lunasvg-2.4.1.tar.gz
        mv lunasvg-2.4.1 "$LUNASVG_DIR"
        cd "$LUNASVG_DIR" && mkdir build && cd build
        cmake $(cmake_flags) \
              -DBUILD_SHARED_LIBS=OFF -DLUNASVG_BUILD_EXAMPLES=OFF ..
        make -j$(nproc) && make install
        cd ../.. && rm -rf "$LUNASVG_DIR" lunasvg-2.4.1.tar.gz
        echo "✓ LunaSVG built successfully"
    else
        echo "✓ LunaSVG already exists, skipping build"
    fi
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "All dependencies processed successfully!"

if $WASM_BUILD; then
    echo "Wasm libraries (in $LIB_DIR):"
    check "$LIB_DIR/libuv_wasm.a"        "libuv_wasm.a"
    check "$LIB_DIR/libwebp_wasm.a"      "libwebp_wasm.a"
    check "$INCLUDE_DIR/glad/gl.h"       "glad (wasm stub)"
    check "$INCLUDE_DIR/BLL"             "BLL"
    check "$INCLUDE_DIR/WITCH"           "WITCH"
    check "$INCLUDE_DIR/BCOL"            "BCOL"
    check "$INCLUDE_DIR/BVEC"            "BVEC"
    check "$INCLUDE_DIR/BDBT"            "BDBT"
else
    echo "Native libraries:"
    check "$INCLUDE_DIR/glad/gl_native.h" "glad"
    check "$LIB_DIR/libfreetype.a"        "FreeType"
    check "$LIB_DIR/libbox2d.a"           "Box2D"
    check "$LIB_DIR/liblunasvg.a"         "LunaSVG"
    check "$INCLUDE_DIR/BLL"              "BLL"
    check "$INCLUDE_DIR/WITCH"            "WITCH"
fi

echo ""
echo "Usage:"
echo "  ./install.sh                 # Build only missing native libraries"
echo "  ./install.sh --force         # Rebuild all native libraries"
echo "  ./install.sh --wasm          # Build wasm libraries (requires emcc in PATH)"
echo "  ./install.sh --wasm --force  # Force rebuild wasm libraries"