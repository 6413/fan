#!/bin/bash
INSTALL_DIR="$(pwd)/third_party/fan"
INCLUDE_DIR="$INSTALL_DIR/include"
LIB_DIR="$INSTALL_DIR/lib"
mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

FORCE_REBUILD=false
WASM_BUILD=false
CORE_ONLY=false

for arg in "$@"; do
  case $arg in
    --force) FORCE_REBUILD=true ;;
    --wasm)  WASM_BUILD=true ;;
    --core)  CORE_ONLY=true ;;
  esac
done

if $WASM_BUILD; then
  command -v emcc >/dev/null 2>&1 || { echo "emcc not found."; exit 1; }
  LIB_DIR="$INSTALL_DIR/lib/wasm"
  mkdir -p "$LIB_DIR"
fi

move_and_pull() {
  local url=$1 name=$2
  local repo="$INSTALL_DIR/repos/$name"
  local target="$INCLUDE_DIR/$name"

  if [[ -d "$target" && "$FORCE_REBUILD" == false ]]; then
    return 0
  fi

  mkdir -p "$INSTALL_DIR/repos"
  if [[ -d "$repo/.git" ]] && ! git -C "$repo" status >/dev/null 2>&1; then
    rm -rf "$repo"
  fi

  if [[ -d "$repo/.git" ]]; then
    git -C "$repo" pull --quiet || exit 1
  else
    echo "Cloning $name..."
    git -c http.version=HTTP/1.1 clone --depth 1 "$url" "$repo" --quiet || exit 1
  fi

  rm -rf "$target" && mkdir -p "$target"
  if [[ -d "$repo/$name" ]]; then
    cp -r "$repo/$name"/* "$target/"
  else
    find "$repo" -maxdepth 1 -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" \) -exec cp {} "$target/" \; 2>/dev/null || true
    find "$repo" -maxdepth 1 -type d ! -name ".git" ! -path "$repo" -exec cp -r {} "$target/" \; 2>/dev/null || true
  fi
}

install_vma() {
  local repo="$INSTALL_DIR/repos/VulkanMemoryAllocator"
  if [[ -f "$INCLUDE_DIR/vk_mem_alloc.h" && "$FORCE_REBUILD" == false ]]; then
    return 0
  fi
  mkdir -p "$INSTALL_DIR/repos"
  if [[ -d "$repo/.git" ]] && ! git -C "$repo" status >/dev/null 2>&1; then
    rm -rf "$repo"
  fi
  if [[ -d "$repo/.git" ]]; then
    git -C "$repo" pull --quiet || exit 1
  else
    echo "Cloning VulkanMemoryAllocator..."
    git -c http.version=HTTP/1.1 clone --depth 1 "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git" "$repo" --quiet || exit 1
  fi
  cp "$repo/include/vk_mem_alloc.h" "$INCLUDE_DIR/vk_mem_alloc.h"
  mkdir -p "$INCLUDE_DIR/VulkanMemoryAllocator/include"
  cp "$repo/include/vk_mem_alloc.h" "$INCLUDE_DIR/VulkanMemoryAllocator/include/vk_mem_alloc.h"
}

install_vulkan_utility_libraries() {
  local repo="$INSTALL_DIR/repos/Vulkan-Utility-Libraries"
  if [[ -f "$INCLUDE_DIR/vulkan/vk_enum_string_helper.h" && "$FORCE_REBUILD" == false ]]; then
    return 0
  fi
  mkdir -p "$INSTALL_DIR/repos"
  rm -rf "$repo"
  echo "Cloning Vulkan-Utility-Libraries..."
  git -c http.version=HTTP/1.1 clone --depth 1 --branch vulkan-sdk-1.4.335.0 "https://github.com/KhronosGroup/Vulkan-Utility-Libraries.git" "$repo" --quiet || exit 1
  mkdir -p "$INCLUDE_DIR/vulkan"
  cp "$repo/include/vulkan/"*.h "$INCLUDE_DIR/vulkan/"
}

if ! $CORE_ONLY; then
  move_and_pull "https://github.com/7244/WITCH.git"      "WITCH"
  move_and_pull "https://github.com/7244/BCOL.git"       "BCOL"
  move_and_pull "https://github.com/7244/BLL.git"        "BLL"
  move_and_pull "https://github.com/7244/BVEC.git"       "BVEC"
  move_and_pull "https://github.com/7244/BDBT.git"       "BDBT"
  move_and_pull "https://github.com/7244/bcontainer.git" "bcontainer"
  move_and_pull "https://github.com/7244/pixfconv.git"   "pixfconv"
  move_and_pull "https://github.com/6413/PIXF.git"       "PIXF"
  install_vma
  install_vulkan_utility_libraries

  touch "$INSTALL_DIR/.gfx.stamp"
fi

if $WASM_BUILD; then
  LIBUV_OUT="$LIB_DIR/libuv_wasm.a"
  if $FORCE_REBUILD || [[ ! -f "$LIBUV_OUT" ]]; then
    echo "Building libuv for wasm..."
    LIBUV_SRC="/tmp/libuv_wasm_build"
    rm -rf "$LIBUV_SRC"
    git -c http.version=HTTP/1.1 clone https://github.com/libuv/libuv.git "$LIBUV_SRC" --depth=1 --quiet
    sed -i 's/return UV__ERR(pthread_setname_np(pthread_self(), namebuf));/return 0;/' "$LIBUV_SRC/src/unix/thread.c"
    sed -i 's/r = pthread_getname_np(\*tid, thread_name, sizeof(thread_name));/r = 0; thread_name[0] = 0;/' "$LIBUV_SRC/src/unix/thread.c"
    sed -i 's/#if defined(__linux__)/#if defined(__EMSCRIPTEN__)\n# include "uv\/posix.h"\n#elif defined(__linux__)/' "$LIBUV_SRC/include/uv/unix.h"
    cat << 'EOF' >> "$LIBUV_SRC/CMakeLists.txt"

if(TARGET uv_a)
  target_sources(uv_a PRIVATE src/unix/posix-hrtime.c src/unix/posix-poll.c src/unix/no-fsevents.c src/unix/no-proctitle.c)
endif()
EOF
    mkdir -p "$LIBUV_SRC/build_wasm"
    cd "$LIBUV_SRC/build_wasm"
    emcmake cmake .. --log-level=WARNING -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-pthread" -DLIBUV_BUILD_SHARED=OFF -DBUILD_TESTING=OFF
    emmake make -j$(nproc)
    cp "$LIBUV_SRC/build_wasm/libuv.a" "$LIBUV_OUT"
    cp "$LIBUV_SRC/include/uv.h"       "$INCLUDE_DIR/uv.h"
    cp -r "$LIBUV_SRC/include/uv"      "$INCLUDE_DIR/uv"
    cd - >/dev/null && rm -rf "$LIBUV_SRC"
  fi

  if ! $CORE_ONLY; then
    WEBP_OUT="$LIB_DIR/libwebp_wasm.a"
    if $FORCE_REBUILD || [[ ! -f "$WEBP_OUT" ]]; then
      echo "Building libwebp for wasm..."
      WEBP_SRC="/tmp/libwebp_wasm_build"
      rm -rf "$WEBP_SRC"
      git -c http.version=HTTP/1.1 clone https://chromium.googlesource.com/webm/libwebp "$WEBP_SRC" --depth=1 --quiet
      mkdir -p "$WEBP_SRC/build_wasm"
      cd "$WEBP_SRC/build_wasm"
      emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-pthread" -DBUILD_SHARED_LIBS=OFF -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_EXTRAS=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF
      emmake make -j$(nproc)
      cp "$WEBP_SRC/build_wasm/libwebp.a" "$WEBP_OUT"
      mkdir -p "$INCLUDE_DIR/webp"
      cp "$WEBP_SRC/src/webp/"*.h "$INCLUDE_DIR/webp/"
      cd - >/dev/null && rm -rf "$WEBP_SRC"
    fi
  fi
fi

touch "$INSTALL_DIR/.core.stamp"

echo ""
echo "All dependencies processed successfully."