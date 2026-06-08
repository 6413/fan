#!/bin/bash
packages=(
	clang-20
	cmake # >= 3.31.1 required
	libwebp-dev
	llvm
	libfmt-dev
	libglfw3-dev
	libopus-dev
	libx11-dev
	libgtk-3-dev
	libassimp-dev
	libpulse-dev
	libopus0
	libopus-dev
	libuv1
	libuv1-dev
	ninja-build
	libglew-dev
	libshaderc-dev
)

for package in "${packages[@]}"; do
    apt install -y "$package" || echo "Failed to install $package, skipping..."
done

echo "\n\n"

# Verify cmake version
cmake_version=$(cmake --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+')
if [[ "$cmake_version" < "3.31.1" ]]; then
    echo "CMake version $cmake_version is too old (need >= 3.31.1)"
fi

# Verify ninja version
ninja_version=$(ninja --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+')
if [[ "$ninja_version" < "1.12" ]]; then
    echo "Ninja version $ninja_version is too old (need >= 1.12)"
fi