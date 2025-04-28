#!/bin/bash

packages=(
	clang-17
	make # 2.8.* >= required
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

#if one package fails continue other
for package in "${packages[@]}"; do
	apt install -y "$package" || echo "failed to install $package, skipping..."
done