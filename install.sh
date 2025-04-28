#!/bin/bash

INSTALL_DIR="$(pwd)/thirdparty/fan"
INCLUDE_DIR="$INSTALL_DIR/include"
LIB_DIR="$INSTALL_DIR/lib"

mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

move_and_pull() {
	REPO_URL=$1
	DIR_NAME=$2
	TARGET_DIR="$INCLUDE_DIR/$DIR_NAME"

	git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
	if [ -d "$TARGET_DIR/.git" ]; then
			cd "$TARGET_DIR"
			git pull || { echo "failed to update $DIR_NAME"; exit 1; }
			cd -
	fi
}

move_and_pull "https://github.com/7244/WITCH.git" "WITCH"
move_and_pull "https://github.com/7244/BCOL.git" "BCOL"
move_and_pull "https://github.com/7244/BLL.git" "BLL"
move_and_pull "https://github.com/7244/BVEC.git" "BVEC"
move_and_pull "https://github.com/7244/BDBT.git" "BDBT"
move_and_pull "https://github.com/7244/bcontainer.git" "bcontainer"

REPO_DIR="$INSTALL_DIR/box2d"
git clone https://github.com/erincatto/box2d.git "$REPO_DIR"
cd "$REPO_DIR"
git checkout v3.0.0
mkdir build
cd build
cmake -DBOX2D_SAMPLES=OFF -DBOX2D_BENCHMARKS=OFF -DBOX2D_DOCS=OFF -DBOX2D_PROFILE=OFF -DBOX2D_VALIDATE=OFF -DBOX2D_UNIT_TESTS=OFF -DUSE_SIMD=OFF -DBOX2D_AVX2=OFF -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" ..
make
make install

cd ..

if [ -d "include/box2d" ]; then
	cp -r include/box2d "$INCLUDE_DIR/"
else
	echo "failed to find includes for box2d"
fi

cd ..

rm -rf "$REPO_DIR"