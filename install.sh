#!/bin/bash

INSTALL_DIR="$(pwd)/thirdparty/fan"
INCLUDE_DIR="$INSTALL_DIR/include"
LIB_DIR="$INSTALL_DIR/lib"

mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

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

move_and_pull "https://github.com/7244/WITCH.git" "WITCH"
move_and_pull "https://github.com/7244/BCOL.git" "BCOL"
move_and_pull "https://github.com/7244/BLL.git" "BLL"
move_and_pull "https://github.com/7244/BVEC.git" "BVEC"
move_and_pull "https://github.com/7244/BDBT.git" "BDBT"
move_and_pull "https://github.com/7244/bcontainer.git" "bcontainer"
move_and_pull "https://github.com/7244/pixfconv.git" "pixfconv"

REPO_DIR="$INSTALL_DIR/box2d"
git clone https://github.com/erincatto/box2d.git "$REPO_DIR"
cd "$REPO_DIR"
git checkout v3.0.0
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