#!/bin/bash
./uninstall.sh

apt install -y clang \
				 make \
				 libwebp-dev \
				 llvm \
				 libfmt-dev \
				 libglfw3-dev \
				 libopus-dev \
				 libx11-dev \
				 libgtk-3-dev \
				 libassimp-dev \
				 pulseaudio \
				 libpulse-dev \
				 libopus0 \
				 libopus-dev \
				 libuv1 \
				 libuv1-dev \
				 ninja-build \
				 libglew-dev

move_and_pull() {
	REPO_URL=$1
	DIR_NAME=$2
	
	git clone "$REPO_URL" "$DIR_NAME"
	
	if [ -d "/usr/local/include/$DIR_NAME" ]; then
		rm -rf "/usr/local/include/$DIR_NAME"
	fi
	if [ ! -L "/usr/local/include/$DIR_NAME" ]; then
		mv "$DIR_NAME" "/usr/local/include/$DIR_NAME"
	else
		cd "/usr/local/include/$DIR_NAME"
		git pull
		if [ $? -ne 0 ]; then
			echo "git pull failed for $DIR_NAME"
			exit 1
		else
			echo "git pull succeeded for $DIR_NAME"
		fi
	fi
}

move_and_pull "https://github.com/7244/WITCH.git" "WITCH"
move_and_pull "https://github.com/7244/BCOL.git" "BCOL"
move_and_pull "https://github.com/7244/BLL.git" "BLL"
move_and_pull "https://github.com/7244/BVEC.git" "BVEC"
move_and_pull "https://github.com/7244/BDBT.git" "BDBT"
move_and_pull "https://github.com/7244/bcontainer.git" "bcontainer"

: '
FOR INSTALLING BOX2D
git clone https://github.com/erincatto/box2d.git
cd box2d
git checkout v3.0.0
mkdir build
cd build
cmake -DBOX2D_SAMPLES=OFF -DBOX2D_BENCHMARKS=OFF -DBOX2D_DOCS=OFF -DBOX2D_PROFILE=OFF -DBOX2D_VALIDATE=OFF -DBOX2D_UNIT_TESTS=OFF -DUSE_SIMD=OFF -DBOX2D_AVX2=OFF ..
make
sudo make install
'