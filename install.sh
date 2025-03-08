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

git clone https://github.com/7244/WITCH.git
mv WITCH /usr/local/include/WITCH

git clone https://github.com/7244/BCOL.git
mv BCOL /usr/local/include/BCOL

git clone https://github.com/7244/BLL.git
mv BLL /usr/local/include/BLL

git clone https://github.com/7244/BVEC.git
mv BVEC /usr/local/include/BVEC

git clone https://github.com/7244/BDBT.git
mv BDBT /usr/local/include/BDBT

git clone https://github.com/7244/bcontainer.git
mv bcontainer /usr/local/include/bcontainer

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