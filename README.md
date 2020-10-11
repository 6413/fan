# OpenGL
2D/3D stuff using C++ OpenGL

Installation and compilation

Windows

Compiled using Visual Studio 2019 (project file is in the directory)

For linux g++ and clang minimum version is 10

Ubuntu/Debian

sudo apt install libglew-dev -y &&
sudo apt install libglfw3-dev -y &&
sudo apt install libassimp-dev -y &&
sudo apt install libfreetype6-dev -y && 
sudo apt install libopenal-dev -y &&
make

Arch linux

sudo pacman -S glfw-x11 -y &&
sudo pacman -S assimp -y &&
sudo pacman -S openal -y &&
make

g++-10 is required otherwise don't use make file and use with g++-10 ...
sudo ln -s /usr/bin/g++-10 /usr/local/bin/g++
