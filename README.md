# 2D
2D/3D stuff using C++ OpenGL GLFW

Installation and compilation

Windows

Compiled using Visual Studio 2019 (project file is in the directory)

For linux g++ and clang minimum version is 8

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
