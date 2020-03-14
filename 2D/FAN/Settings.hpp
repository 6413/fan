#pragma once
#include <GLFW/glfw3.h>
#include <FAN/Vectors.hpp>

void GetFps(bool print = true);

constexpr auto WINDOWSIZE = _vec2<int>(1000, 800);
extern float delta_time;
static constexpr int block_size = 64;
extern GLFWwindow* window;
static constexpr _vec2<int> view(WINDOWSIZE.x / block_size, WINDOWSIZE.y / block_size);
constexpr auto grid_size = _vec2<int>(WINDOWSIZE.x / block_size, WINDOWSIZE.y / block_size);