# fan

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A powerful and flexible 2D graphics library for OpenGL, designed to support various shapes, lighting, GUI elements, collisions, and particle effects.

## Features

- **Shapes:** Draw lines, rectangles, and sprites with ease.
- **Lighting:** Illuminate your scenes with customizable lighting effects.
- **GUI:** Easily integrate graphical user interfaces into your applications.
- **Collisions** Simple collisions Circle<->Circle, Rectangle<->Circle

## Limitations
- Uses .webp files for loading images
- Limited shapes for collisions

## Getting Started

## Installation

Requires c++20

### Windows
1. Clone the repository: `git clone https://github.com/614/fan.git`
2. Using Visual Studio use fan.sln. Alternatively compile using clang++/g++ using make_imgui, make_pch, make, in order

### Linux
1. Install required dependencies libx11-dev, libxrandr-dev, libwebp-dev, libxcursor-dev, llvm, clang/gcc, libopus-dev, OpenGL
2. Clone the repository: `git clone https://github.com/614/fan.git`
3. Setup and compile using make_imgui, make_pch, make, in order

## Notes for compiling with Makefile
Using the library from outside the project, change ROOT_PATH in Makefiles

### Usage

1. Hello world rectangle
    ```cpp
    #include fan_pch

    int main() {
      loco_t loco;
    
      fan::graphics::rectangle_t rect{{
          .position = 400,
          .size = 200,
          .color = fan::colors::red
      }};
      
      loco.loop([&] {
    
      });
    }
    ```

## Examples

Check out the [examples](examples/) directory for sample projects demonstrating various features of the library.

## Demos

#### Game demo using fan
![image_2023-11-11_20-56-49](https://github.com/6413/fan/assets/56801084/81409f6f-8be4-4754-bcef-dc3e8691097f)


[quadtree_visualize](examples/graphics/quadtree_visualize.cpp)
![image_2023-11-11_20-24-01](https://github.com/6413/fan/assets/56801084/0aac1cbb-2d41-40ef-b0d0-5ab838b9b3d1)

[particles](examples/graphics/particles.cpp)
![image_2023-11-11_20-29-09](https://github.com/6413/fan/assets/56801084/8c63a7a0-a8c1-451e-82be-af14aabb69b3)

[tilemap_editor](examples/gui/tilemap_editor.cpp)
![image_2023-11-11_20-32-21](https://github.com/6413/fan/assets/56801084/b41e7417-04fb-4d7f-be6a-2e13379cf521)

[sort_visualizer](examples/graphics/sort_visualizer.cpp)
![image_2023-11-11_20-33-48](https://github.com/6413/fan/assets/56801084/a39c3f93-e902-4401-9efe-2ae15e0035ad)

[function_graph](examples/graphics/function_graph.cpp)
![image_2023-11-11_20-36-10](https://github.com/6413/fan/assets/56801084/c69cf128-b1be-4c2d-8ef2-50d7281ddf07)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
