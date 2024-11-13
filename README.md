# fan

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

2D graphics library for OpenGL, designed to support various shapes, lighting, GUI elements, collisions, and particle effects.

## Features

- **Shapes:** Draw lines, rectangles, and sprite.
- **Lighting:** Illuminate your scenes with customizable lighting effects.
- **GUI:** Easily integrate graphical user interfaces into your applications.
- **Collisions:** Simple collisions Circle<->Circle, Rectangle<->Circle (no rotation for rectangles)
- **Image formats:**
  JPEG (baseline & progressive),
  PNG (1/2/4/8/16-bit-per-channel),
  TGA (a subset),
  BMP (non-1bpp, non-RLE),
  PSD (composited view only, no extra channels, 8/16 bit-per-channel),
  GIF (always reports as 4-channel),
  HDR (radiance rgbE format),
  PIC (Softimage PIC),
  PNM (PPM and PGM binary only),
  WebP

## Limitations
- Limited shapes for collisions
- The library is still in development, bugs and missing features to be expected

## Getting Started

## Installation

Requires c++20

### Windows
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Using Visual Studio use fan.sln. Alternatively compile using clang++/g++ using make_imgui, make_pch, make, in order

### Linux
1. Install required dependencies libx11-dev, libxrandr-dev, libwebp-dev, libxcursor-dev, llvm, clang/gcc, libopus-dev, OpenGL, GLFW
 ```
   sudo apt install clang -y &&
   sudo apt install make -y &&
   sudo apt install libwebp-dev -y &&
   sudo apt install llvm -y &&
   sudo apt install clang -y &&
   sudo apt install libfmt-dev -y &&
   sudo apt install libglfw3-dev -y &&
   sudo apt install libopus-dev -y &&
   sudo apt install libx11-dev -y
   ```
3. Clone the repository: `git clone https://github.com/6413/fan.git`
4. To compile libs `./compile_all_libs.sh` you can set thread amount for compile using -tN
5. To compile main `./compile_main.sh -DMAIN="examples/graphics/2D/shapes/rectangle.cpp"`

### Usage

1. Hello world rectangle
    ```cpp
    #include <fan/pch.h>

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
![image](https://github.com/6413/fan/assets/56801084/973f2fa6-fcd7-4b6a-b66b-b92eefae9bba)

Developer console support (F3)
![image](https://github.com/6413/fan/assets/56801084/7556ce24-ba0f-43c6-85d6-b951351bb59c)


[quadtree_visualize](examples/graphics/2D/quadtree_visualize.cpp)
![image_2023-11-11_20-24-01](https://github.com/6413/fan/assets/56801084/0aac1cbb-2d41-40ef-b0d0-5ab838b9b3d1)

[particles](examples/graphics/2D/shapes/particles.cpp)
![image_2023-11-11_20-29-09](https://github.com/6413/fan/assets/56801084/8c63a7a0-a8c1-451e-82be-af14aabb69b3)

[tilemap_editor](examples/graphics/gui/tilemap_editor.cpp)
![image_2023-11-11_20-32-21](https://github.com/6413/fan/assets/56801084/b41e7417-04fb-4d7f-be6a-2e13379cf521)

[sort_visualizer](examples/graphics/2D/sort_visualizer.cpp)
![image_2023-11-11_20-33-48](https://github.com/6413/fan/assets/56801084/a39c3f93-e902-4401-9efe-2ae15e0035ad)

[function_graph](examples/graphics/2D/function_graph.cpp)
![image_2023-11-11_20-36-10](https://github.com/6413/fan/assets/56801084/c69cf128-b1be-4c2d-8ef2-50d7281ddf07)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
