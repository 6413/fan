# fan

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

2D graphics library using OpenGL, designed to support various shapes, lighting, GUI elements, collisions, and particle effects.

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
- **Audio**: Currently only [SAC](https://github.com/7244/SAC-container) is supported.

## Limitations
- Limited shapes for collisions
- The library is still in development, bugs and missing features to be expected

## Installation

Requires c++20

### Windows
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Using Visual Studio use fan.sln. Alternatively compile using clang++ (currently not working for windows)

### Linux
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Install required dependencies
    ```
    sudo ./install.sh
    ```
    (```./uninstall.sh```) to remove repos from /usr/local/include/*
3. Main usage:
-  To compile fan and it's libs to an object file use `./compile_fan.sh` you can set thread amount for compile using -jN
-  For linux you can export files to /usr/local/lib, by doing `sudo ./copy_to_default_paths.sh`
-  To compile main use `./compile_main.sh -DMAIN="examples/graphics/2D/shapes/rectangle.cpp`


   **Optionally you can compile fan with main file like so (good for if fan source changes):**
   -  To compile libs `./compile_all_libs.sh` you can set thread amount for compile using -tN
   -  For linux you can export files to /usr/local/lib, by doing `sudo ./copy_to_default_paths.sh`
   -  To compile main `./compile_main.sh -DBUILD_FAN=1 -DMAIN="examples/graphics/2D/shapes/rectangle.cpp"`

## Usage

- Hello world rectangle:
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
- Text rendering:
  ```cpp
  #include <fan/pch.h>
  
  int main() {
    loco_t loco;
  
    loco.loop([&] {
      fan::graphics::text("top left", fan::vec2(0, 0), fan::colors::red);
      fan::graphics::text_bottom_right("bottom right", fan::colors::green);
    });
  
    return 0;
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

This project is licensed under the MIT License, however some external libraries might have different license - see the [LICENSE](LICENSE) file for details.
