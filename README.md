# fan

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

2D graphics library using OpenGL 3.3, (Vulkan Alpha).

## Features

- **Shapes:** line, circle, rectangle, sprite, etc...
- **Lighting:** Library supports lighting.
- **GUI: (Dear ImGui)** Uses Dear ImGui for the GUI.
- **Collisions (Box2D)**:
  - Continuous collision detection
  - Contact events and sensors
  - Convex polygons, capsules, circles, rounded polygons, segments, and chains
  - Multiple shapes per body
  - Collision filtering
  - Ray casts, shape casts, and overlap queries
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
## Installation

Requires C++23, Clang-20 >=, CMake 2.8.* >=, Ninja 1.11.1 >=.
Library plans to use modules in future.

### Linux
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Install required dependencies
   
    ```sudo ./apt_install.sh``` (optional)
   
    ```./install.sh```
    
    (```./uninstall.sh```) to remove repos from /usr/local/include/*
4. Main usage:
-  To compile main use `./compile_main.sh -DMAIN="examples/graphics/2D/shapes/rectangle.cpp"`, which also compiles fan

### Windows
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Code compiles using Visual Studio, external libraries are to be installed.

## Basic Usage Examples

- Hello world rectangle:
    ```cpp
    import fan;

    int main() {
      fan::graphics::engine_t engine;
    
      fan::graphics::rectangle_t rect{{
          .position = 400,
          .size = 200,
          .color = fan::colors::red
      }};
      
      engine.loop();
    }
    ```
- Text rendering:
  ```cpp
  import fan;
  
  int main() {
    fan::graphics::engine_t engine;
  
    engine.loop([]{
      fan::graphics::gui::text("top left", fan::vec2(0, 0), fan::colors::red);
      fan::graphics::gui::text_bottom_right("bottom right", fan::colors::green);
    });
  }
  ```
## Examples

Check out the [examples](examples/) directory for sample projects demonstrating various features of the library.

## Demos

A game demo using fan
![image](https://github.com/6413/fan/assets/56801084/973f2fa6-fcd7-4b6a-b66b-b92eefae9bba)

[A program that demonstrates the engine's features](examples/engine_demos/engine_demo.cpp)
![image](https://github.com/user-attachments/assets/3b6af145-c91e-4cfd-89b3-9fa2e93905c8)


Developer console support (F3)
![image](https://github.com/6413/fan/assets/56801084/7556ce24-ba0f-43c6-85d6-b951351bb59c)

[laser shooting mirrors](examples/engine_demos/mirrors.cpp)
![output](https://github.com/user-attachments/assets/fdb8ae2e-0c76-49bf-9ee4-088a1a582945)

[tilemap editor](examples/graphics/gui/tilemap_editor.cpp)
![image](https://github.com/user-attachments/assets/3d1b82d1-63d2-40d5-b3da-07821232ee0d)
![image_2023-11-11_20-32-21](https://github.com/6413/fan/assets/56801084/b41e7417-04fb-4d7f-be6a-2e13379cf521)


[alpha blending](examples/graphics/2D/blending_test.cpp)  
![image](https://github.com/user-attachments/assets/ba4637e6-c102-408e-b043-0d724d02e350)

[quadtree_visualize](examples/graphics/2D/quadtree_visualize.cpp)
![image_2023-11-11_20-24-01](https://github.com/6413/fan/assets/56801084/0aac1cbb-2d41-40ef-b0d0-5ab838b9b3d1)

[particles](examples/graphics/2D/shapes/particles.cpp)
![image_2023-11-11_20-29-09](https://github.com/6413/fan/assets/56801084/8c63a7a0-a8c1-451e-82be-af14aabb69b3)

[sort_visualizer](examples/graphics/2D/sort_visualizer.cpp)
![image_2023-11-11_20-33-48](https://github.com/6413/fan/assets/56801084/a39c3f93-e902-4401-9efe-2ae15e0035ad)

[function_graph](examples/graphics/2D/function_graph.cpp)
![image_2023-11-11_20-36-10](https://github.com/6413/fan/assets/56801084/c69cf128-b1be-4c2d-8ef2-50d7281ddf07)


## License

This project is licensed under the MIT License, however some external libraries might have different license - see the [LICENSE](LICENSE) file for details.
