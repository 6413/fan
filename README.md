# fan

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

2D graphics library using OpenGL 3.3, (Vulkan Alpha).

## Features

- **Shapes:** lines, circles, rectangles, sprites, etc.
- **Depth sorting**
- **Batch rendering**
- **Lighting**
- **GUI**
- **Collisions**:
  - Continuous collision detection
  - Contact events and sensors
  - Convex polygons, capsules, circles, rounded polygons, segments, and chains
  - Multiple shapes per body
  - Collision filtering
  - Ray casts, shape casts, and overlap queries
- **Image formats:**
  - JPEG (baseline & progressive)
  - PNG (1/2/4/8/16-bit-per-channel)
  - TGA (a subset)
  - BMP (non-1bpp, non-RLE)
  - PSD (composited view only, no extra channels, 8/16 bit-per-channel)
  - GIF (always reports as 4-channel)
  - HDR (radiance rgbE format)
  - PIC (Softimage PIC)
  - PNM (PPM and PGM binary only)
  - WebP
- **Audio**: Currently only [SAC](https://github.com/7244/SAC-container) is supported.
## Installation

- **Requirements:**
  - C++23 (modules)
  - Clang >= 20
  - CMake >= 2.8.*
  - Ninja >= 1.11.1

### Linux
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Install required dependencies:
   
    ```sudo ./apt_install.sh``` (optional)
   
    ```./install.sh```
    
    (```./uninstall.sh```) to remove repos from /usr/local/include/*
3. Main usage:
  -  To compile main use `./compile_main.sh -DMAIN="examples/engine_demos/engine_demo.cpp"`, which also compiles fan

### Compiling fan as a library:
-  Using `-DBUILD_FAN_LIBRARY=` can be set to either `STATIC` or `SHARED`

### Windows
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Code compiles using Visual Studio, external libraries are to be installed.

## Basic Usage Examples

- Hello world rectangle:
    ```cpp
    // Creates graphics engine that opens a window and draws:
    // red rectangle at the position (400, 400), size 200x200 in pixels.
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
  // Creates graphics engine that opens a window and draws:
  // red text at the top-left of window (0, 0) and
  // green text at the bottom-right using immediate-mode GUI.
  import fan;
  
  int main() {
    fan::graphics::engine_t engine;
  
    engine.loop([]{
      fan::graphics::gui::text("top left", fan::vec2(0, 0), fan::colors::red);
      fan::graphics::gui::text_bottom_right("bottom right", fan::colors::green);
    });
  }
  ```

## Demos

[Engine demo](examples/engine_demos/engine_demo.cpp)
![image](https://github.com/user-attachments/assets/32d41e02-09a0-4202-a932-f06f4f319620)

Game
![image](https://github.com/6413/fan/assets/56801084/973f2fa6-fcd7-4b6a-b66b-b92eefae9bba)

Console
![image](https://github.com/user-attachments/assets/0e157c6c-7243-4828-8a94-4efe6397ebd3)

![output](https://github.com/user-attachments/assets/fdb8ae2e-0c76-49bf-9ee4-088a1a582945)

[tilemap editor](examples/graphics/gui/tilemap_editor.cpp)
![image](https://github.com/user-attachments/assets/623dacea-93b6-46e3-bd47-0a4b916c9aa9)
Older versions
![image](https://github.com/user-attachments/assets/3d1b82d1-63d2-40d5-b3da-07821232ee0d)
![image_2023-11-11_20-32-21](https://github.com/6413/fan/assets/56801084/b41e7417-04fb-4d7f-be6a-2e13379cf521)

Particles editor
![output](https://github.com/user-attachments/assets/1b323519-50e4-4dd0-9012-b6b1fc18df40)
![image](https://github.com/user-attachments/assets/b8b19b22-ea8c-4ab3-a0f6-0c2c00f36128)




## License

This project is licensed under the MIT License, however some external libraries might have different license - see the [LICENSE](LICENSE) file for details.
