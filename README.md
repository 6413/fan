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
  - xmake >= v3.0.8 (tested)
  - Ninja >= 1.11.1

### Linux
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Install required dependencies:
   
    ```sudo ./apt_install.sh``` (optional)
   
    ```./install.sh```
    
    (```./uninstall.sh```) to remove repos from /usr/local/include/*
3. Main usage:
  -  To compile main use `./xcompile_main.sh`, which also compiles fan

### Windows
1. Clone the repository: `git clone https://github.com/6413/fan.git`
2. Code compiles using Visual Studio, external libraries are to be installed.

## Basic Usage Examples

- Hello World Rectangle:
  ```cpp
  import fan;

  int main() {
    fan::graphics::engine_t engine;
    
    // args: position(x,y,z), half_size(w,h), color
    fan::graphics::rectangle_t rect{
      fan::vec3(400.f, 400.f, 0.f), 
      fan::vec2(200.f), 
      fan::colors::red
    };
    
    engine.loop([&] {
      // per-frame logic
    });
  }
  ```

- Input & Immediate Mode Rendering:
  ```cpp
  import fan;

  int main() {
    fan::graphics::engine_t engine;
    fan::vec2 pos = engine.viewport_get_size() / 2.f;
  
    engine.loop([&] {
      pos += engine.get_input_vector(400.f) * engine.get_delta_time();
      
      // immediate shapes auto-manage lifetime and draw for one frame
      // args: position(x, y, z), radius, color
      fan::graphics::circle(pos, 30.f, fan::colors::aqua);
    });
  }
  ```

- 2D Physics Synchronization
  ```cpp
  import fan;

  int main() {
    fan::graphics::engine_t engine;
    engine.update_physics(true);
    auto& physics_ctx = engine.get_physics_context();
    
    // static ground body
    fan::physics::body_id_t ground_body_id = physics_ctx.create_rectangle(
      fan::vec2(400.f, 700.f), 
      fan::vec2(400.f, 20.f)
    );
  
    // visual rectangle natively synced to a dynamic box2d body
    fan::graphics::physics::rectangle_t box{{
      .position = fan::vec3(400.f, 100.f, 0.f),
      .size = fan::vec2(30.f),
      .color = fan::colors::orange,
      .body_type = fan::physics::body_type_e::dynamic_body
    }};
    
    engine.loop([&] {
      if (engine.is_mouse_clicked()) {
        box.apply_linear_impulse_center(fan::vec2(0.f, -800.f));
      }
    });
  }
  ```

- UI & Camera Tracking
  ```cpp
  import fan;

  int main() {
    fan::graphics::engine_t engine;
    fan::graphics::sprite_t player{
      fan::vec3(0.f), 
      fan::vec2(32.f), 
      "player.png"
    };
    
    engine.loop([&] {
      if (auto gui_wnd = fan::graphics::gui::window("Settings")) {
        fan::graphics::gui::text("Use WASD to move.");
      }
      
      fan::vec2 player_pos = player.get_position();
      fan::vec2 new_player_pos = player_pos + engine.get_input_vector(300.f) * engine.get_delta_time();
      player.set_position(new_player_pos);
      // args: pos, dt
      engine.camera_set_target(new_player_pos, 5.f);
    });
  }
  ```

## Exporting

Use `export.py` to collect all assets and shaders needed by your exe into a folder:
```
python export.py your_exe.exe export_folder
```

On Windows you can also just drag an exe onto `export.bat`. On Linux run `./export.sh` (run `chmod +x export.sh` once to make it executable).

If the output folder isn't empty it will ask to clear it first, or pass `--force` to skip the prompt.

## Demos

[Engine demo](examples/engine_demos/engine_demo.cpp)
![image](https://github.com/user-attachments/assets/f37fbf05-5f37-426a-add7-cedfeb869b62)

[Scenery](examples/graphics/random/scenery/main.cpp)
![image](https://github.com/user-attachments/assets/083bf9d6-396e-4d5e-9a5b-e73623984376)

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
![image](https://github.com/user-attachments/assets/4b7f863b-449b-433e-ba23-244acf28de07)





## License

This project is licensed under the MIT License, however some external libraries might have different license - see the [LICENSE](LICENSE) file for details.
