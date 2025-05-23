module;
// extension to loco.h


#if defined(fan_gui)
  #include <fan/imgui/imgui_internal.h>
  #include <fan/graphics/gui/imgui_themes.h>
#endif

#include <fan/types/types.h>
#include <fan/graphics/opengl/init.h>
#include <fan/event/types.h>

#include <fan/math/math.h>

#include <filesystem>

#define loco_vfi

export module fan:graphics;

//import :graphics.opengl.core; // TODO this should not be here
import :graphics.loco;
import :io.directory;
import :io.file;
import :types.color;

// user friendly functions
/***************************************/

export namespace fan {
  namespace window {
    fan::vec2 get_size() {
      return gloco->window.get_size();
    }
    void set_size(const fan::vec2& size) {
      gloco->window.set_size(size);
      gloco->viewport_set(gloco->orthographic_camera.viewport, fan::vec2(0, 0), size, size);
      gloco->camera_set_ortho(
        gloco->orthographic_camera.camera,
        fan::vec2(0, size.x),
        fan::vec2(0, size.y)
      );
      gloco->viewport_set(gloco->perspective_camera.viewport, fan::vec2(0, 0), size, size);
      gloco->camera_set_ortho(
        gloco->perspective_camera.camera,
        fan::vec2(0, size.x),
        fan::vec2(0, size.y)
      );
    }

    bool is_mouse_clicked(int button = fan::mouse_left) {
      return gloco->is_mouse_clicked(button);
    }
    bool is_mouse_down(int button = fan::mouse_left) {
      return gloco->is_mouse_down(button);
    }
    bool is_mouse_released(int button = fan::mouse_left) {
      return gloco->is_mouse_released(button);
    }
    fan::vec2 get_mouse_drag(int button = fan::mouse_left) {
      return gloco->get_mouse_drag(button);
    }

    bool is_key_pressed(int key) {
      return gloco->is_key_pressed(key);
    }
    bool is_key_down(int key) {
      return gloco->is_key_down(key);
    }
    bool is_key_released(int key) {
      return gloco->is_key_released(key);
    }
  }
}

export namespace fan {
  namespace window {
    fan::vec2 get_mouse_position() {
      return gloco->get_mouse_position();
    }
  }
}

export namespace fan {
  namespace graphics {
    using vfi_t = loco_t::vfi_t;

    using engine_t = loco_t;
    using image_t = loco_t::image_t;
    using camera_impl_t = loco_t::camera_impl_t;
    using camera_t = camera_impl_t;
    using viewport_t = loco_t::viewport_t;

    fan::graphics::image_t invalid_image = []{
      image_t image;
      image.sic();
      return image;
    }();
    void add_input_action(const int* keys, std::size_t count, std::string_view action_name) {
      gloco->input_action.add(keys, count, action_name);
    }
    void add_input_action(std::initializer_list<int> keys, std::string_view action_name) {
      gloco->input_action.add(keys, action_name);
    }
    void add_input_action(int key, std::string_view action_name) {
      gloco->input_action.add(key, action_name);
    }
    bool is_input_action_active(std::string_view action_name, int pstate = loco_t::input_action_t::press) {
      return gloco->input_action.is_active(action_name);
    }

    fan::vec2 get_mouse_position() {
      return gloco->get_mouse_position();
    }
    fan::vec2 get_mouse_position(const fan::graphics::camera_t& camera) {
      return loco_t::transform_position(gloco->get_mouse_position(), camera.viewport, camera.camera);
    }
    fan::vec2 get_mouse_position(
      const loco_t::camera_t& camera,
      const loco_t::viewport_t& viewport
    ) {
      return gloco->get_mouse_position(camera, viewport);
    }

#if defined(fan_gui)

    void text_partial_render(const std::string& text, size_t render_pos, f32_t wrap_width, f32_t line_spacing = 0) {
      static auto find_next_word = [](const std::string& str, std::size_t offset) -> std::size_t {
        std::size_t found = str.find(' ', offset);
        if (found == std::string::npos) {
          found = str.size();
        }
        if (found != std::string::npos) {
        }
        return found;
        };
      static auto find_previous_word = [](const std::string& str, std::size_t offset) -> std::size_t {
        std::size_t found = str.rfind(' ', offset);
        if (found == std::string::npos) {
          found = str.size();
        }
        if (found != std::string::npos) {
        }
        return found;
        };

      std::vector<std::string> lines;
      std::size_t previous_word = 0;
      std::size_t previous_push = 0;
      bool found = false;
      for (std::size_t i = 0; i < text.size(); ++i) {
        std::size_t word_index = text.find(' ', i);
        if (word_index == std::string::npos) {
          word_index = text.size();
        }

        std::string str = text.substr(previous_push, word_index - previous_push);
        f32_t width = ImGui::CalcTextSize(str.c_str()).x;

        if (width >= wrap_width) {
          if (previous_push == previous_word) {
            lines.push_back(text.substr(previous_push, i - previous_push));
            previous_push = i;
          }
          else {
            lines.push_back(text.substr(previous_push, previous_word - previous_push));
            previous_push = previous_word + 1;
            i = previous_word;
          }
        }
        previous_word = word_index;
        i = word_index;
      }

      // Add remaining text as last line
      if (previous_push < text.size()) {
        lines.push_back(text.substr(previous_push));
      }

      std::size_t empty_lines = 0;
      std::size_t character_offset = 0;
      ImVec2 pos = ImGui::GetCursorScreenPos();
      for (const auto& line : lines) {
        if (line.empty()) {
          ++empty_lines;
          continue;
        }
        std::size_t empty = 0;
        if (empty >= line.size()) {
          break;
        }
        while (line[empty] == ' ') {
          if (empty >= line.size()) {
            break;
          }
          ++empty;
        }
        if (character_offset >= render_pos) {
          break;
        }
        std::string render_text = line.substr(empty).c_str();
        ImGui::SetCursorScreenPos(pos);
        if (character_offset + render_text.size() >= render_pos) {
          ImGui::TextUnformatted(render_text.substr(0, render_pos - character_offset).c_str());
          break;
        }
        else {
          ImGui::TextUnformatted(render_text.c_str());
          if (render_text.back() != ' ') {
            character_offset += 1;
          }
          character_offset += render_text.size();
          pos.y += ImGui::GetTextLineHeightWithSpacing() + line_spacing;
        }
      }
      if (empty_lines) {
        ImGui::TextColored(fan::colors::red, "warning empty lines:%zu", empty_lines);
      }
    }
#endif
  }
}


export namespace fan {
  inline void printclnn(auto&&... values) {
#if defined (fan_gui)
    gloco->printclnn(values...);
#endif
  }
  inline void printcl(auto&&... values) {
#if defined(fan_gui)
    gloco->printcl(values...);
#endif
  }

  inline void printclnnh(int highlight, auto&&... values) {
#if defined(fan_gui)
    gloco->printclnnh(highlight, values...);
#endif
  }

  inline void printclh(int highlight, auto&&... values) {
#if defined(fan_gui)
    gloco->printclh(highlight, values...);
#endif
  }
  inline void printcl_err(auto&&... values) {
#if defined(fan_gui)
    printclh(fan::graphics::highlight_e::error, values...);
#endif
  }
  inline void printcl_warn(auto&&... values) {
#if defined(fan_gui)
    printclh(fan::graphics::highlight_e::warning, values...);
#endif
  }
}

bool init_fan_track_opengl_print = []() {
  fan_opengl_track_print = [](std::string func_name, uint64_t elapsed) {
    fan::printclnnh(fan::graphics::highlight_e::text, func_name + ":");
    fan::printclh(fan::graphics::highlight_e::warning, std::to_string(elapsed / 1e+6f)/*fan::to_string(elapsed / 1e+6)*/ + "ms");
    };
  return 1;
  }();

export namespace fan {
  namespace graphics {
    using engine_t = fan::graphics::engine_t;
    fan::graphics::image_nr_t image_load(const fan::image::image_info_t& image_info) {
      return gloco->image_load(image_info);
    }
    fan::graphics::image_nr_t image_load(const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) {
      return gloco->image_load(image_info, p);
    }
    fan::graphics::image_nr_t image_load(const std::string& path) {
      return gloco->image_load(path);
    }
    fan::graphics::image_nr_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p) {
      return gloco->image_load(path, p);
    }
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size) {
      return gloco->image_load(colors, size);
    }
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
      return gloco->image_load(colors, size, p);
    }
    void image_unload(fan::graphics::image_nr_t nr) {
      return gloco->image_unload(nr);
    }

    using light_flags_e = loco_t::light_flags_e;

    struct light_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t parallax_factor = 0;
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      uint32_t flags = 0;
      fan::vec3 angle = fan::vec3(0, 0, 0);
    };

    struct light_t : loco_t::shape_t {
      light_t(light_properties_t p = light_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::light_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .parallax_factor = p.parallax_factor,
            .size = p.size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .flags = p.flags,
            .angle = p.angle
          ));
      }
    };

    #if defined(loco_line)

      struct line_properties_t {
        camera_impl_t* camera = &gloco->orthographic_camera;
        fan::vec3 src = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
        fan::vec2 dst = fan::vec2(1, 1);
        fan::color color = fan::color(1, 1, 1, 1);
        bool blending = false;
      };

      struct line_t : loco_t::shape_t {
        line_t(line_properties_t p = line_properties_t()) {
          *(loco_t::shape_t*)this = loco_t::shape_t(
            fan_init_struct(
              typename loco_t::line_t::properties_t,
              .camera = p.camera->camera,
              .viewport = p.camera->viewport,
              .src = p.src,
              .dst = p.dst,
              .color = p.color,
              .blending = p.blending
            ));
        }
      };
    #endif

//#if defined(loco_rectangle)
    struct rectangle_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
      fan::vec2 size = fan::vec2(32, 32);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::color outline_color = color;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;
      bool blending = false;
    };

    // make sure you dont do position = vec2
    struct rectangle_t : loco_t::shape_t {
      rectangle_t(rectangle_properties_t p = rectangle_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::rectangle_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .color = p.color,
            .outline_color = p.outline_color,
            .angle = p.angle,
            .rotation_point = p.rotation_point,
            .blending = p.blending
          )
        );
      }
    };

    struct sprite_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
      fan::vec2 size = fan::vec2(32, 32);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;
      f32_t parallax_factor = 0;
      bool blending = true;
      uint32_t flags = light_flags_e::circle | light_flags_e::multiplicative;
    };


    struct sprite_t : loco_t::shape_t {
      sprite_t(sprite_properties_t p = sprite_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::sprite_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .parallax_factor = p.parallax_factor,
            .position = p.position,
            .size = p.size,
            .angle = p.angle,
            .image = p.image,
            .images = p.images,
            .color = p.color,
            .rotation_point = p.rotation_point,
            .blending = p.blending,
            .flags = p.flags
          ));
      }
    };

    struct unlit_sprite_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      bool blending = false;
    };

    struct unlit_sprite_t : loco_t::shape_t {
      unlit_sprite_t(unlit_sprite_properties_t p = unlit_sprite_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::unlit_sprite_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .angle = p.angle,
            .image = p.image,
            .images = p.images,
            .color = p.color,
            .tc_position = p.tc_position,
            .tc_size = p.tc_size,
            .rotation_point = p.rotation_point,
            .blending = p.blending
          ));
      }
    };
#if defined(loco_circle)
    struct circle_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
      f32_t radius = 32.f;
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = true;
      uint32_t flags = 0;
    };

    struct circle_t : loco_t::shape_t {
      circle_t(circle_properties_t p = circle_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::circle_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .radius = p.radius,
            .angle = p.angle,
            .color = p.color,
            .blending = p.blending,
            .flags = p.flags
          ));
      }
    };
#endif

    struct capsule_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
      fan::vec2 center0 = 0;
      fan::vec2 center1{0, 128.f};
      f32_t radius = 64.0f;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::color outline_color = color;
      bool blending = true;
      uint32_t flags = 0;
    };

    struct capsule_t : loco_t::shape_t {
      capsule_t(capsule_properties_t p = capsule_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::capsule_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .center0 = p.center0,
            .center1 = p.center1,
            .radius = p.radius,
            .color = p.color,
            .outline_color = p.outline_color,
            .blending = p.blending,
            .flags = p.flags
          ));
      }
    };

    using vertex_t = loco_t::vertex_t;
    struct polygon_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
      std::vector<vertex_t> vertices;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;
      bool blending = true;
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_strip;
    };

    struct polygon_t : loco_t::shape_t {
      polygon_t() = default;
      polygon_t(polygon_properties_t p) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::polygon_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .vertices = p.vertices,
            .angle = p.angle,
            .rotation_point = p.rotation_point,
            .blending = p.blending,
            .draw_mode = p.draw_mode
          ));
      }
    };

    struct grid_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 grid_size = fan::vec2(1, 1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec3 angle = fan::vec3(0, 0, 0);
    };
    struct grid_t : loco_t::shape_t {
      grid_t(grid_properties_t p = grid_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::grid_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .grid_size = p.grid_size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .angle = p.angle
          ));
      }
    };

    struct universal_image_renderer_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;

      bool blending = false;

      std::array<loco_t::image_t, 4> images = {
        gloco->default_texture,
        gloco->default_texture,
        gloco->default_texture,
        gloco->default_texture
      };
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    };

    struct universal_image_renderer_t : loco_t::shape_t {
      universal_image_renderer_t(const universal_image_renderer_properties_t& p = universal_image_renderer_properties_t()) {
         *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::universal_image_renderer_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .tc_position = p.tc_position,
            .tc_size = p.tc_size,
            .blending = p.blending,
            .images = p.images,
            .draw_mode = p.draw_mode,
          )
        );
      }
    };

    struct gradient_properties_t {
      camera_impl_t* camera = &gloco->perspective_camera;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      std::array<fan::color, 4> color = {
        fan::random::color(),
        fan::random::color(),
        fan::random::color(),
        fan::random::color()
      };
      bool blending = true;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    };

    struct gradient_t : loco_t::shape_t{
      gradient_t(const gradient_properties_t& p = gradient_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::gradient_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .color = p.color,
            .blending = p.blending,
            .angle = p.angle,
            .rotation_point = p.rotation_point
          )
        );
      }
    };

    struct shader_shape_properties_t {
      camera_impl_t* camera = &gloco->perspective_camera;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      uint32_t flags = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      loco_t::shader_t shader;
      bool blending = true;

      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    };

    struct shader_shape_t : loco_t::shape_t {
      shader_shape_t(const shader_shape_properties_t& p = shader_shape_properties_t()) {
       *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::shader_shape_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .angle = p.angle,
            .flags = p.flags,
            .tc_position = p.tc_position,
            .tc_size = p.tc_size,
            .shader = p.shader,
            .blending = p.blending,
            .image = p.image,
            .images = p.images,
            .draw_mode = p.draw_mode
          )
        );
      }
    };

    struct line3d_properties_t {
      camera_impl_t* camera = &gloco->perspective_camera;
      fan::vec3 src = fan::vec3(0, 0, 0);
      fan::vec3 dst = fan::vec3(10, 10, 10);
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
    };

    struct line3d_t : loco_t::shape_t {
      line3d_t(line3d_properties_t p = line3d_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::line3d_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .src = p.src,
            .dst = p.dst,
            .color = p.color,
            .blending = p.blending
          ));
      }
    };

    loco_t::polygon_t::properties_t create_hexagon(f32_t radius, const fan::color& color) {
  loco_t::polygon_t::properties_t pp;
  // for triangle strip
  for (int i = 0; i < 6; ++i) {
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(0, 0, 0), color });

    f32_t angle1 = 2 * fan::math::pi * i / 6;
    f32_t x1 = radius * std::cos(angle1);
    f32_t y1 = radius * std::sin(angle1);
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x1, y1), 0), color });

    f32_t angle2 = 2 * fan::math::pi * ((i + 1) % 6) / 6;
    f32_t x2 = radius * std::cos(angle2);
    f32_t y2 = radius * std::sin(angle2);
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x2, y2), 0), color });
  }

  return pp;
}

#if defined(loco_vfi)

    // for line
    fan::line3 get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index) {
  fan::line3 positions;
  fan::vec2 op = op_;
  switch (index) {
  case 0:
    positions[0] = fan::vec3(op - os, op_.z + 1);
    positions[1] = fan::vec3(op + fan::vec2(os.x, -os.y), op_.z + 1);
    break;
  case 1:
    positions[0] = fan::vec3(op + fan::vec2(os.x, -os.y), op_.z + 1);
    positions[1] = fan::vec3(op + os, op_.z + 1);
    break;
  case 2:
    positions[0] = fan::vec3(op + os, op_.z + 1);
    positions[1] = fan::vec3(op + fan::vec2(-os.x, os.y), op_.z + 1);
    break;
  case 3:
    positions[0] = fan::vec3(op + fan::vec2(-os.x, os.y), op_.z + 1);
    positions[1] = fan::vec3(op - os, op_.z + 1);
    break;
  default:
    // Handle invalid index if necessary
    break;
  }

  return positions;
}

    // REQUIRES to be allocated by new since lambda captures this
    // also container that it's stored in, must not change pointers
    template <typename T>
    struct vfi_root_custom_t {
      using shape_t = loco_t::shape_t;

      void create_highlight() {
        apply_highlight([](auto& h, const fan::line3& line, fan::graphics::camera_t& cam) {
          h = fan::graphics::line_t{ {
            .camera = &cam,
            .src = line[0],
            .dst = line[1],
            .color = fan::color(1, 0.5, 0, 1)
          } };
          });
      }

      void disable_highlight() {
        fan::vec3 op = children[0].get_position();
        fan::vec2 os = children[0].get_size();
        for (auto& row : highlight) {
          for (size_t i = 0; i < row.size(); ++i) {
            auto& h = row[i];
            if (!h.iic()) {
              h.set_line(0, 0);
            }
          }
        }
      }

      void set_root(const loco_t::vfi_t::properties_t& p) {
        fan::graphics::vfi_t::properties_t in = p;
        in.shape_type = 1;
        in.shape.rectangle->camera = p.shape.rectangle->camera;
        in.shape.rectangle->viewport = p.shape.rectangle->viewport;

        in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) {
          resize = (d.key == fan::key_c &&
            (d.keyboard_state == fan::keyboard_state::press ||
              d.keyboard_state == fan::keyboard_state::repeat));
          return resize ? user_cb(d) : 0;
        };

        in.mouse_button_cb = [this, user_cb = p.mouse_button_cb](const auto& d) {
          if (g_ignore_mouse || d.button != fan::mouse_left) return 0;
          if (d.button_state != fan::mouse_state::press) {
            move = moving_object = false;
            d.flag->ignore_move_focus_check = false;
            if (previous_click_position == d.position) {
              for (auto& i : selected_objects) i->disable_highlight();
              selected_objects = { this };
              create_highlight();
            }
            return 0;
          }

          if (d.mouse_stage != loco_t::vfi_t::mouse_stage_e::viewport_inside) return 0;

          if (previous_focus && previous_focus != this) {
            for (std::size_t i = 0; i < previous_focus->highlight[0].size(); ++i) {
              if (previous_focus->highlight[0][i].iic() == false) {
                previous_focus->highlight[0][i].set_line(0, 0);
              }
            }

            // if only changing focus from one to another
            // if there is multiple objects selected, don't remove the previous focus
            if (selected_objects.size() == 1) {
              if (selected_objects.back() == previous_focus) {
                selected_objects.erase(selected_objects.begin());
              }
            }
          }

          if (std::find(selected_objects.begin(), selected_objects.end(), this) == selected_objects.end()) {
            selected_objects.push_back(this);
          }

          create_highlight();
          previous_focus = this;

          if (move_and_resize_auto) {
            previous_click_position = d.position;
            click_offset = get_position() - d.position;
            move = moving_object = true;
            d.flag->ignore_move_focus_check = true;
            gloco->vfi.set_focus_keyboard(d.vfi->focus.mouse);
          }

          return user_cb(d);
        };

        in.mouse_move_cb = [this, user_cb = p.mouse_move_cb](const auto& d) {
          if (g_ignore_mouse) return 0;
          if (!move_and_resize_auto) return user_cb(d);

          if (resize && move) {
            fan::vec2 old_size = get_size();
            f32_t aspect_ratio = old_size.x / old_size.y;
            fan::vec2 drag_delta = d.position - get_position();
            if (snap) {
              drag_delta = (drag_delta / snap).round() * snap;
            }
            drag_delta = drag_delta.abs();
            fan::vec2 new_size = fan::vec2(drag_delta.x, drag_delta.x / aspect_ratio);
            if (new_size.x < 1.0f) {
              new_size.x = 1.0f;
              new_size.y = 1.0f / aspect_ratio;
            }
            if (new_size.y < 1.0f) {
              new_size.y = 1.0f;
              new_size.x = aspect_ratio;
            }
            set_size(new_size);
            update_highlight_position(this);
          }
          else if (move) {
            fan::vec3 new_pos = fan::vec3(d.position + click_offset, get_position().z);
            if (snap) {
              new_pos = (new_pos / snap).round() * snap;
            }
            set_position(new_pos, false);
          }

          return user_cb(d);
          };

        vfi_root = in;
      }

      void push_child(const shape_t& shape) {
        children.push_back({ shape });
      }

      fan::vec3 get_position() {
        return vfi_root.get_position();
      }

      void set_position(const fan::vec3& position, bool modify_depth = true) {
        fan::vec3 old_root_pos = vfi_root.get_position();
        fan::vec2 delta = fan::vec2(position - old_root_pos);
        vfi_root.set_position(position);

        for (auto& child : children) {
          fan::vec3 child_pos = child.get_position();
          child.set_position(fan::vec3(fan::vec2(child_pos) + delta, modify_depth ? position.z : child_pos.z));
        }
        update_highlight_position(this);

        for (auto* i : selected_objects) {
          if (i == this) continue;

          fan::vec3 other_old_pos = i->vfi_root.get_position();
          fan::vec2 other_delta = fan::vec2(position - old_root_pos);
          i->vfi_root.set_position(fan::vec3(fan::vec2(other_old_pos) + other_delta, modify_depth ? position.z : other_old_pos.z));

          for (auto& child : i->children) {
            fan::vec3 child_pos = child.get_position();
            child.set_position(fan::vec3(fan::vec2(child_pos) + other_delta, modify_depth ? position.z : child_pos.z));
          }
          update_highlight_position(i);
        }
      }


      fan::vec2 get_size() {
        return vfi_root.get_size();
      }

      void set_size(const fan::vec2& size) {
        fan::vec2 offset = size - vfi_root.get_size();
        vfi_root.set_size(size);
        for (auto& child : children) {
          child.set_size(child.get_size() + offset);
        }
      }

      fan::color get_color() {
        return children.empty() ? fan::color(1) : children[0].get_color();
      }

      void set_color(const fan::color& c) {
        for (auto& child : children) child.set_color(c);
      }

      static void update_highlight_position(vfi_root_custom_t<T>* instance) {
        instance->apply_highlight([](auto& h, const fan::line3& line, fan::graphics::camera_t&) {
          if (!h.iic()) h.set_line(line[0], line[1]);
          });
      }

      template<typename F>
      void apply_highlight(F&& func) {
        fan::vec3 op = children[0].get_position();
        fan::vec2 os = children[0].get_size();
        fan::graphics::camera_t cam{ children[0].get_camera(), children[0].get_viewport() };
        for (size_t j = 0; j < highlight.size(); ++j) {
          for (size_t i = 0; i < highlight[0].size(); ++i) {
            auto& h = highlight[j][i];
            auto line = get_highlight_positions(op, os, i);
            func(h, line, cam);
          }
        }
      }

      inline static bool g_ignore_mouse = false;
      inline static bool moving_object = false;

      fan::vec2 click_offset = 0;
      fan::vec2 previous_click_position;
      bool move = false;
      bool resize = false;
      bool move_and_resize_auto = true;

      shape_t vfi_root;

      struct child_data_t : shape_t, T {};
      std::vector<child_data_t> children;

      inline static std::vector<vfi_root_custom_t<T>*> selected_objects;
      inline static vfi_root_custom_t<T>* previous_focus = nullptr;

      std::vector<std::array<shape_t, 4>> highlight{ 1 };
      inline static f32_t snap = 32.f;
    };

    using vfi_root_t = vfi_root_custom_t<__empty_struct>;

  #endif
//#endif
  }
}

void init_imgui();

export namespace fan {
  namespace graphics {
    struct interactive_camera_t {
      loco_t::update_callback_nr_t uc_nr;
      f32_t zoom = 2;
      bool hovered = false;
      loco_t::camera_t reference_camera;
      loco_t::viewport_t reference_viewport;
      fan::window_t::buttons_callback_t::nr_t button_cb_nr;

      interactive_camera_t(
        loco_t::camera_t camera_nr = gloco->orthographic_camera.camera, 
        loco_t::viewport_t viewport_nr = gloco->orthographic_camera.viewport
      ) : reference_camera(camera_nr), reference_viewport(viewport_nr)
      {
        auto& window = gloco->window;
        static auto update_ortho = [&](loco_t* loco) {
          fan::vec2 s = loco->viewport_get_size(reference_viewport);
          loco->camera_set_ortho(
            reference_camera,
            fan::vec2(-s.x, s.x) / zoom,
            fan::vec2(-s.y, s.y) / zoom
          );
          };

        auto it = gloco->m_update_callback.NewNodeLast();
        gloco->m_update_callback[it] = update_ortho;

        button_cb_nr = window.add_buttons_callback([&](const auto& d) {
          if (d.button == fan::mouse_scroll_up) {
            zoom *= 1.2;
          }
          else if (d.button == fan::mouse_scroll_down) {
            zoom /= 1.2;
          }
          });
      }
      ~interactive_camera_t() {
        if (button_cb_nr.iic() == false) {
          gloco->window.remove_buttons_callback(button_cb_nr);
          button_cb_nr.sic();
        }
        if (uc_nr.iic() == false) {
          gloco->m_update_callback.unlrec(uc_nr);
          uc_nr.sic();
        }
      }
    };

    struct animator_t {
      fan::vec2 prev_dir = 0;

      uint64_t animation_update_time = 150;//ms
      uint16_t i_down = 0, i_up = 0, i_left = 0, i_right = 0;

      template <std::size_t images_per_action>
      void process_walk(loco_t::shape_t& shape,
        const fan::vec2& vel,
        const std::array<loco_t::image_t, 4>& img_idle,
        const std::array<loco_t::image_t, images_per_action>& img_movement_left,
        const std::array<loco_t::image_t, images_per_action>& img_movement_right,
        const std::array<loco_t::image_t, images_per_action>& img_movement_up,
        const std::array<loco_t::image_t, images_per_action>& img_movement_down
      ) {
        f32_t animation_velocity_threshold = 10.f;
        fan_ev_timer_loop_init(animation_update_time,
          0/*vel.y*/,
          {
          if (vel.y > animation_velocity_threshold) {
            shape.set_image(img_movement_down[i_down % images_per_action]);
            prev_dir.y = 1;
            prev_dir.x = 0;
            ++i_down;
          }
          else if (vel.y < -animation_velocity_threshold) {
            static int i = 0;
            shape.set_image(img_movement_up[i_up % images_per_action]);
            prev_dir.y = -1;
            prev_dir.x = 0;
            ++i_up;
          }
          else {
            if (prev_dir.y < 0) {
              shape.set_image(img_idle[2]);
            }
            else if (prev_dir.y > 0) {
              shape.set_image(img_idle[3]);
            }
            prev_dir.y = 0;
          }
          });

        if (prev_dir.y == 0) {
          fan_ev_timer_loop_init(animation_update_time,
            0/*vel.x*/,
            {
            if (vel.x > animation_velocity_threshold) {
              static int i = 0;
              shape.set_image(img_movement_right[i_right % images_per_action]);
              prev_dir.y = 0;
              prev_dir.x = 1;
              ++i_right;
            }
            else if (vel.x < -animation_velocity_threshold) {
              static int i = 0;
              shape.set_image(img_movement_left[i_left % images_per_action]);
              prev_dir.y = 0;
              prev_dir.x = -1;
              ++i_left;
            }
            else {
              if (prev_dir.x < 0) {
                shape.set_image(img_idle[0]);
              }
              else if (prev_dir.x > 0) {
                shape.set_image(img_idle[1]);
              }
              prev_dir.x = 0;
            }
            });
        }
      }
    };

    loco_t::polygon_t::properties_t create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
      loco_t::polygon_t::properties_t pp;
      // for triangle strip
      for (f32_t x = 0; x < groundWidth - width; x += width) {
        f32_t y1 = position.y / 2 + amplitude * std::sin(frequency * x);
        f32_t y2 = position.y / 2 + amplitude * std::sin(frequency * (x + width));

        // top
        pp.vertices.push_back({ fan::vec2(position.x + x, y1), fan::colors::red });
        // bottom
        pp.vertices.push_back({ fan::vec2(position.x + x, position.y), fan::colors::white });
        // next top
        pp.vertices.push_back({ fan::vec2(position.x + x + width, y2), fan::colors::red });
        // next bottom
        pp.vertices.push_back({ fan::vec2(position.x + x + width, position.y), fan::colors::white });
      }

      return pp;
    }
    std::vector<fan::vec2> ground_points(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
      std::vector<fan::vec2> outline_points;
      for (f32_t x = 0; x <= groundWidth; x += width) {
        f32_t y = position.y / 2 + amplitude * std::sin(frequency * x);
        outline_points.push_back(fan::vec2(position.x + x, y));
      }
      outline_points.push_back(fan::vec2(position.x + groundWidth, position.y));
      outline_points.push_back(fan::vec2(position.x, position.y));
      return outline_points;
    }
  }

  struct movement_e {
    fan_enum_string(
      ,
      left,
      right,
      up,
      down
    );
  };
}

// makes shorter code
#define fan_language \
  void main_entry(); \
  using namespace fan::graphics; \
  using namespace fan; \
  int main() { \
    fan::graphics::engine_t engine; \
    main_entry(); \
    return 0; \
  } \
  void main_entry()