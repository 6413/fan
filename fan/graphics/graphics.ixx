module;
// extension to loco.ixx


#if defined(fan_gui)
  #include <fan/imgui/imgui_internal.h>
  #include <fan/graphics/gui/imgui_themes.h>
#endif

#include <fan/graphics/opengl/init.h>
#include <fan/event/types.h>



#include <filesystem>
#include <source_location>

#define loco_vfi
#define loco_line
#define loco_rectangle
#define loco_sprite
#define loco_light
#define loco_circle
#define loco_responsive_text
#define loco_universal_image_renderer

export module fan.graphics;

//import :graphics.opengl.core; // TODO this should not be here
export import fan.graphics.loco;
import fan.graphics.common_types;
import fan.io.directory;
import fan.io.file;
import fan.time;

#if defined(fan_physics)
  import fan.physics.b2_integration;
#endif

// user friendly functions
/***************************************/

export namespace fan {
  namespace window {
    fan::vec2 get_size() {
      return gloco->window.get_size();
    }
    void set_size(const fan::vec2& size) {
      gloco->window.set_size(size);
      gloco->viewport_set(gloco->orthographic_render_view.viewport, fan::vec2(0, 0), size);
      gloco->camera_set_ortho(
        gloco->orthographic_render_view.camera,
        fan::vec2(0, size.x),
        fan::vec2(0, size.y)
      );
      gloco->viewport_set(gloco->perspective_render_view.viewport, fan::vec2(0, 0), size);
      gloco->camera_set_ortho(
        gloco->perspective_render_view.camera,
        fan::vec2(0, size.x),
        fan::vec2(0, size.y)
      );
    }

    fan::vec2 get_mouse_position() {
      return gloco->get_mouse_position();
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
  namespace graphics {
    using vfi_t = loco_t::vfi_t;

    using engine_t = loco_t;

    // creates opengl texture
    struct image_t : loco_t::image_t {
      using loco_t::image_t::image_t;
      // for no gloco access
      explicit image_t(bool) : loco_t::image_t() {}
      image_t() : loco_t::image_t(gloco->default_texture) {}
      image_t(loco_t::image_t image) : loco_t::image_t(image) {

      }
      image_t(const char* path, const std::source_location& callers_path = std::source_location::current())
        : image_t(std::string(path), callers_path) { }
      image_t(const std::string& path, const std::source_location& callers_path = std::source_location::current())
        : loco_t::image_t(gloco->image_load(path, callers_path)) {}

      fan::vec2 get_size() const {
        return gloco->image_get_data(*this).size;
      }
    };
    using render_view_t = loco_t::render_view_t;
    using viewport_t = loco_t::viewport_t;

    using shape_t = loco_t::shape_t;
    using shader_t = loco_t::shader_t;

    using shape_type_e = loco_t::shape_type_t;

    fan::graphics::image_t invalid_image = []{
      image_t image{ false };
      image.sic();
      return image;
    }();
    void add_input_action(const int* keys, std::size_t count, const std::string& action_name) {
      gloco->input_action.add(keys, count, action_name);
    }
    void add_input_action(std::initializer_list<int> keys, const std::string& action_name) {
      gloco->input_action.add(keys, action_name);
    }
    void add_input_action(int key, const std::string& action_name) {
      gloco->input_action.add(key, action_name);
    }
    bool is_input_action_active(const std::string& action_name, int pstate = loco_t::input_action_t::press) {
      return gloco->input_action.is_active(action_name);
    }

    fan::vec2 get_mouse_position() {
      return gloco->get_mouse_position();
    }
    fan::vec2 get_mouse_position(const fan::graphics::render_view_t& render_view) {
      return loco_t::transform_position(gloco->get_mouse_position(), render_view.viewport, render_view.camera);
    }
    fan::vec2 get_mouse_position(
      const loco_t::camera_t& camera,
      const loco_t::viewport_t& viewport
    ) {
      return gloco->get_mouse_position(camera, viewport);
    }

    fan::graphics::render_view_t add_render_view() {
      return gloco->render_view_create();
    }

    loco_t::render_view_t add_render_view(
      const fan::vec2& ortho_x, const fan::vec2& ortho_y,
      const fan::vec2& viewport_position, const fan::vec2& viewport_size
    ) {
      return gloco->render_view_create(ortho_x, ortho_y, viewport_position, viewport_size);
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
    fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info) {
      return gloco->image_load(image_info);
    }
    fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
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
      render_view_t* render_view = &gloco->orthographic_render_view;
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
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      light_t(light_properties_t p = light_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::light_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
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
        render_view_t* render_view = &gloco->orthographic_render_view;
        fan::vec3 src = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
        fan::vec2 dst = fan::vec2(1, 1);
        fan::color color = fan::color(1, 1, 1, 1);
        f32_t thickness = 2.0f;
        bool blending = true;
        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      };

      struct line_t : loco_t::shape_t {
        using loco_t::shape_t::shape_t;
        using loco_t::shape_t::operator=;
        line_t(line_properties_t p = line_properties_t()) {
          *(loco_t::shape_t*)this = loco_t::shape_t(
            fan_init_struct(
              typename loco_t::line_t::properties_t,
              .camera = p.render_view->camera,
              .viewport = p.render_view->viewport,
              .src = p.src,
              .dst = p.dst,
              .color = p.color,
              .thickness = p.thickness,
              .blending = p.blending,
              .draw_mode = p.draw_mode
            ));
        }
      };
    #endif

//#if defined(loco_rectangle)
    struct rectangle_properties_t {
      render_view_t* render_view = &gloco->orthographic_render_view;
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
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      rectangle_t(rectangle_properties_t p = rectangle_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::rectangle_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
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
      render_view_t* render_view = &gloco->orthographic_render_view;
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
      loco_t::texture_pack_unique_t texture_pack_unique_id;
    };


    struct sprite_t : loco_t::shape_t {
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      sprite_t(sprite_properties_t p = sprite_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::sprite_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
            .parallax_factor = p.parallax_factor,
            .position = p.position,
            .size = p.size,
            .angle = p.angle,
            .image = p.image,
            .images = p.images,
            .color = p.color,
            .rotation_point = p.rotation_point,
            .blending = p.blending,
            .flags = p.flags,
            .texture_pack_unique_id = p.texture_pack_unique_id
          ));
      }
    };

    struct unlit_sprite_properties_t {
      render_view_t* render_view = &gloco->orthographic_render_view;
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
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      unlit_sprite_t(unlit_sprite_properties_t p = unlit_sprite_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::unlit_sprite_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
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
      render_view_t* render_view = &gloco->orthographic_render_view;
      fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
      f32_t radius = 32.f;
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = true;
      uint32_t flags = 0;
    };

    struct circle_t : loco_t::shape_t {
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      circle_t(circle_properties_t p = circle_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::circle_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
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
      render_view_t* render_view = &gloco->orthographic_render_view;
      fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
      fan::vec2 center0 = 0;
      fan::vec2 center1{0, 128.f};
      f32_t radius = 64.0f;
      fan::vec3 angle = 0.f;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::color outline_color = color;
      bool blending = true;
      uint32_t flags = 0;
    };

    struct capsule_t : loco_t::shape_t {
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      capsule_t(capsule_properties_t p = capsule_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::capsule_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
            .position = p.position,
            .center0 = p.center0,
            .center1 = p.center1,
            .radius = p.radius,
            .angle = p.angle,
            .color = p.color,
            .outline_color = p.outline_color,
            .blending = p.blending,
            .flags = p.flags
          ));
      }
    };

    using vertex_t = loco_t::vertex_t;
    struct polygon_properties_t {
      render_view_t* render_view = &gloco->orthographic_render_view;
      fan::vec3 position = fan::vec3(0, 0, 0);
      std::vector<vertex_t> vertices;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;
      bool blending = true;
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_strip;
      uint32_t vertex_count = 3;
    };

    struct polygon_t : loco_t::shape_t {
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      polygon_t() = default;
      polygon_t(polygon_properties_t p) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::polygon_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
            .position = p.position,
            .vertices = p.vertices,
            .angle = p.angle,
            .rotation_point = p.rotation_point,
            .blending = p.blending,
            .draw_mode = p.draw_mode,
            .vertex_count = p.vertex_count
          ));
      }
    };

    struct grid_properties_t {
      render_view_t* render_view = &gloco->orthographic_render_view;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 grid_size = fan::vec2(1, 1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec3 angle = fan::vec3(0, 0, 0);
    };
    struct grid_t : loco_t::shape_t {
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      grid_t(grid_properties_t p = grid_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::grid_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
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
      render_view_t* render_view = &gloco->orthographic_render_view;
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
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;
      universal_image_renderer_t(const universal_image_renderer_properties_t& p = universal_image_renderer_properties_t()) {
         *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::universal_image_renderer_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
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
      render_view_t* render_view = &gloco->perspective_render_view;

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
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
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
      render_view_t* render_view = &gloco->perspective_render_view;

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

    struct shadow_properties_t {
      render_view_t* render_view = &gloco->orthographic_render_view;
      fan::vec3 position = fan::vec3(0, 0, 0);
      int shape = loco_t::shadow_t::rectangle;
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      uint32_t flags = 0;
      fan::vec3 angle = fan::vec3(0, 0, 0);
      fan::vec2 light_position = fan::vec2(0, 0);
      f32_t light_radius = 100.f;
    };

    struct shadow_t : loco_t::shape_t {
      using loco_t::shape_t::shape_t;
      using loco_t::shape_t::operator=;

      using shape_e = loco_t::shadow_t::shape_e;

      shadow_t(shadow_properties_t p = shadow_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::shadow_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
            .position = p.position,
            .shape = p.shape,
            .size = p.size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .flags = p.flags,
            .angle = p.angle,
            .light_position = p.light_position,
            .light_radius = p.light_radius
          ));
      }
    };


    struct shader_shape_t : loco_t::shape_t {
      shader_shape_t(const shader_shape_properties_t& p = shader_shape_properties_t()) {
       *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::shader_shape_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
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

  #if defined(fan_3D)
    struct line3d_properties_t {
      render_view_t* render_view = &gloco->perspective_render_view;
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
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
            .src = p.src,
            .dst = p.dst,
            .color = p.color,
            .blending = p.blending
          ));
      }
    };
  #endif


  // immediate mode draw functions
  void rectangle(const rectangle_properties_t& props) {
    gloco->add_shape_to_immediate_draw(rectangle_t(props));
  }
  void sprite(const sprite_properties_t& props) {
    gloco->add_shape_to_immediate_draw(sprite_t(props));
  }
  void unlit_sprite(const unlit_sprite_properties_t& props) {
    gloco->add_shape_to_immediate_draw(unlit_sprite_t(props));
  }
  void line(const line_properties_t& props) {
    gloco->add_shape_to_immediate_draw(line_t(props));
  }
  void light(const light_properties_t& props) {
    gloco->add_shape_to_immediate_draw(light_t(props));
  }
  void circle(const circle_properties_t& props) {
    gloco->add_shape_to_immediate_draw(circle_t(props));
  }
  void capsule(const capsule_properties_t& props) {
    gloco->add_shape_to_immediate_draw(capsule_t(props));
  }
  void polygon(const polygon_properties_t& props) {
    gloco->add_shape_to_immediate_draw(polygon_t(props));
  }
  void grid(const grid_properties_t& props) {
    gloco->add_shape_to_immediate_draw(grid_t(props));
  }
  void aabb(const fan::physics::aabb_t& b, f32_t depth = 55000, const fan::color& c = fan::color(1, 0, 0, 1)) {
    fan::graphics::line({ .src = {b.min, depth}, .dst = {b.max.x, b.min.y}, .color = c });
    fan::graphics::line({ .src = {b.max.x, b.min.y, depth}, .dst = {b.max}, .color = c });
    fan::graphics::line({ .src = {b.max, depth}, .dst = {b.min.x, b.max.y}, .color = c });
    fan::graphics::line({ .src = {b.min.x, b.max.y, depth}, .dst = {b.min}, .color = c });
  }
  void aabb(const loco_t::shape_t& s, f32_t depth = 55000, const fan::color& c = fan::color(1, 0, 0, 1)) {
    fan::graphics::aabb(s.get_aabb(), depth, c);
  }

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
        apply_highlight([](auto& h, const fan::line3& line, fan::graphics::render_view_t& render_view) {
          h = fan::graphics::line_t{ {
            .render_view = &render_view,
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
        in.shape_type = loco_t::vfi_t::shape_t::rectangle;
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
        instance->apply_highlight([](auto& h, const fan::line3& line, fan::graphics::render_view_t&) {
          if (!h.iic()) h.set_line(line[0], line[1]);
          });
      }

      template<typename F>
      void apply_highlight(F&& func) {
        fan::vec3 op = children[0].get_position();
        fan::vec2 os = children[0].get_size();
        fan::graphics::render_view_t render_view{ children[0].get_camera(), children[0].get_viewport() };
        for (size_t j = 0; j < highlight.size(); ++j) {
          for (size_t i = 0; i < highlight[0].size(); ++i) {
            auto& h = highlight[j][i];
            auto line = get_highlight_positions(op, os, i);
            func(h, line, render_view);
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
      f32_t zoom = 2;
      bool hovered = false;
      bool zoom_on_window_resize = true;
      bool pan_with_middle_mouse = false;
      fan::vec2 old_window_size{};
      fan::vec2 camera_offset{};
      loco_t::camera_t reference_camera;
      loco_t::viewport_t reference_viewport;
      fan::window_t::resize_callback_NodeReference_t resize_callback_nr;
      fan::window_t::buttons_callback_t::nr_t button_cb_nr;
      fan::window_t::mouse_motion_callback_t::nr_t mouse_motion_nr;
      loco_t::update_callback_nr_t uc_nr;

      interactive_camera_t(
        loco_t::camera_t camera_nr = gloco->orthographic_render_view.camera,
        loco_t::viewport_t viewport_nr = gloco->orthographic_render_view.viewport
      ) : reference_camera(camera_nr), reference_viewport(viewport_nr)
      {
        auto& window = gloco->window;
        old_window_size = window.get_size();

        static auto update_ortho = [&](loco_t* loco) {
          fan::vec2 s = loco->viewport_get_size(reference_viewport);
          fan::vec2 ortho_size = s / zoom;
          loco->camera_set_ortho(
            reference_camera,
            fan::vec2(-ortho_size.x, ortho_size.x),
            fan::vec2(-ortho_size.y, ortho_size.y)
          );
        };

        auto it = gloco->m_update_callback.NewNodeLast();
        gloco->m_update_callback[it] = update_ortho;

        resize_callback_nr = window.add_resize_callback([&](const auto& d) {
          if (old_window_size.x > 0 && old_window_size.y > 0) {
            fan::vec2 ratio = fan::vec2(d.size) / old_window_size;
            f32_t size_ratio = (ratio.y + ratio.x) / 2.0f;
            f32_t zoom_change = zoom * (size_ratio - 1.0f);
            zoom += zoom_change;
          }
          old_window_size = d.size;
        });

        button_cb_nr = window.add_buttons_callback([&](const auto& d) {
          if (d.button == fan::mouse_scroll_up) {
            zoom *= 1.2;
          }
          else if (d.button == fan::mouse_scroll_down) {
            zoom /= 1.2;
          }
        });

        mouse_motion_nr = window.add_mouse_motion_callback([&](const auto& d) {
          auto state = d.window->key_state(fan::mouse_middle);
          if (state == (int)fan::mouse_state::press ||
             state == (int)fan::mouse_state::repeat
            ) {
            if (pan_with_middle_mouse) {
              fan::vec2 viewport_size = gloco->viewport_get_size(reference_viewport);
              camera_offset -= (d.motion * viewport_size / (viewport_size * zoom)) * 2.f;
              gloco->camera_set_position(reference_camera, camera_offset);
            }
          }
        });
      }

      ~interactive_camera_t() {
        if (resize_callback_nr.iic() == false) {
          gloco->window.remove_resize_callback(resize_callback_nr);
          resize_callback_nr.sic();
        }
        if (button_cb_nr.iic() == false) {
          gloco->window.remove_buttons_callback(button_cb_nr);
          button_cb_nr.sic();
        }
        if (mouse_motion_nr.iic() == false) {
          gloco->window.remove_mouse_motion_callback(mouse_motion_nr);
          mouse_motion_nr.sic();
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

    struct image_divider_t {
      struct image_t {
        fan::vec2 uv_pos;
        fan::vec2 uv_size;
        loco_t::image_t image;
      };

      loco_t::image_t root_image = gloco->default_texture;
      //
      std::vector<std::vector<image_t>> images;
      //
      struct image_click_t {
        int highlight = 0;
        int count_index;
      };

      loco_t::texture_packe0::open_properties_t open_properties;
      loco_t::texture_packe0 e;
      loco_t::texture_packe0::texture_properties_t texture_properties;

      //
      image_divider_t() {
        e.open(open_properties);
        texture_properties.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_edge;
        texture_properties.min_filter = fan::graphics::image_filter::nearest;
        texture_properties.mag_filter = fan::graphics::image_filter::nearest;
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

    struct trail_segment_t {
      fan::graphics::polygon_t polygon;
      std::vector<fan::graphics::vertex_t> vertices;
      uint64_t creation_time;
      f32_t base_alpha;
    };

    struct trail_t {
      std::vector<trail_segment_t> trails;
      fan::color color = fan::colors::black.set_alpha(0.5);
      f32_t thickness = 2.f;
      uint64_t fade_duration = 2e9;
      uint64_t max_trail_lifetime = 5e9;
      loco_t::update_callback_nr_t update_callback_nr;

      trail_t() {
        update_callback_nr = gloco->m_update_callback.NewNodeLast();
        gloco->m_update_callback[update_callback_nr] = [this] (loco_t*) {
          update();
        };
      }
      ~trail_t() {
        if (!update_callback_nr) {
          return;
        }
        gloco->m_update_callback.unlrec(update_callback_nr);
        update_callback_nr.sic();
      }

      void set_point(const fan::vec3& point, f32_t drift_intensity) {
        static fan::time::timer timer{ 300000000ULL, true };
        bool should_reset = trails.empty() || timer;

        if (should_reset) {
          trails.resize(trails.size() + 1);
          trails.back().vertices.clear();
          trails.back().creation_time = fan::time::clock::now();
          trails.back().base_alpha = 0.2f + (drift_intensity * 0.6f);
        }

        bool start_new_trail = false;
        if (!trails.empty() && !trails.back().vertices.empty()) {
          fan::vec3 last_point = trails.back().vertices.back().position;
          f32_t distance = sqrt(pow(point.x - last_point.x, 2) + pow(point.y - last_point.y, 2));
          if (distance > 50.0f) {
            start_new_trail = true;
          }
        }

        if (start_new_trail) {
          trails.resize(trails.size() + 1);
          trails.back().vertices.clear();
          trails.back().creation_time = fan::time::clock::now();
          trails.back().base_alpha = 0.2f + (drift_intensity * 0.6f);
        }

        fan::vec2 direction = fan::vec2(1, 0);
        if (trails.back().vertices.size() >= 2) {
          fan::vec3 last_point = trails.back().vertices[trails.back().vertices.size() - 2].position;
          fan::vec3 diff = point - last_point;
          direction = fan::vec2(diff.x, diff.y);
          f32_t len = sqrt(direction.x * direction.x + direction.y * direction.y);
          if (len > 0) {
            direction.x /= len;
            direction.y /= len;
          }
        }

        fan::vec2 perp = fan::vec2(-direction.y, direction.x);
        fan::graphics::vertex_t vertex;
        vertex.color = fan::color(color.r, color.g, color.b, trails.back().base_alpha);

        vertex.position = point + fan::vec3(perp.x * thickness * 0.5f, perp.y * thickness * 0.5f, 0);
        trails.back().vertices.emplace_back(vertex);

        vertex.position = point - fan::vec3(perp.x * thickness * 0.5f, perp.y * thickness * 0.5f, 0);
        trails.back().vertices.emplace_back(vertex);

        trails.back().polygon = fan::graphics::polygon_t{ {
          .position = fan::vec3(0, 0, point.z),
          .vertices = trails.back().vertices,
          .draw_mode = fan::graphics::primitive_topology_t::triangle_strip,
        } };

        timer.restart();
      }

      void update() {
        uint64_t current_time = fan::time::clock::now();

        for (auto& trail : trails) {
          uint64_t age = current_time - trail.creation_time;
          f32_t fade_factor = 1.0f;

          if (age > fade_duration) {
            fade_factor = std::max(0.0f, 1.0f - static_cast<f32_t>(age - fade_duration) / static_cast<f32_t>(max_trail_lifetime - fade_duration));
          }
          f32_t current_alpha = trail.base_alpha * fade_factor;

          for (size_t i = 0; i < trail.vertices.size(); i += 2) {
            f32_t position_factor = static_cast<f32_t>(i) / static_cast<f32_t>(trail.vertices.size() - 2);
            f32_t vertex_alpha = current_alpha * (0.2f + 0.8f * position_factor);

            trail.vertices[i].color.a = vertex_alpha;     // left vertex
            trail.vertices[i + 1].color.a = vertex_alpha;   // right vertex
          }
          fan::vec3 pos = trail.polygon.get_position();
          trail.polygon = {{
            .position = pos,
            .vertices = trail.vertices,
            .draw_mode = fan::graphics::primitive_topology_t::triangle_strip,
          }};
        }

        trails.erase(
          std::remove_if(trails.begin(), trails.end(), [&](const trail_segment_t& trail) {
            uint64_t age = current_time - trail.creation_time;
            return age > max_trail_lifetime;
          }),
          trails.end()
        );
      }
    };

  } // namespace graphics

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