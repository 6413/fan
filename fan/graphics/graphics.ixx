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
#include <cmath>
#include <unordered_set>
#include <coroutine>

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
export import fan.graphics.common_context;
export import fan.graphics.common_types;
export import fan.graphics.shapes;
export import fan.io.directory;
export import fan.io.file;
export import fan.time;
export import fan.window;
export import fan.window.input_action;
export import fan.texture_pack.tp0;
export import fan.graphics.algorithm.raycast_grid;
export import fan.graphics.algorithm.pathfind;
export import fan.event;

import fan.random;

import fan.graphics.opengl.core;
#if defined(fan_physics)
  import fan.physics.types;
#endif

// user friendly functions
/***************************************/
//
export namespace fan::window {
  void add_input_action(const int* keys, std::size_t count, const std::string& action_name) {
    fan::graphics::g_render_context_handle.input_action->add(keys, count, action_name);
  }
  void add_input_action(std::initializer_list<int> keys, const std::string& action_name) {
    fan::graphics::g_render_context_handle.input_action->add(keys, action_name);
  }
  void add_input_action(int key, const std::string& action_name) {
    fan::graphics::g_render_context_handle.input_action->add(key, action_name);
  }
  bool is_input_action_active(const std::string& action_name, int pstate = fan::window::input_action_t::press) {
    return fan::graphics::g_render_context_handle.input_action->is_active(action_name);
  }
  bool is_action_clicked(const std::string& action_name) {
    return fan::graphics::g_render_context_handle.input_action->is_active(action_name);
  }
  bool is_action_down(const std::string& action_name) {
    return fan::graphics::g_render_context_handle.input_action->is_active(action_name, fan::window::input_action_t::press_or_repeat);
  }
  bool exists(const std::string& action_name) {
    return fan::graphics::g_render_context_handle.input_action->input_actions.find(action_name) != fan::graphics::g_render_context_handle.input_action->input_actions.end();
  }
}

export namespace fan {
  namespace graphics {
    using vfi_t = fan::graphics::shapes::vfi_t;

    using shape_t = fan::graphics::shapes::shape_t;
    using shape_type_t = fan::graphics::shapes::shape_type_t;

    using renderer_t = fan::window_t::renderer_t;

    fan::graphics::image_t invalid_image = []{
      image_t image{ false };
      image.sic();
      return image;
    }();

    fan::graphics::render_view_t add_render_view() {
      fan::graphics::render_view_t render_view;
      render_view.create();
      return render_view;
    }

    fan::graphics::render_view_t add_render_view(
      const fan::vec2& ortho_x, const fan::vec2& ortho_y,
      const fan::vec2& viewport_position, const fan::vec2& viewport_size
    ) {
      fan::graphics::render_view_t render_view;
      render_view.create();
      render_view.set(ortho_x, ortho_y, viewport_position, viewport_size, fan::graphics::get_window().get_size());
      return render_view;
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
  void printclnn(auto&&... values) {
  #if defined (fan_gui)
    ([&](const auto& value) {
      std::ostringstream oss;
      oss << value;
      fan::graphics::ctx().console->print(oss.str() + " ", 0);
      }(values), ...);
  #endif
  }
  void printcl(auto&&... values) {
  #if defined(fan_gui)
    printclnn(values...);
    fan::graphics::ctx().console->print("\n", 0);
  #endif
  }

  void printclnnh(int highlight, auto&&... values) {
  #if defined(fan_gui)
    ([&](const auto& value) {
      std::ostringstream oss;
      oss << value;
      fan::graphics::ctx().console->print(oss.str() + " ", highlight);
      }(values), ...);
  #endif
  }

  void printclh(int highlight, auto&&... values) {
#if defined(fan_gui)
    printclnnh(highlight, values...);
    fan::graphics::ctx().console->print("\n", highlight);
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

  #if defined(fan_opengl)
    fan::opengl::context_t& get_gl_context() {
      return (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle)));
    }
  #endif

    namespace image_presets {
      image_load_properties_t pixel_art() {
        image_load_properties_t props;
        props.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
        props.min_filter = image_filter::nearest;
        props.mag_filter = image_filter::nearest;
        return props;
      }

      image_load_properties_t smooth() {
        image_load_properties_t props;
        props.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
        props.min_filter = image_filter::linear;
        props.mag_filter = image_filter::linear;
        return props;
      }

      image_load_properties_t mipmapped() {
        image_load_properties_t props;
        props.min_filter = image_filter::linear_mipmap_linear;
        props.mag_filter = image_filter::linear;
        return props;
      }
    }

    std::vector<uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp = 0, fan::vec2 uvs = 1) {
      if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) {
        return fan::graphics::ctx()->image_get_pixel_data(fan::graphics::ctx(), nr, fan::opengl::context_t::global_to_opengl_format(image_format), uvp, uvs);
      }
      else {
        fan::throw_error("");
        return {};
      }
    }
    fan::graphics::image_nr_t image_create() {
      return fan::graphics::ctx()->image_create(fan::graphics::ctx());
    }
    fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr) {
      fan::graphics::context_image_t img;

      if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) {
        img.gl = *(fan::opengl::context_t::image_t*)fan::graphics::ctx()->image_get(fan::graphics::ctx(), nr);
      }
    #if defined(fan_vulkan)
      else if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) {
        img.vk = *(fan::vulkan::context_t::image_t*)fan::graphics::ctx()->image_get(fan::graphics::ctx(), nr);
      }
    #endif
      return img;
    }
    uint64_t image_get_handle(fan::graphics::image_nr_t nr) {
      return fan::graphics::ctx()->image_get_handle(fan::graphics::ctx(), nr);
    }
    void image_erase(fan::graphics::image_nr_t nr) {
      fan::graphics::ctx()->image_erase(fan::graphics::ctx(), nr);
    }
    void image_bind(fan::graphics::image_nr_t nr) {
      fan::graphics::ctx()->image_bind(fan::graphics::ctx(), nr);
    }
    void image_unbind(fan::graphics::image_nr_t nr) {
      fan::graphics::ctx()->image_unbind(fan::graphics::ctx(), nr);
    }
    fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr) {
      return fan::graphics::ctx()->image_get_settings(fan::graphics::ctx(), nr);
    }
    void image_set_settings(fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
      fan::graphics::ctx()->image_set_settings(fan::graphics::ctx(), nr, settings);
    }
    fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info) {
      return fan::graphics::ctx()->image_load_info(fan::graphics::ctx(), image_info);
    }
    fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
      return fan::graphics::ctx()->image_load_info_props(fan::graphics::ctx(), image_info, p);
    }
    fan::graphics::image_nr_t image_load(const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
      return fan::graphics::ctx()->image_load_path(fan::graphics::ctx(), path, callers_path);
    }
    fan::graphics::image_nr_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
      return fan::graphics::ctx()->image_load_path_props(fan::graphics::ctx(), path, p, callers_path);
    }
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size) {
      return fan::graphics::ctx()->image_load_colors(fan::graphics::ctx(), colors, size);
    }
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
      return fan::graphics::ctx()->image_load_colors_props(fan::graphics::ctx(), colors, size, p);
    }
    void image_unload(fan::graphics::image_nr_t nr) {
      fan::graphics::ctx()->image_unload(fan::graphics::ctx(), nr);
    }
    bool is_image_valid(fan::graphics::image_nr_t nr) {
      return nr != fan::graphics::ctx().default_texture && nr.iic() == false;
    }
    fan::graphics::image_t image_load_pixel_art(const std::string& path) {
      return image_load(path, image_presets::pixel_art());
    }
    fan::graphics::image_t image_load_smooth(const std::string& path) {
      return image_load(path, image_presets::smooth());
    }

    fan::graphics::image_nr_t create_missing_texture() {
      return fan::graphics::ctx()->create_missing_texture(fan::graphics::ctx());
    }
    fan::graphics::image_nr_t create_transparent_texture() {
      return fan::graphics::ctx()->create_transparent_texture(fan::graphics::ctx());
    }
    void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
      fan::graphics::ctx()->image_reload_image_info(fan::graphics::ctx(), nr, image_info);
    }
    void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
      fan::graphics::ctx()->image_reload_image_info_props(fan::graphics::ctx(), nr, image_info, p);
    }
    void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
      fan::graphics::ctx()->image_reload_path(fan::graphics::ctx(), nr, path, callers_path);
    }
    void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
      fan::graphics::ctx()->image_reload_path_props(fan::graphics::ctx(), nr, path, p, callers_path);
    }
    fan::graphics::image_nr_t image_create(const fan::color& color) {
      return fan::graphics::ctx()->image_create_color(fan::graphics::ctx(), color);
    }
    fan::graphics::image_nr_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p) {
      return fan::graphics::ctx()->image_create_color_props(fan::graphics::ctx(), color, p);
    }

    fan::graphics::shader_nr_t shader_create() {
      return fan::graphics::ctx()->shader_create(fan::graphics::ctx());
    }
    void shader_erase(fan::graphics::shader_nr_t nr) {
      fan::graphics::ctx()->shader_erase(fan::graphics::ctx(), nr);
    }
    void shader_use(fan::graphics::shader_nr_t nr) {
      fan::graphics::ctx()->shader_use(fan::graphics::ctx(), nr);
    }
    void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
      fan::graphics::ctx()->shader_set_vertex(fan::graphics::ctx(), nr, vertex_code);
    }
    void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
      fan::graphics::ctx()->shader_set_fragment(fan::graphics::ctx(), nr, fragment_code);
    }
    bool shader_compile(fan::graphics::shader_nr_t nr) {
      return fan::graphics::ctx()->shader_compile(fan::graphics::ctx(), nr);
    }
    template <typename T>
    void shader_set_value(fan::graphics::shader_nr_t nr, const std::string& name, const T& val) {
      if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) {
        get_gl_context().shader_set_value(nr, name, val);
      }
      else if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) {
        fan::throw_error("todo");
      }
    }
    void shader_set_camera(fan::graphics::shader_t nr, fan::graphics::camera_t camera_nr) {
      if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) {
        get_gl_context().shader_set_camera(nr, camera_nr);
      }
    #if defined(fan_vulkan)
      else if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) {
        fan::throw_error("todo");
      }
    #endif
    }
    fan::graphics::shader_nr_t shader_get_nr(uint16_t shape_type) {
      return fan::graphics::get_shapes().shaper.GetShader(shape_type);
    }
    auto& shader_get_data(uint16_t shape_type) {
      return (*fan::graphics::ctx().shader_list)[shader_get_nr(shape_type)];
    }
    bool shader_update_fragment(uint16_t shape_type, const std::string& fragment) {
      auto shader_nr = shader_get_nr(shape_type);
      auto shader_data = shader_get_data(shape_type);
      shader_set_vertex(shader_nr, shader_data.svertex);
      shader_set_fragment(shader_nr, fragment);
      return shader_compile(shader_nr);
    }

    fan::graphics::camera_nr_t camera_create() {
      return fan::graphics::ctx()->camera_create(fan::graphics::ctx());
    }
    fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr) {
      return fan::graphics::ctx()->camera_get(fan::graphics::ctx(), nr);
    }
    void camera_erase(fan::graphics::camera_nr_t nr) {
      fan::graphics::ctx()->camera_erase(fan::graphics::ctx(), nr);
    }
    fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y) {
      return fan::graphics::ctx()->camera_create_params(fan::graphics::ctx(), x, y);
    }
    fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr) {
      return fan::graphics::ctx()->camera_get_position(fan::graphics::ctx(), nr);
    }
    void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
      fan::graphics::ctx()->camera_set_position(fan::graphics::ctx(), nr, cp);
    }
    fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr) {
      return fan::graphics::ctx()->camera_get_size(fan::graphics::ctx(), nr);
    }
    fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr) {
      return fan::graphics::ctx()->viewport_get_size(fan::graphics::ctx(), nr);
    }
    f32_t camera_get_zoom(fan::graphics::camera_nr_t nr, fan::graphics::viewport_nr_t viewport) {
      fan::vec2 s = viewport_get_size(viewport);
      auto& camera = camera_get(nr);
      return (s.x * 2) / (camera.coordinates.right - camera.coordinates.left);
    }
    void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
      fan::graphics::ctx()->camera_set_ortho(fan::graphics::ctx(), nr, x, y);
    }
    void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
      fan::graphics::ctx()->camera_set_perspective(fan::graphics::ctx(), nr, fov, window_size);
    }
    void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
      fan::graphics::ctx()->camera_rotate(fan::graphics::ctx(), nr, offset);
    }
    void camera_set_target(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed = 10) {
      f32_t screen_height = fan::graphics::get_window().get_size()[1];
      f32_t pixels_from_bottom = 400.0f;

      /* target - (screen_height / 2 - pixels_from_bottom) / (ic.zoom * 1.5))*/;

      fan::vec2 src = camera_get_position(fan::graphics::get_orthographic_render_view().camera);
      camera_set_position(
        fan::graphics::get_orthographic_render_view().camera,
        move_speed == 0 ? target : src + (target - src) * fan::graphics::get_window().m_delta_time * move_speed
      );
    }

    fan::graphics::viewport_nr_t viewport_create() {
      return fan::graphics::ctx()->viewport_create(fan::graphics::ctx());
    }
    fan::graphics::viewport_nr_t viewport_create(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
      return fan::graphics::ctx()->viewport_create_params(fan::graphics::ctx(), viewport_position, viewport_size, fan::graphics::get_window().get_size());
    }
    fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr) {
      return fan::graphics::ctx()->viewport_get(fan::graphics::ctx(), nr);
    }
    void viewport_erase(fan::graphics::viewport_nr_t nr) {
      fan::graphics::ctx()->viewport_erase(fan::graphics::ctx(), nr);
    }
    fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr) {
      return fan::graphics::ctx()->viewport_get_position(fan::graphics::ctx(), nr);
    }
    void viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
      fan::graphics::ctx()->viewport_set(fan::graphics::ctx(), viewport_position, viewport_size, fan::graphics::get_window().get_size());
    }
    void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
      fan::graphics::ctx()->viewport_set_nr(fan::graphics::ctx(), nr, viewport_position, viewport_size, fan::graphics::get_window().get_size());
    }
    void viewport_zero(fan::graphics::viewport_nr_t nr) {
      fan::graphics::ctx()->viewport_zero(fan::graphics::ctx(), nr);
    }
    bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
      return fan::graphics::ctx()->viewport_inside(fan::graphics::ctx(), nr, position);
    }
    bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
      return fan::graphics::ctx()->viewport_inside_wir(fan::graphics::ctx(), nr, position);
    }
    bool inside(const fan::graphics::render_view_t& render_view, const fan::vec2& position) {
      fan::vec2 tp = fan::graphics::transform_position(position, render_view.viewport, render_view.camera);

      auto c = fan::graphics::camera_get(render_view.camera);
      f32_t l = c.coordinates.left;
      f32_t r = c.coordinates.right;
      f32_t t = c.coordinates.up;
      f32_t b = c.coordinates.down;

      return tp.x >= l && tp.x <= r &&
        tp.y >= t && tp.y <= b;
    }
    bool is_mouse_inside(const fan::graphics::render_view_t& render_view) {
      return inside(render_view, get_mouse_position());;
    }


    using light_flags_e = fan::graphics::light_flags_e;

    struct light_properties_t {
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t parallax_factor = 0;
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      uint32_t flags = 0;
      fan::vec3 angle = fan::vec3(0, 0, 0);
    };

    struct light_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      light_t(light_properties_t p = light_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::light_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
            .position = p.position,
            .parallax_factor = p.parallax_factor,
            .size = p.size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .flags = p.flags,
            .angle = p.angle
          )
        );
      }
      light_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view)
        : light_t(light_properties_t{
          .render_view = render_view,
          .position = position,
          .size = size,
          .color = color
          }) {
      }
    };

    #if defined(loco_line)

      struct line_properties_t {
        render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
        fan::vec3 src = fan::vec3(fan::vec2(fan::graphics::g_render_context_handle.window->get_size() / 2), 0);
        fan::vec2 dst = fan::vec2(1, 1);
        fan::color color = fan::color(1, 1, 1, 1);
        f32_t thickness = 4.0f;
        bool blending = true;
        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      };

      struct line_t : fan::graphics::shapes::shape_t {
        using fan::graphics::shapes::shape_t::shape_t;
        using fan::graphics::shapes::shape_t::operator=;
        operator fan::graphics::shapes::shape_t& () { return *this; }
        operator const fan::graphics::shapes::shape_t& () const { return *this; }
        line_t(line_properties_t p = line_properties_t()) {
          *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
            fan_init_struct(
              typename fan::graphics::shapes::line_t::properties_t,
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
        line_t(const fan::vec3& src, const fan::vec3& dst, const fan::color& color, f32_t thickness = 3.f, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view)
          : line_t(line_properties_t{
            .render_view = render_view,
            .src = src,
            .dst = dst,
            .color = color,
            .thickness = thickness
          }) {
        }
      };
    #endif

//#if defined(loco_rectangle)
    struct rectangle_properties_t {
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::g_render_context_handle.window->get_size() / 2), 0);
      fan::vec2 size = fan::vec2(32, 32);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::color outline_color = color;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;
      bool blending = true;
    };

    // make sure you dont do position = vec2
    struct rectangle_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      rectangle_t(rectangle_properties_t p = rectangle_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::rectangle_t::properties_t,
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
      rectangle_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view)
        : rectangle_t(rectangle_properties_t{
          .render_view = render_view,
          .position = position,
          .size = size,
          .color = color
        }) {
      }
    };

    struct sprite_properties_t {
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::g_render_context_handle.window->get_size() / 2), 0);
      fan::vec2 size = fan::vec2(32, 32);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      fan::graphics::image_t image = fan::graphics::ctx().default_texture;
      std::array<fan::graphics::image_t, 30> images;
      f32_t parallax_factor = 0;
      bool blending = true;
      uint32_t flags = light_flags_e::circle | light_flags_e::multiplicative;
      fan::graphics::texture_pack::unique_t texture_pack_unique_id;
    };


    struct sprite_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      sprite_t(sprite_properties_t p = sprite_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::sprite_t::properties_t,
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
          )
        );
      }
      sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view)
        : sprite_t(sprite_properties_t{
          .render_view = render_view,
          .position = position,
          .size = size,
          .image = image
        }) {
      }
    };

    struct unlit_sprite_properties_t {
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::g_render_context_handle.window->get_size() / 2), 0);
      fan::vec2 size = fan::vec2(32, 32);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      fan::graphics::image_t image = fan::graphics::ctx().default_texture;
      std::array<fan::graphics::image_t, 30> images;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      bool blending = false;
    };

    struct unlit_sprite_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      unlit_sprite_t(unlit_sprite_properties_t p = unlit_sprite_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::unlit_sprite_t::properties_t,
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
          )
        );
      }
      unlit_sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view)
        : unlit_sprite_t(unlit_sprite_properties_t{
          .render_view = render_view,
          .position = position,
          .size = size,
          .image = image
         }) {
      }
    };
#if defined(loco_circle)
    struct circle_properties_t {
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::g_render_context_handle.window->get_size() / 2), 0);
      f32_t radius = 32.f;
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = true;
      uint32_t flags = 0;
    };

    struct circle_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      circle_t(circle_properties_t p = circle_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::circle_t::properties_t,
            .camera = p.render_view->camera,
            .viewport = p.render_view->viewport,
            .position = p.position,
            .radius = p.radius,
            .angle = p.angle,
            .color = p.color,
            .blending = p.blending,
            .flags = p.flags
          )
        );
      }
      circle_t(const fan::vec3& position, float radius, const fan::color& color, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view)
        : circle_t(circle_properties_t{
          .render_view = render_view,
          .position = position,
          .radius = radius,
          .color = color
          }) {
      }
    };
#endif

    struct capsule_properties_t {
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::g_render_context_handle.window->get_size() / 2), 0);
      fan::vec2 center0 = 0;
      fan::vec2 center1{0, 128.f};
      f32_t radius = 64.0f;
      fan::vec3 angle = 0.f;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::color outline_color = color;
      bool blending = true;
      uint32_t flags = 0;
    };

    struct capsule_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      capsule_t(capsule_properties_t p = capsule_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::capsule_t::properties_t,
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

    struct polygon_properties_t {
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(0, 0, 0);
      std::vector<vertex_t> vertices;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;
      bool blending = true;
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_strip;
      uint32_t vertex_count = 3;
    };

    struct polygon_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      polygon_t() = default;
      polygon_t(polygon_properties_t p) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::polygon_t::properties_t,
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
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 grid_size = fan::vec2(1, 1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec3 angle = fan::vec3(0, 0, 0);
    };
    struct grid_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      grid_t(grid_properties_t p = grid_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::grid_t::properties_t,
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
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;

      bool blending = false;

      std::array<fan::graphics::image_t, 4> images = {
        fan::graphics::ctx().default_texture,
        fan::graphics::ctx().default_texture,
        fan::graphics::ctx().default_texture,
        fan::graphics::ctx().default_texture
      };
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    };

    struct universal_image_renderer_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      universal_image_renderer_t(const universal_image_renderer_properties_t& p = universal_image_renderer_properties_t()) {
         *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::universal_image_renderer_t::properties_t,
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
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;

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

    struct gradient_t : fan::graphics::shapes::shape_t{
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      gradient_t(const gradient_properties_t& p = gradient_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::gradient_t::properties_t,
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
      render_view_t* render_view = fan::graphics::ctx().perspective_render_view;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      uint32_t flags = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      fan::graphics::shader_t shader;
      bool blending = true;

      fan::graphics::image_t image = fan::graphics::ctx().default_texture;
      std::array<fan::graphics::image_t, 30> images;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    };

    struct shadow_properties_t {
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
      fan::vec3 position = fan::vec3(0, 0, 0);
      int shape = fan::graphics::shapes::shadow_t::rectangle;
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      uint32_t flags = 0;
      fan::vec3 angle = fan::vec3(0, 0, 0);
      fan::vec2 light_position = fan::vec2(0, 0);
      f32_t light_radius = 100.f;
    };

    struct shadow_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      using shape_e = fan::graphics::shapes::shadow_t::shape_e;

      shadow_t(shadow_properties_t p = shadow_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::shadow_t::properties_t,
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


    struct shader_shape_t : fan::graphics::shapes::shape_t {
      shader_shape_t(const shader_shape_properties_t& p = shader_shape_properties_t()) {
       *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::shader_shape_t::properties_t,
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
      render_view_t* render_view = fan::graphics::ctx().perspective_render_view;
      fan::vec3 src = fan::vec3(0, 0, 0);
      fan::vec3 dst = fan::vec3(10, 10, 10);
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
    };

    struct line3d_t : fan::graphics::shapes::shape_t {
      using fan::graphics::shapes::shape_t::shape_t;
      using fan::graphics::shapes::shape_t::operator=;
      operator fan::graphics::shapes::shape_t& () { return *this; }
      operator const fan::graphics::shapes::shape_t& () const { return *this; }
      line3d_t(line3d_properties_t p = line3d_properties_t()) {
        *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
          fan_init_struct(
            typename fan::graphics::shapes::line3d_t::properties_t,
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


  struct aabb_t {
    fan::vec3 center;
    fan::vec2 half_size;
    fan::color color = fan::color(1, 0, 0, 1);
    f32_t depth = 55000;
    std::array<fan::graphics::shapes::shape_t, 4> edges;

    aabb_t() = default;

    aabb_t(const fan::vec3& c, const fan::vec2& hsize, f32_t d = 55000, const fan::color& col = fan::color(1, 0, 0, 1))
      : center(c), half_size(hsize), depth(d), color(col)
    {
      fan::vec3 bl(center.x - half_size.x, center.y - half_size.y, depth);
      fan::vec3 br(center.x + half_size.x, center.y - half_size.y, depth);
      fan::vec3 tr(center.x + half_size.x, center.y + half_size.y, depth);
      fan::vec3 tl(center.x - half_size.x, center.y + half_size.y, depth);

      edges[0] = line_t(line_properties_t{ .src = bl, .dst = br, .color = color });
      edges[1] = line_t(line_properties_t{ .src = br, .dst = tr, .color = color });
      edges[2] = line_t(line_properties_t{ .src = tr, .dst = tl, .color = color });
      edges[3] = line_t(line_properties_t{ .src = tl, .dst = bl, .color = color });
    }
  };

  fan::graphics::shapes::shape_t& add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s) {
    fan::graphics::get_shapes().immediate_render_list->emplace_back(std::move(s));
    return fan::graphics::get_shapes().immediate_render_list->back();
  }
  auto add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s) {
    auto ret = s.NRI;
    (*fan::graphics::get_shapes().static_render_list)[ret] = std::move(s);
    return ret;
  }
  void remove_static_shape_draw(const fan::graphics::shapes::shape_t& s) {
    fan::graphics::get_shapes().static_render_list->erase(s.NRI);
  }

  // immediate mode draw functions. Dont store the references of shapes given by the immediate draw, they are destroyed after end of current frame
  fan::graphics::shapes::shape_t& rectangle(const rectangle_properties_t& props = {}) {
    return add_shape_to_immediate_draw(rectangle_t(props));
  }
  fan::graphics::shapes::shape_t& rectangle(const fan::vec3& position, const fan::vec2& size, const fan::color& color = fan::colors::white, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view) {
    return add_shape_to_immediate_draw(rectangle_t(rectangle_properties_t{
      .render_view = render_view,
      .position = position,
      .size = size,
      .color = color
      }));
  }
  fan::graphics::shapes::shape_t& sprite(const sprite_properties_t& props = {}) {
    return add_shape_to_immediate_draw(sprite_t(props));
  }
  fan::graphics::shapes::shape_t& unlit_sprite(const unlit_sprite_properties_t& props = {}) {
    return add_shape_to_immediate_draw(unlit_sprite_t(props));
  }
  fan::graphics::shapes::shape_t& line(const line_properties_t& props = {}) {
    return add_shape_to_immediate_draw(line_t(props));
  }
  fan::graphics::shapes::shape_t& line(const fan::vec3& src, const fan::vec3& dst, const fan::color& color, f32_t thickness = 3.f, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view) {
    return add_shape_to_immediate_draw(line_t(line_properties_t{
      .render_view = render_view,
      .src = src,
      .dst = dst,
      .color = color,
      .thickness = thickness
      }));
  }
  fan::graphics::shapes::shape_t& light(const light_properties_t& props = {}) {
    return add_shape_to_immediate_draw(light_t(props));
  }
  fan::graphics::shapes::shape_t& circle(const circle_properties_t& props = {}) {
    return add_shape_to_immediate_draw(circle_t(props));
  }
  fan::graphics::shapes::shape_t& circle(const fan::vec3& position, f32_t radius, const fan::color& color, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view) {
    return add_shape_to_immediate_draw(circle_t(circle_properties_t{
      .render_view = render_view,
      .position = position,
      .radius = radius,
      .color = color
      }));
  }
  fan::graphics::shapes::shape_t& capsule(const capsule_properties_t& props = {}) {
    return add_shape_to_immediate_draw(capsule_t(props));
  }
  fan::graphics::shapes::shape_t& polygon(const polygon_properties_t& props = {}) {
    return add_shape_to_immediate_draw(polygon_t(props));
  }
  fan::graphics::shapes::shape_t& grid(const grid_properties_t& props = {}) {
    return add_shape_to_immediate_draw(grid_t(props));
  }
  #if defined(fan_physics)
  void aabb(const fan::physics::aabb_t& b, f32_t depth = 55000, const fan::color& c = fan::color(1, 0, 0, 1)) {
    fan::graphics::line({ .src = {b.min, depth}, .dst = {b.max.x, b.min.y}, .color = c });
    fan::graphics::line({ .src = {b.max.x, b.min.y, depth}, .dst = {b.max}, .color = c });
    fan::graphics::line({ .src = {b.max, depth}, .dst = {b.min.x, b.max.y}, .color = c });
    fan::graphics::line({ .src = {b.min.x, b.max.y, depth}, .dst = {b.min}, .color = c });
  }
  void aabb(const fan::graphics::shapes::shape_t& s, f32_t depth = 55000, const fan::color& c = fan::color(1, 0, 0, 1)) {
    fan::graphics::aabb(s.get_aabb(), depth, c);
  }
  #endif

  fan::graphics::shapes::polygon_t::properties_t create_hexagon(f32_t radius, const fan::color& color) {
    fan::graphics::shapes::polygon_t::properties_t pp;
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
    using shape_t = fan::graphics::shapes::shape_t;

    void enable_highlight() {
      apply_highlight([](auto& h, const fan::line3& line, fan::graphics::render_view_t& rv) {
        h = fan::graphics::line_t{ {&rv, line[0], line[1], fan::color(1, 0.5, 0, 1)} };
      });
    }

    void disable_highlight() {
      apply_highlight([](auto& h, const fan::line3&, fan::graphics::render_view_t&) {
        if (!h.iic()) {
          h.set_line(0, 0);
        }
      });
    }

    void set_root(const fan::graphics::shapes::vfi_t::properties_t& p) {
      fan::graphics::vfi_t::properties_t in = p;
      in.shape_type = fan::graphics::shapes::vfi_t::shape_t::rectangle;
      in.shape.rectangle->camera = p.shape.rectangle->camera;
      in.shape.rectangle->viewport = p.shape.rectangle->viewport;

      in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) {
        resize = d.key == fan::key_c &&
          (d.keyboard_state == fan::keyboard_state::press ||
            d.keyboard_state == fan::keyboard_state::repeat);
        return resize ? user_cb(d) : 0;
        };

      in.mouse_button_cb = [this, user_cb = p.mouse_button_cb](const auto& d) {
        if (g_ignore_mouse || d.button != fan::mouse_left) {
          return 0;
        }

        if (d.button_state != fan::mouse_state::press) {
          move = moving_object = false;
          d.flag->ignore_move_focus_check = false;
          if (previous_click_position == d.position) {
            for (auto& i : selected_objects) {
              i->disable_highlight();
            }
            selected_objects = { this };
            enable_highlight();
          }
          return 0;
        }

        if (d.mouse_stage != fan::graphics::shapes::vfi_t::mouse_stage_e::viewport_inside) {
          return 0;
        }

        if (previous_focus && previous_focus != this) {
          previous_focus->disable_highlight();
          if (selected_objects.size() == 1 && selected_objects.back() == previous_focus) {
            selected_objects.erase(selected_objects.begin());
          }
        }

        if (std::find(selected_objects.begin(), selected_objects.end(), this) == selected_objects.end()) {
          selected_objects.push_back(this);
        }

        enable_highlight();
        previous_focus = this;

        if (move_and_resize_auto) {
          previous_click_position = d.position;
          click_offset = get_position() - d.position;
          move = moving_object = true;
          d.flag->ignore_move_focus_check = true;
          fan::graphics::g_shapes->vfi.set_focus_keyboard(d.vfi->focus.mouse);
        }

        return user_cb(d);
      };

      in.mouse_move_cb = [this, user_cb = p.mouse_move_cb](const auto& d) {
        if (g_ignore_mouse || !move_and_resize_auto) {
          return user_cb ? user_cb(d) : 0;
        }

        if (resize && move) {
          fan::vec2 old_size = get_size();
          f32_t aspect_ratio = old_size.x / old_size.y;
          fan::vec2 drag_delta = d.position - get_position();
          if (snap) {
            drag_delta = (drag_delta / snap).round() * snap;
          }
          drag_delta = drag_delta.abs();
          fan::vec2 new_size(drag_delta.x, drag_delta.x / aspect_ratio);
          if (new_size.x < 1.0f) {
            new_size = { 1.0f, 1.0f / aspect_ratio };
          }
          if (new_size.y < 1.0f) {
            new_size = { aspect_ratio, 1.0f };
          }
          set_size(new_size);
          update_highlight_position(this);
        }
        else if (move) {
          fan::vec3 new_pos(d.position + click_offset, get_position().z);
          if (snap) {
            new_pos = (new_pos / snap).round() * snap;
          }
          set_position(new_pos, false);
          for (auto& child : children) {
      auto c = child.get_color();
      auto i = child.get_image();
      if (c.a != 1.0f) {
        fan::print("Alpha changed during drag:", c.a);
        c.a = 1.0f;
        child.set_color(c);
      }
    }
        }
        return user_cb ? user_cb(d) : 0;
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
      fan::vec2 delta = fan::vec2(position - vfi_root.get_position());
      modify_depth ? vfi_root.set_position(position) : vfi_root.set_position(fan::vec2(position));

      for (auto& child : children) {
        fan::vec3 cp = child.get_position();
        fan::vec3 new_pos = fan::vec3(fan::vec2(cp) + delta, modify_depth ? position.z : cp.z);
        modify_depth ? child.set_position(new_pos) : child.set_position(fan::vec2(new_pos));
      }
      update_highlight_position(this);

      for (auto* i : selected_objects) {
        if (i == this) {
          continue;
        }
        fan::vec3 old_pos = i->vfi_root.get_position();
        fan::vec3 new_pos = fan::vec3(fan::vec2(old_pos) + delta, modify_depth ? position.z : old_pos.z);
        modify_depth ? i->vfi_root.set_position(new_pos) : i->vfi_root.set_position(fan::vec2(new_pos));

        for (auto& child : i->children) {
          fan::vec3 cp = child.get_position();
          fan::vec3 new_child_pos = fan::vec3(fan::vec2(cp) + delta, modify_depth ? position.z : cp.z);
          modify_depth ? child.set_position(new_child_pos) : child.set_position(fan::vec2(new_child_pos));
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
      for (auto& child : children) {
        child.set_color(c);
      }
    }

    static void update_highlight_position(vfi_root_custom_t<T>* instance) {
      instance->apply_highlight([](auto& h, const fan::line3& line, fan::graphics::render_view_t&) {
        if (!h.iic()) {
          h.set_line(line[0], line[1]);
        }
        });
    }

    template<typename F>
    void apply_highlight(F&& func) {
      fan::vec3 op = children[0].get_position();
      fan::vec2 os = children[0].get_size();
      fan::graphics::render_view_t rv{ children[0].get_camera(), children[0].get_viewport() };
      for (size_t j = 0; j < highlight.size(); ++j) {
        for (size_t i = 0; i < highlight[0].size(); ++i) {
          func(highlight[j][i], get_highlight_positions(op, os, i), rv);
        }
      }
    }

    inline static bool g_ignore_mouse = false;
    inline static bool moving_object = false;
    inline static std::vector<vfi_root_custom_t<T>*> selected_objects;
    inline static vfi_root_custom_t<T>* previous_focus = nullptr;
    inline static f32_t snap = 32.f;

    fan::vec2 click_offset = 0;
    fan::vec2 previous_click_position;
    bool move = false;
    bool resize = false;
    bool move_and_resize_auto = true;
    shape_t vfi_root;
    struct child_data_t : shape_t, T {};
    std::vector<child_data_t> children;
    std::vector<std::array<shape_t, 4>> highlight{ 1 };
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
      interactive_camera_t(const interactive_camera_t&) = delete;
      interactive_camera_t(interactive_camera_t&&) = delete;

      interactive_camera_t& operator=(interactive_camera_t&&) = delete;
      interactive_camera_t& operator=(const interactive_camera_t&) = delete;

      void reset() {
        ignore = false;
        zoom_on_window_resize = true;
        pan_with_middle_mouse = false;
        reset_view();
      }
      void reset_view() {
        zoom = 1;
        camera_offset = {};
        update();
      }

      void update() {
        fan::vec2 s = fan::graphics::g_render_context_handle->viewport_get_size(
          fan::graphics::g_render_context_handle,
          reference_viewport
        );
        fan::vec2 ortho_size = s / zoom;
        fan::graphics::g_render_context_handle->camera_set_ortho(
          fan::graphics::g_render_context_handle,
          reference_camera,
          fan::vec2(-ortho_size.x / 2.f, ortho_size.x / 2.f),
          fan::vec2(-ortho_size.y / 2.f, ortho_size.y / 2.f)
        );
      }

      
      void create(
        fan::graphics::camera_t camera_nr = fan::graphics::get_orthographic_render_view().camera,
        fan::graphics::viewport_t viewport_nr = fan::graphics::get_orthographic_render_view().viewport,
        f32_t new_zoom = 2
      ) {
        reference_camera = camera_nr;
        reference_viewport = viewport_nr;
        zoom = new_zoom;
        auto& window = fan::graphics::get_window();
        old_window_size = window.get_size();

        static auto update_ortho = [this](void* ptr) {
          update();
        };

        auto it = fan::graphics::ctx().update_callback->NewNodeLast();
        (*fan::graphics::ctx().update_callback)[it] = update_ortho;

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
          if (ignore) {
            return;
          }

          bool mouse_inside_viewport = fan::graphics::inside(
            reference_viewport,
            fan::window::get_mouse_position()
          );
          if (mouse_inside_viewport) {
            if (d.button == fan::mouse_scroll_up) {
              zoom *= 1.2;
              update();
            }
            else if (d.button == fan::mouse_scroll_down) {
              zoom /= 1.2;
              update();
            }
          }
          auto state = d.window->key_state(fan::mouse_middle);
          if (state == (int)fan::mouse_state::press) {
            clicked_inside_viewport = mouse_inside_viewport;
          }
          else if (state == (int)fan::mouse_state::release) {
            clicked_inside_viewport = false;
          }
        });
        mouse_motion_nr = window.add_mouse_motion_callback([&](const auto& d) {
          auto state = d.window->key_state(fan::mouse_middle);
          if (state == (int)fan::mouse_state::press ||
            state == (int)fan::mouse_state::repeat
            ) {
            if (pan_with_middle_mouse && clicked_inside_viewport) {
              fan::vec2 viewport_size = fan::graphics::viewport_get_size(reference_viewport);
              camera_offset -= (d.motion * viewport_size / (viewport_size * zoom));
              fan::graphics::camera_set_position(reference_camera, camera_offset);
            }
          }
        });
      }
      void create(const fan::graphics::render_view_t& render_view, f32_t new_zoom = 2.f) {
        create(render_view.camera, render_view.viewport, new_zoom);
      }

      interactive_camera_t(
        fan::graphics::camera_t camera_nr = fan::graphics::get_orthographic_render_view().camera,
        fan::graphics::viewport_t viewport_nr = fan::graphics::get_orthographic_render_view().viewport,
        f32_t new_zoom = 1
      ) {
        create(camera_nr, viewport_nr, new_zoom);
      }

      interactive_camera_t(const fan::graphics::render_view_t& render_view, f32_t new_zoom = 2.f) :
        interactive_camera_t(render_view.camera, render_view.viewport, new_zoom) {

      }

      ~interactive_camera_t() {
        if (uc_nr.iic() == false) {
          fan::graphics::ctx().update_callback->unlrec(uc_nr);
          uc_nr.sic();
        }
      }

      fan::vec2 get_position() const {
        return camera_offset;
      }
      void set_position(const fan::vec2& position) {
        camera_offset = position;
        fan::graphics::camera_set_position(reference_camera, camera_offset);
        update();
      }
      void set_zoom(f32_t new_zoom) {
        zoom = new_zoom;
        update();
      }
      fan::vec2 get_size() const {
        return fan::graphics::g_render_context_handle->camera_get_size(
          fan::graphics::g_render_context_handle,
          reference_camera);
      }
      fan::vec2 get_viewport_size() const {
        return fan::graphics::viewport_get_size(reference_viewport);
      }


      f32_t zoom = 1;
      bool ignore = false;
      bool zoom_on_window_resize = true;
      bool pan_with_middle_mouse = false;
      bool clicked_inside_viewport = false;
      fan::vec2 old_window_size{};
      fan::vec2 camera_offset{};
      fan::graphics::camera_t reference_camera;
      fan::graphics::viewport_t reference_viewport;
      fan::window_t::resize_callback_NodeReference_t resize_callback_nr;
      fan::window_t::buttons_handle_t button_cb_nr;
      fan::window_t::mouse_motion_handle_t mouse_motion_nr;
      fan::graphics::update_callback_nr_t uc_nr;
    };

    struct animator_t {
      fan::vec2 prev_dir = 0;

      uint64_t animation_update_time = 150;//ms
      uint16_t i_down = 0, i_up = 0, i_left = 0, i_right = 0;

      template <std::size_t images_per_action>
      void process_walk(fan::graphics::shapes::shape_t& shape,
        const fan::vec2& vel,
        const std::array<fan::graphics::image_t, 4>& img_idle,
        const std::array<fan::graphics::image_t, images_per_action>& img_movement_left,
        const std::array<fan::graphics::image_t, images_per_action>& img_movement_right,
        const std::array<fan::graphics::image_t, images_per_action>& img_movement_up,
        const std::array<fan::graphics::image_t, images_per_action>& img_movement_down
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
        fan::graphics::image_t image;
      };

      fan::graphics::image_t root_image = fan::graphics::ctx().default_texture;
      //
      std::vector<std::vector<image_t>> images;
      //
      struct image_click_t {
        int highlight = 0;
        int count_index;
      };

      fan::graphics::texture_pack::internal_t::open_properties_t open_properties;
      fan::graphics::texture_pack::internal_t e;
      fan::graphics::texture_pack::internal_t::texture_properties_t texture_properties;

      //
      image_divider_t() {
        e.open(open_properties);
        texture_properties.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_edge;
        texture_properties.min_filter = fan::graphics::image_filter::nearest;
        texture_properties.mag_filter = fan::graphics::image_filter::nearest;
      }
    };

    fan::graphics::shapes::polygon_t::properties_t create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
      fan::graphics::shapes::polygon_t::properties_t pp;
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
      fan::graphics::update_callback_nr_t update_callback_nr;

      trail_t() {
        update_callback_nr = fan::graphics::ctx().update_callback->NewNodeLast();
        (*fan::graphics::ctx().update_callback)[update_callback_nr] = [this] (void* ptr) {
          update();
        };
      }
      ~trail_t() {
        if (!update_callback_nr) {
          return;
        }
        fan::graphics::ctx().update_callback->unlrec(update_callback_nr);
        update_callback_nr.sic();
      }

      void set_point(const fan::vec3& point, f32_t drift_intensity) {
        static fan::time::timer timer{ 300000000ULL, true };
        bool should_reset = trails.empty() || timer;

        if (should_reset) {
          trails.resize(trails.size() + 1);
          trails.back().vertices.clear();
          trails.back().creation_time = fan::time::now();
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
          trails.back().creation_time = fan::time::now();
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
        uint64_t current_time = fan::time::now();

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

    f32_t get_depth_from_y(const fan::vec2& position, f32_t tile_size_y) {
      return std::floor((position.y) / tile_size_y) + (0xFAAA - 2) / 2 + 18.f;
    }

    struct tilemap_t {

      tilemap_t() = default;
      tilemap_t(const fan::vec2& tile_size,
        const fan::color& color,
        const fan::vec2& area = fan::window::get_size(),
        const fan::vec2& offset = fan::vec2(0, 0),
        render_view_t* render_view = fan::graphics::g_render_context_handle.orthographic_render_view) {
        create(tile_size, color, area, offset, render_view);
      }

      fan::vec2i get_cell_count() {
        return fan::vec2i(
          shapes.empty() ? 0 : (int)shapes[0].size(),
          (int)shapes.size()
        );
      }

      template <typename T>
      fan::graphics::shapes::shape_t& operator[](const fan::vec2_wrap_t<T>& v) {
        return shapes[v.y][v.x];
      }

      void add_wall(const fan::vec2i& cell, fan::graphics::algorithm::pathfind::generator& gen) {
        if (wall_cells.find(cell) == wall_cells.end()) {
          gen.add_collision(cell);
          wall_cells.insert(cell);
          shapes[cell.y][cell.x].set_color(fan::colors::gray / 3);
        }
      }
      void remove_wall(const fan::vec2i& cell, fan::graphics::algorithm::pathfind::generator& gen) {
        if (wall_cells.find(cell) != wall_cells.end()) {
          gen.remove_collision(cell);
          wall_cells.erase(cell);
          shapes[cell.y][cell.x].set_color(fan::colors::gray);
        }
      }

      void reset_colors(const fan::color& color) {
        for (int i = 0; i < size.y; i++) {
          for (int j = 0; j < size.x; j++) {
            fan::vec2i cell(j, i);
            if (wall_cells.contains(cell)) {
              continue;
            }
            shapes[i][j].set_color(color);
          }
        }
      }
      void set_source(const fan::vec2i& cell, const fan::color& color) {
        if (!wall_cells.contains(cell) &&
          cell.x >= 0 && cell.x < shapes[0].size() &&
          cell.y >= 0 && cell.y < shapes.size()) {
          shapes[cell.y][cell.x].set_color(color);
        }
      }
      void set_destination(const fan::vec2i& cell, const fan::color& color) {
        if (!wall_cells.contains(cell) &&
          cell.x >= 0 && cell.x < shapes[0].size() &&
          cell.y >= 0 && cell.y < shapes.size()) {
          shapes[cell.y][cell.x].set_color(color);
        }
      }
      void highlight_path(
        const fan::graphics::algorithm::pathfind::coordinate_list& path,
        const fan::color& color
      ) {
        for (const auto& p : path) {
          if (!wall_cells.contains(p) &&
            p.x >= 0 && p.x < shapes[0].size() &&
            p.y >= 0 && p.y < shapes.size()) {
            shapes[p.y][p.x].set_color(color);
          }
        }
      }

      fan::graphics::algorithm::pathfind::coordinate_list find_path(
        const fan::vec2i& src,
        const fan::vec2i& dst,
        fan::graphics::algorithm::pathfind::generator& gen,
        fan::graphics::algorithm::pathfind::heuristic_function heuristic,
        bool diagonal
      ) {
        gen.set_heuristic(heuristic);
        gen.set_diagonal_movement(diagonal);
        return gen.find_path(src, dst);
      }

      void create(
        const fan::vec2& tile_size,
        const fan::color& color,
        const fan::vec2& area = fan::window::get_size(),
        const fan::vec2& offset = fan::vec2(0, 0),
        render_view_t* render_view = fan::graphics::g_render_context_handle.orthographic_render_view
      ) {
        fan::vec2 map_size(
          std::floor(area.x / tile_size.x),
          std::floor(area.y / tile_size.y)
        );

        size = map_size;
        this->tile_size = tile_size;
        positions.resize(map_size.y, std::vector<fan::vec2>(map_size.x));
        shapes.resize(map_size.y, std::vector<fan::graphics::shape_t>(map_size.x));

        for (int i = 0; i < map_size.y; i++) {
          for (int j = 0; j < map_size.x; j++) {
            positions[i][j] = offset + tile_size / 2 + fan::vec2(j * tile_size.x, i * tile_size.y);

            shapes[i][j] = fan::graphics::rectangle_t{ {
              .render_view = render_view,
              .position = fan::vec3(positions[i][j], 0),
              .size = tile_size / 2,
              .color = color
            } };
          }
        }
      }
      void set_tile_color(const fan::vec2i& pos, const fan::color& c) {
        if (pos.x < 0 || pos.x >= size.x) return;
        if (pos.y < 0 || pos.y >= size.y) return;
        shapes[pos.y][pos.x].set_color(c);
      }

      static constexpr f32_t circle_overlap(f32_t r, f32_t i0, f32_t i1) {
        if (i0 <= 0 && i1 >= 0) return r;
        f32_t y = std::min(std::min(fabs(i0), fabs(i1)) / r, 1.f);
        return std::sqrt(1.f - y * y) * r;
      }

      void highlight_circle(const fan::graphics::shapes::shape_t& circle,
        const fan::color& highlight_color) {
        fan::vec2 wp = circle.get_position();
        f32_t r = circle.get_radius();
        auto gi = fan::cast<sint32_t>(decltype(wp){});

        constexpr auto recurse = []<uint32_t d>(const auto& self,
          tilemap_t & tilemap,
          auto& gi,
          fan::vec2 wp,
          f32_t r,
          f32_t er,
          const fan::color & hc
        ) {
          f32_t cell_size = tilemap.tile_size[d];

          if constexpr (d + 1 < wp.size()) {
            gi[d] = (wp[d] - er) / cell_size;
            f32_t sp = (f32_t)gi[d] * cell_size;
            while (true) {
              f32_t rp = sp - wp[d];
              f32_t roff = circle_overlap(r, rp, rp + cell_size);
              self.template operator()<d + 1>(self, tilemap, gi, wp, r, roff, hc);
              gi[d]++;
              sp += cell_size;
              if (sp > wp[d] + er) break;
            }
          }
          else if constexpr (d < wp.size()) {
            gi[d] = (wp[d] - er) / cell_size;
            sint32_t to = (wp[d] + er) / cell_size;
            for (; gi[d] <= to; gi[d]++) {
              fan::vec2i pos{ gi[0], gi[1] };
              tilemap.set_tile_color(pos, hc);
            }
          }
        };

        recurse.template operator()<0>(recurse, *this, gi, wp, r, r, highlight_color);
      }


      void highlight_line(const fan::graphics::shapes::shape_t& line,
        const fan::color& color,
        render_view_t* render_view = fan::graphics::g_render_context_handle.orthographic_render_view) {
        fan::vec2 src = line.get_src();
        fan::vec2 dst = line.get_dst();

        auto raycast_positions = fan::graphics::algorithm::grid_raycast(
          { src, dst },
          tile_size
        );

        for (auto& pos : raycast_positions) {
          set_tile_color(pos, color);
        }
      }
      void highlight(const fan::graphics::shapes::shape_t& shape,
        const fan::color& color)
      {
        using namespace fan::graphics;

        switch (shape.get_shape_type()) {
        case shapes::shape_type_t::circle:
          highlight_circle(shape, color);
          break;
        case shapes::shape_type_t::line:
          highlight_line(shape, color);
          break;
        default:
          fan::throw_error("method not implemented");
          break;
        }
      }

      std::vector<std::vector<fan::vec2>> positions;
      std::vector<std::vector<fan::graphics::shapes::shape_t>> shapes;
      fan::vec2 size;
      fan::vec2 tile_size;

      struct vec2i_hash {
        size_t operator()(const fan::vec2i& v) const noexcept {
          return std::hash<int>()(v.x) ^ (std::hash<int>()(v.y) << 1);
        }
      };
      std::unordered_set<fan::vec2i, vec2i_hash> wall_cells;
    };



    struct terrain_palette_t {
      std::vector<std::pair<int, fan::color>> stops;

      constexpr terrain_palette_t() {
        stops = {
          { 50, fan::color::from_rgba(0x003eb2ff) }, // deep water
          { 80, fan::color::from_rgba(0x0952c6ff) }, // shallow water
          { 100, fan::color::from_rgba(0x726231ff) }, // coast
          { 150, fan::color::from_rgba(0xa49463ff) }, // lowlands
          { 200, fan::color::from_rgba(0x3c6114ff) }, // midlands
          { 250, fan::color::from_rgba(0x4f6b31ff) }, // highlands
          { 300, fan::color::from_rgba(0xffffffff) }  // snow
        };
      }

      fan::color get(int value) const {
        if (value <= stops.front().first) return stops.front().second;
        if (value >= stops.back().first) return stops.back().second;

        for (size_t i = 0; i < stops.size() - 1; ++i) {
          if (value >= stops[i].first && value <= stops[i + 1].first) {
            int v1 = stops[i].first;
            int v2 = stops[i + 1].first;
            auto c1 = stops[i].second;
            auto c2 = stops[i + 1].second;
            f32_t factor = (value - v1) / f32_t(v2 - v1);
            return c1.lerp(c2, factor);
          }
        }
        return fan::color::rgb(0, 0, 0);
      }
    };

    void generate_mesh(
      const vec2& noise_size,
      const std::vector<uint8_t>& noise_data,
      const fan::graphics::image_t& texture,
      std::vector<fan::graphics::shape_t>& out_mesh,
      const terrain_palette_t& palette) {
      fan::graphics::shapes::sprite_t::properties_t sp;
      sp.size = fan::window::get_size() / noise_size / 2;

      for (int i = 0; i < noise_size.y; ++i) {
        for (int j = 0; j < noise_size.x; ++j) {
          int index = (i * noise_size.x + j) * 3;
          int grayscale = noise_data[index];

          sp.position = fan::vec2(i, j) * sp.size * 2;
          sp.image = texture;
          sp.color = palette.get(grayscale);
          sp.color.a = 1;
          out_mesh.push_back(sp);
        }
      }
    }

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

export namespace fan::image {
  struct plane_split_t {
    void* planes[3]{};
    operator void** () {
      return planes;
    }
    operator const void* const* () const {
      return planes;
    }
  };

  plane_split_t plane_split(void* pixel_data, const fan::vec2ui& size, const fan::graphics::image_format& format) {
    plane_split_t result;
    uint64_t offset = 0;
    if (format == fan::graphics::image_format::yuv420p) {
      result.planes[0] = pixel_data;
      result.planes[1] = (uint8_t*)pixel_data + (offset += size.multiply());
      result.planes[2] = (uint8_t*)pixel_data + (offset += size.multiply() / 4);
    }
    else {
      fan::throw_error_impl("undefined");
    }
    return result;
  }
}

export namespace fan::graphics {
  struct tile_world_generator_t {

    bool is_solid(int x, int y) {
      if (x < 0 || x >= map_size.x || y < 0 || y >= map_size.y) {
        return true;
      }
      return tiles[x + y * map_size.x];
    }

    int count_neighbors(int x, int y) {
      int c = 0;
      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          if (is_solid(x + i, y + j)) {
            c++;
          }
        }
      }
      return c;
    }

    void iterate() {
      std::vector<bool> nt(map_size.x * map_size.y);
      for (int y = 0; y < map_size.y; y++) {
        for (int x = 0; x < map_size.x; x++) {
          nt[x + y * map_size.x] = count_neighbors(x, y) >= 5;
        }
      }
      tiles = std::move(nt);
    }

    void init_tile_world() {
      tiles.resize(map_size.x * map_size.y);
      for (int i = 0; i < map_size.x * map_size.y; i++) {
        tiles[i] = fan::random::value(0.f, 1.0f) < 0.45f;
      }
    }

    void init() {
      init_tile_world();
    }

    fan::vec2i map_size = 64;
    f32_t cell_size = 32;
    std::vector<bool> tiles;
  };
}

// graphics event wrappers
export namespace fan::event {
  template <typename derived_t>
  struct callback_awaiter : fan::event::condition_awaiter<derived_t> {
    template<typename promise_t>
    void await_suspend(std::coroutine_handle<promise_t> h) {
      auto* callbacks = fan::graphics::g_render_context_handle.update_callback;
      node = callbacks->NewNodeLast();
      (*callbacks)[node] = [this, h](void*) {
        if (static_cast<const derived_t*>(this)->check_condition()) {
          unlink();
          fan::event::schedule_resume(h);
        }
      };
    }
    void unlink() {
      if (node) {
        fan::graphics::g_render_context_handle.update_callback->unlrec(node);
        node.sic();
      }
    }
    ~callback_awaiter() {
      unlink();
    }
  private:
    fan::graphics::update_callback_nr_t node;
  };
}

// graphics awaiters
export namespace fan::graphics {
  struct animation_frame_awaiter : fan::event::callback_awaiter<animation_frame_awaiter> {
    animation_frame_awaiter(
      fan::graphics::shapes::shape_t* sprite_,
      const std::string& anim_,
      int frame_
    ) : sprite(sprite_), animation_name(anim_), target_frame(frame_) {
    }
    bool check_condition() const {
      return sprite && sprite->get_current_animation_frame() >= (target_frame - 1) &&
        sprite->get_current_animation().name == animation_name;
    }
    fan::graphics::shapes::shape_t* sprite = nullptr;
    std::string animation_name;
    int target_frame = 0;
  };
}