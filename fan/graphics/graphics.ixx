module;
// extension to loco.ixx

#include <fan/graphics/opengl/init.h>
#include <fan/event/types.h>

#include <coroutine>
#include <source_location>
#include <unordered_set>
#include <algorithm>
#include <cmath>

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
export import fan.math;
#if defined(FAN_GUI)
  export import fan.graphics.gui.text_logger;
#endif

#if defined(FAN_JSON)
  import fan.types.json;
#endif

import fan.random;
import fan.graphics.opengl.core;

import fan.physics.types;

// user friendly functions
/***************************************/
//
export namespace fan::window {
  void add_input_action(const int* keys, std::size_t count, const std::string& action_name);
  void add_input_action(std::initializer_list<int> keys, const std::string& action_name);
  void add_input_action(int key, const std::string& action_name);
  bool is_input_action_active(const std::string& action_name, int pstate = fan::window::input_action_t::press);
  bool is_action_clicked(const std::string& action_name);
  bool is_action_down(const std::string& action_name);
  bool exists(const std::string& action_name);
}
export namespace fan::graphics {
#if defined(FAN_2D)
  using vfi_t = fan::graphics::shapes::vfi_t;
  using shape_t = fan::graphics::shapes::shape_t;
  using shape_type_t = fan::graphics::shapes::shape_type_t;
#endif
  using renderer_t = fan::window_t::renderer_t;
  extern fan::graphics::image_t invalid_image;
  fan::graphics::render_view_t add_render_view();
  fan::graphics::render_view_t add_render_view(const fan::vec2& ortho_x, const fan::vec2& ortho_y, const fan::vec2& viewport_position, const fan::vec2& viewport_size);
}

export namespace fan {
  void printclnn(auto&&... values) {
  #if defined (FAN_GUI)
    ([&](const auto& value) {
      std::ostringstream oss;
      oss << value;
      fan::graphics::ctx().console->print(oss.str() + " ", 0);
    }(values), ...);
  #endif
  }
  void printcl(auto&&... values) {
  #if defined(FAN_GUI)
    printclnn(values...);
    fan::graphics::ctx().console->print("\n", 0);
  #endif
  }

  void printclnnh(int highlight, auto&&... values) {
  #if defined(FAN_GUI)
    ([&](const auto& value) {
      std::ostringstream oss;
      oss << value;
      fan::graphics::ctx().console->print(oss.str() + " ", highlight);
    }(values), ...);
  #endif
  }

  void printclh(int highlight, auto&&... values) {
  #if defined(FAN_GUI)
    printclnnh(highlight, values...);
    fan::graphics::ctx().console->print("\n", highlight);
  #endif
  }
  inline void printcl_err(auto&&... values) {
  #if defined(FAN_GUI)
    printclh(fan::graphics::highlight_e::error, values...);
  #endif
  }
  inline void printcl_warn(auto&&... values) {
  #if defined(FAN_GUI)
    printclh(fan::graphics::highlight_e::warning, values...);
  #endif
  }
}

bool init_fan_track_opengl_print = []() {
  fan_opengl_track_print() = [](std::string func_name, uint64_t elapsed) {
    fan::printclnnh(fan::graphics::highlight_e::text, func_name + ":");
    fan::printclh(fan::graphics::highlight_e::warning, std::to_string(elapsed / 1e+6f)/*fan::to_string(elapsed / 1e+6)*/ + "ms");
  };
  return 1;
}();

export namespace fan::graphics {
  namespace image_presets {
    image_load_properties_t pixel_art() {
      image_load_properties_t props;
      props.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_border;
      props.min_filter = image_filter::nearest;
      props.mag_filter = image_filter::nearest;
      return props;
    }
    image_load_properties_t pixel_art_repeat() {
      auto props = pixel_art();
      props.visual_output = fan::graphics::image_sampler_address_mode::repeat;
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

  std::vector<uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp = 0, fan::vec2 uvs = 1);
  fan::graphics::image_nr_t image_create();
  fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr);
  uint64_t image_get_handle(fan::graphics::image_nr_t nr);
  void image_erase(fan::graphics::image_nr_t nr);
  void image_bind(fan::graphics::image_nr_t nr);
  void image_unbind(fan::graphics::image_nr_t nr);
  fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr);
  void image_set_settings(fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings);
  fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info);
  fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_nr_t image_load(const std::string& path, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_nr_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size);
  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p);
  void image_unload(fan::graphics::image_nr_t nr);
  bool is_image_valid(fan::graphics::image_nr_t nr);
  fan::graphics::image_t image_load_pixel_art(const std::string& path);
  fan::graphics::image_t image_load_smooth(const std::string& path);
  fan::graphics::image_nr_t create_missing_texture();
  fan::graphics::image_nr_t create_transparent_texture();
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info);
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p);
  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path = std::source_location::current());
  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_nr_t image_create(const fan::color& color);
  fan::graphics::image_nr_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p);
  std::vector<uint8_t> read_pixels(fan::graphics::image_nr_t nr, const fan::vec2& position, const fan::vec2& size);
  std::vector<uint8_t> read_pixels_from_image(fan::graphics::image_nr_t nr, const fan::vec2& uv_position = 0, const fan::vec2& uv_size = 1);

  fan::graphics::shader_nr_t shader_create();
  void shader_erase(fan::graphics::shader_nr_t nr);
  void shader_use(fan::graphics::shader_nr_t nr);
  void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code);
  void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code);
  bool shader_compile(fan::graphics::shader_nr_t nr);
#if defined(FAN_2D)
  fan::graphics::shader_nr_t shader_get_nr(uint16_t shape_type);
  fan::graphics::shader_list_t::nd_t& shader_get_data(uint16_t shape_type);
  bool shader_update_fragment(uint16_t shape_type, const std::string& fragment);
#endif

  fan::graphics::camera_nr_t camera_create();
  fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr = fan::graphics::get_orthographic_render_view().camera);
  void camera_erase(fan::graphics::camera_nr_t nr);
  fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y);
  fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y);
  fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr);
  void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
  fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr);
  fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr = fan::graphics::get_orthographic_render_view().viewport);
  f32_t camera_get_zoom(fan::graphics::camera_nr_t nr);
  void camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom);
  void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y);
  void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size);
  void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset);
  void camera_set_target(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed = 10);
  void camera_look_at(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed = 10.f);
  void camera_look_at(const fan::vec2& target, f32_t move_speed = 10.f);

  fan::graphics::viewport_nr_t viewport_create();
  fan::graphics::viewport_nr_t viewport_create(const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr);
  void viewport_erase(fan::graphics::viewport_nr_t nr);
  fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr);
  void viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  void viewport_zero(fan::graphics::viewport_nr_t nr);
  bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
  bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
  bool inside(const fan::graphics::render_view_t& render_view, const fan::vec2& position);
  bool is_mouse_inside(const fan::graphics::render_view_t& render_view);

#if defined(FAN_2D)

  using sprite_flags_e = fan::graphics::sprite_flags_e;

  struct light_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(0, 0, 0);
    f32_t parallax_factor = 0;
    fan::vec2 size = fan::vec2(0.1, 0.1);
    fan::vec2 rotation_point = fan::vec2(0, 0);
    fan::color color = fan::color(1, 1, 1, 1);
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    uint32_t flags = 0;
    fan::vec3 angle = fan::vec3(0, 0, 0);
    bool enable_culling = true;
  };

  struct light_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    light_t() = default;
    light_t(light_properties_t p);
    light_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color = fan::colors::white, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  };

  struct line_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 src = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
    fan::vec2 dst = fan::vec2(1, 1);
    fan::color color = fan::color(1, 1, 1, 1);
    f32_t thickness = 4.0f;
    bool blending = true;
    uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    bool enable_culling = true;
  };

  struct line_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    line_t() = default;
    line_t(line_properties_t p);
    line_t(const fan::vec3& src, const fan::vec3& dst, const fan::color& color = fan::colors::white, f32_t thickness = 3.f, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  };

  struct rectangle_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
    fan::vec2 size = fan::vec2(32, 32);
    fan::color color = fan::color(1, 1, 1, 1);
    fan::color outline_color = color;
    fan::vec3 angle = 0;
    fan::vec2 rotation_point = 0;
    bool blending = true;
    bool enable_culling = true;
  };

  struct rectangle_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    rectangle_t() = default;
    rectangle_t(rectangle_properties_t p);
    rectangle_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color = fan::colors::white, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  };

  struct sprite_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
    fan::vec2 size = fan::vec2(32, 32);
    fan::vec3 angle = 0;
    fan::color color = fan::color(1, 1, 1, 1);
    fan::vec2 rotation_point = 0;
    fan::graphics::image_t image = fan::graphics::ctx().default_texture;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
    std::array<fan::graphics::image_t, 30> images;
    f32_t parallax_factor = 0;
    bool blending = true;
    uint32_t flags = sprite_flags_e::circle | sprite_flags_e::multiplicative;
    fan::graphics::texture_pack::unique_t texture_pack_unique_id;
    bool enable_culling = true;
  };

  struct sprite_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    sprite_t() = default;
    sprite_t(sprite_properties_t p);
    sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  };

  struct unlit_sprite_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
    fan::vec2 size = fan::vec2(32, 32);
    fan::vec3 angle = 0;
    fan::color color = fan::color(1, 1, 1, 1);
    fan::vec2 rotation_point = 0;
    fan::graphics::image_t image = fan::graphics::ctx().default_texture;
    std::array<fan::graphics::image_t, 30> images;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
    bool blending = true;
    bool enable_culling = true;
  };

  struct unlit_sprite_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    unlit_sprite_t() = default;
    unlit_sprite_t(unlit_sprite_properties_t p);
    unlit_sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  };


  struct circle_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
    f32_t radius = 32.f;
    fan::vec3 angle = 0;
    fan::color color = fan::color(1, 1, 1, 1);
    bool blending = true;
    uint32_t flags = 0;
    bool enable_culling = true;
  };

  struct circle_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    circle_t() = default;
    circle_t(circle_properties_t p);
    circle_t(const fan::vec3& position, f32_t radius, const fan::color& color = fan::colors::white, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  };

  struct capsule_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
    fan::vec2 center0 = 0;
    fan::vec2 center1 {0, 128.f};
    f32_t radius = 64.0f;
    fan::vec3 angle = 0.f;
    fan::color color = fan::color(1, 1, 1, 1);
    fan::color outline_color = color;
    bool blending = true;
    uint32_t flags = 0;
    bool enable_culling = true;
  };

  struct capsule_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    capsule_t() = default;
    capsule_t(capsule_properties_t p);
  };

  struct polygon_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(0, 0, 0);
    std::vector<vertex_t> vertices;
    fan::vec3 angle = 0;
    fan::vec2 rotation_point = 0;
    bool blending = true;
    uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_strip;
    uint32_t vertex_count = 3;
    bool enable_culling = true;
  };

  struct polygon_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    polygon_t() = default;
    polygon_t(polygon_properties_t p);
  };

  struct grid_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(0, 0, 0);
    fan::vec2 size = fan::vec2(0.1, 0.1);
    fan::vec2 grid_size = fan::vec2(1, 1);
    fan::vec2 rotation_point = fan::vec2(0, 0);
    fan::color color = fan::color(1, 1, 1, 1);
    fan::vec3 angle = fan::vec3(0, 0, 0);
    bool enable_culling = true;
  };
  struct grid_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    grid_t() = default;
    grid_t(grid_properties_t p);
  };

  struct universal_image_renderer_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = 0;
    fan::vec2 size = 0;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;

    bool blending = true;

    std::array<fan::graphics::image_t, 4> images = {
      fan::graphics::ctx().default_texture,
      fan::graphics::ctx().default_texture,
      fan::graphics::ctx().default_texture,
      fan::graphics::ctx().default_texture
    };
    uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    bool enable_culling = true;
  };

  struct universal_image_renderer_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    universal_image_renderer_t() = default;
    universal_image_renderer_t(const universal_image_renderer_properties_t& p);
  };

  struct gradient_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;

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
    bool enable_culling = true;
  };

  struct gradient_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    gradient_t() = default;
    gradient_t(const gradient_properties_t& p);
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
    bool enable_culling = true;
  };

  struct shader_shape_t : fan::graphics::shapes::shape_t {
    shader_shape_t() = default;
    shader_shape_t(const shader_shape_properties_t& p);
  };

  struct shadow_properties_t {
    const render_view_t* render_view = fan::graphics::ctx().orthographic_render_view;
    fan::vec3 position = fan::vec3(0, 0, 0);
    int shape = fan::graphics::shapes::shadow_t::rectangle;
    fan::vec2 size = fan::vec2(0.1, 0.1);
    fan::vec2 rotation_point = fan::vec2(0, 0);
    fan::color color = fan::color(1, 1, 1, 1);
    uint32_t flags = 0;
    fan::vec3 angle = fan::vec3(0, 0, 0);
    fan::vec2 light_position = fan::vec2(0, 0);
    f32_t light_radius = 100.f;
    bool enable_culling = true;
  };

  struct shadow_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;

    shadow_t() = default;
    shadow_t(shadow_properties_t p = shadow_properties_t());
  };


#if defined(FAN_3D)
  struct line3d_properties_t {
    render_view_t* render_view = fan::graphics::ctx().perspective_render_view;
    fan::vec3 src = fan::vec3(0, 0, 0);
    fan::vec3 dst = fan::vec3(10, 10, 10);
    fan::color color = fan::color(1, 1, 1, 1);
    bool blending = true;
  };

  struct line3d_t : fan::graphics::shapes::shape_t {
    using fan::graphics::shapes::shape_t::shape_t;
    using fan::graphics::shapes::shape_t::operator=;
    
    
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
      : center(c), half_size(hsize), color(col), depth(d) {
      fan::vec3 bl(center.x - half_size.x, center.y - half_size.y, depth);
      fan::vec3 br(center.x + half_size.x, center.y - half_size.y, depth);
      fan::vec3 tr(center.x + half_size.x, center.y + half_size.y, depth);
      fan::vec3 tl(center.x - half_size.x, center.y + half_size.y, depth);

      edges[0] = line_t(line_properties_t {.src = bl, .dst = br, .color = color});
      edges[1] = line_t(line_properties_t {.src = br, .dst = tr, .color = color});
      edges[2] = line_t(line_properties_t {.src = tr, .dst = tl, .color = color});
      edges[3] = line_t(line_properties_t {.src = tl, .dst = bl, .color = color});
    }
  };

  fan::graphics::shapes::shape_t& add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s);
  uint32_t add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s);
  void remove_static_shape_draw(const fan::graphics::shapes::shape_t& s);
  fan::graphics::shapes::shape_t& rectangle(const rectangle_properties_t& props = {});
  fan::graphics::shapes::shape_t& rectangle(const fan::vec3& position, const fan::vec2& size, const fan::color& color = fan::colors::white, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  fan::graphics::shapes::shape_t& sprite(const sprite_properties_t& props = {});
  fan::graphics::shapes::shape_t& unlit_sprite(const unlit_sprite_properties_t& props = {});
  fan::graphics::shapes::shape_t& line(const line_properties_t& props = {});
  fan::graphics::shapes::shape_t& line(const fan::vec3& src, const fan::vec3& dst, const fan::color& color = fan::colors::white, f32_t thickness = line_properties_t().thickness, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  fan::graphics::shapes::shape_t& light(const light_properties_t& props = {});
  fan::graphics::shapes::shape_t& circle(const circle_properties_t& props = {});
  fan::graphics::shapes::shape_t& circle(const fan::vec3& position, f32_t radius, const fan::color& color = fan::colors::white, render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
  fan::graphics::shapes::shape_t& capsule(const capsule_properties_t& props = {});
  fan::graphics::shapes::shape_t& polygon(const polygon_properties_t& props = {});
  fan::graphics::shapes::shape_t& grid(const grid_properties_t& props = {});
#if defined(FAN_PHYSICS_2D)
  void aabb(const fan::physics::aabb_t& b, f32_t depth = 55000, const fan::color& c = fan::color(1, 0, 0, 1), f32_t thickness = line_properties_t().thickness, render_view_t* render_view = &fan::graphics::get_orthographic_render_view());
  void aabb(const fan::graphics::shapes::shape_t& s, f32_t depth = 55000, const fan::color& c = fan::color(1, 0, 0, 1), f32_t thickness = line_properties_t().thickness, render_view_t* render_view = &fan::graphics::get_orthographic_render_view());
  void aabb(const fan::vec2& min, const fan::vec2& max, f32_t depth, const fan::color& c = fan::colors::white, f32_t thickness = line_properties_t().thickness, render_view_t* render_view = &fan::graphics::get_orthographic_render_view());
  void aabb(const fan::vec2& min, const fan::vec2& max, f32_t thickness = line_properties_t().thickness, render_view_t* render_view = &fan::graphics::get_orthographic_render_view());
#endif

  struct sprite_sheet_config_t {
    std::string path;
    bool loop = true;
    bool start = true;
  };

#if defined(FAN_JSON)

  fan::graphics::shape_t shape_from_json(
    const std::string& json_path,
    const std::source_location& callers_path = std::source_location::current()
  );

  void resolve_json_image_paths(
    fan::json& out,
    const std::string& json_path,
    const std::source_location& callers_path = std::source_location::current()
  );
  fan::graphics::sprite_t sprite_sheet_from_json(
    const sprite_sheet_config_t flags,
    const std::source_location& callers_path = std::source_location::current()
  );
#endif

  fan::graphics::shapes::polygon_t::properties_t create_hexagon(f32_t radius, const fan::color& color = fan::colors::white);
  // for line
  fan::line3 get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index);

  // REQUIRES to be allocated by new since lambda captures this
    // also container that it's stored in, must not change pointers
  template <typename T>
  struct vfi_root_custom_t {
    using shape_t = fan::graphics::shapes::shape_t;

    void enable_highlight() {
      apply_highlight([](auto& h, const fan::line3& line, fan::graphics::render_view_t& rv) {
        h = fan::graphics::line_t {{&rv, line[0], line[1], fan::color(1, 0.5, 0, 1)}};
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
            selected_objects = {this};
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
            new_size = {1.0f, 1.0f / aspect_ratio};
          }
          if (new_size.y < 1.0f) {
            new_size = {aspect_ratio, 1.0f};
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
      children.push_back({shape});
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

    fan::color get_color() const {
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
      fan::graphics::render_view_t rv;
      rv.camera = children[0].get_camera();
      rv.viewport = children[0].get_viewport();
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
    std::vector<std::array<shape_t, 4>> highlight {1};
  };

  using vfi_root_t = vfi_root_custom_t<__empty_struct>;
  //#endif

#endif
}

export namespace fan::graphics {
  struct interactive_camera_t {
    interactive_camera_t(const interactive_camera_t&) = delete;
    interactive_camera_t(interactive_camera_t&&) = delete;

    interactive_camera_t& operator=(interactive_camera_t&&) = delete;
    interactive_camera_t& operator=(const interactive_camera_t&) = delete;

    operator render_view_t*();

    void reset();
    void reset_view();
    void update();
    void create(
      fan::graphics::camera_t camera_nr = fan::graphics::get_orthographic_render_view().camera,
      fan::graphics::viewport_t viewport_nr = fan::graphics::get_orthographic_render_view().viewport,
      f32_t new_zoom = 1.f,
      const fan::vec2& initial_pos = -0xFAFA
    );
    void create(const fan::graphics::render_view_t& render_view, f32_t new_zoom = 1.f);
    void create_default(f32_t zoom = 1.f);
    interactive_camera_t(f32_t zoom); // calls create_default
    interactive_camera_t(
      fan::graphics::camera_t camera_nr = fan::graphics::get_orthographic_render_view().camera,
      fan::graphics::viewport_t viewport_nr = fan::graphics::get_orthographic_render_view().viewport,
      f32_t new_zoom = 1,
      const fan::vec2& initial_pos = -0xFAFA
    );
    interactive_camera_t(const fan::graphics::render_view_t& render_view, f32_t new_zoom = 1.f);
    ~interactive_camera_t();

    fan::vec2 get_initial_position() const;
    void set_initial_position(const fan::vec2& position);
    fan::vec2 get_position() const;
    void set_position(const fan::vec2& position);
    f32_t get_zoom() const;
    void set_zoom(f32_t new_zoom);
    fan::vec2 get_size() const;
    fan::vec4 get_ortho() const;
    fan::vec2 get_viewport_size() const;

    fan::vec2 old_window_size {};
    fan::vec2 camera_offset {};
    fan::graphics::render_view_t render_view;
    fan::window_t::resize_handle_t resize_callback_nr;
    fan::window_t::buttons_handle_t button_cb_nr;
    fan::window_t::mouse_motion_handle_t mouse_motion_nr;
    fan::graphics::update_callback_nr_t uc_nr;
    fan::vec2 initial_position = 0;
    bool ignore_input = false;
    bool zoom_on_window_resize = true;
    bool pan_with_middle_mouse = true;
    bool clicked_inside_viewport = false;
  };

  struct world_window_t {
    world_window_t();
    void update(
      const fan::vec2& viewport_pos = 0, 
      const fan::vec2& viewport_size = fan::window::get_size()
    );
    operator render_view_t*();
    render_view_t render_view;
    fan::graphics::interactive_camera_t cam;
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
    image_divider_t();
  };

#if defined(FAN_2D)

  fan::graphics::shapes::polygon_t::properties_t create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth);
  std::vector<fan::vec2> ground_points(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth);

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

    trail_t();
    ~trail_t();

    void set_point(const fan::vec3& point, f32_t drift_intensity);
    void update();
  };

  f32_t get_depth_from_y(const fan::vec2& position, f32_t tile_size_y);

  struct tilemap_t {
    tilemap_t() = default;
    tilemap_t(const fan::vec2& tile_size,
      const fan::color& color,
      const fan::vec2& area = fan::window::get_size(),
      const fan::vec2& offset = fan::vec2(0, 0),
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);

    fan::vec2i get_cell_count();

    template <typename T>
    fan::graphics::shapes::shape_t& operator[](const fan::vec2_wrap_t<T>& v);

    void add_wall(const fan::vec2i& cell, fan::graphics::algorithm::pathfind::generator& gen);
    void remove_wall(const fan::vec2i& cell, fan::graphics::algorithm::pathfind::generator& gen);
    void reset_colors(const fan::color& color);
    void set_source(const fan::vec2i& cell, const fan::color& color);
    void set_destination(const fan::vec2i& cell, const fan::color& color);
    void highlight_path(
      const fan::graphics::algorithm::pathfind::coordinate_list& path,
      const fan::color& color
    );
    fan::graphics::algorithm::pathfind::coordinate_list find_path(
      const fan::vec2i& src,
      const fan::vec2i& dst,
      fan::graphics::algorithm::pathfind::generator& gen,
      fan::graphics::algorithm::pathfind::heuristic_function heuristic,
      bool diagonal
    );
    void create(
      const fan::vec2& tile_size,
      const fan::color& color,
      const fan::vec2& area = fan::window::get_size(),
      const fan::vec2& offset = fan::vec2(0, 0),
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view
    );
    void set_tile_color(const fan::vec2i& pos, const fan::color& c);
    static constexpr f32_t circle_overlap(f32_t r, f32_t i0, f32_t i1);
    void highlight_circle(const fan::graphics::shapes::shape_t& circle,
      const fan::color& highlight_color);
    void highlight_line(const fan::graphics::shapes::shape_t& line,
      const fan::color& color,
      render_view_t* render_view = fan::graphics::ctx().orthographic_render_view);
    void highlight(const fan::graphics::shapes::shape_t& shape,
      const fan::color& color);

    std::vector<std::vector<fan::vec2>> positions;
    std::vector<std::vector<fan::graphics::shapes::shape_t>> shapes;
    fan::vec2 size;
    fan::vec2 tile_size;

    std::unordered_set<fan::vec2i> wall_cells;
  };

  struct terrain_palette_t {
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

    fan::color get(int value) const;
    std::vector<std::pair<int, fan::color>> stops;
  };

  void generate_mesh(
    const vec2& noise_size,
    const std::vector<uint8_t>& noise_data,
    const fan::graphics::image_t& texture,
    std::vector<fan::graphics::shape_t>& out_mesh,
    const terrain_palette_t& palette,
    const sprite_properties_t& cp = {}
  );

  fan::event::task_t async_generate_mesh(
    const vec2& noise_size,
    const std::vector<uint8_t>& noise_data,
    const fan::graphics::image_t& texture,
    std::vector<fan::graphics::shape_t>& out_mesh,
    const terrain_palette_t& palette,
    const sprite_properties_t& cp = {}
  );
#endif
}

export namespace fan {
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

export namespace fan::image {
  struct plane_split_t {
    void* planes[3] {};
    operator void** () {
      return planes;
    }
    operator const void* const* () const {
      return planes;
    }
  };

  plane_split_t plane_split(void* pixel_data, const fan::vec2ui& size, const fan::graphics::image_format& format);
}

#if defined(FAN_2D)

export namespace fan::graphics {
  struct tile_world_generator_t {
    bool is_solid(int x, int y);
    int count_neighbors(int x, int y);
    void iterate();
    void init_tile_world();
    void init();

    fan::vec2i map_size = 64;
    f32_t cell_size = 32;
    std::vector<bool> tiles;
  };

  namespace effects {
    struct particle_pool_t {
      template<size_t N>
      struct pool_t {
        fan::graphics::shape_t particles[N];
        size_t current_index = 0;

        void from_json(
          const std::string& path,
          const std::source_location& callers_path = std::source_location::current()
        ) {
          auto base = fan::graphics::shape_from_json(path, callers_path);
          base.stop_particles();
          base.set_dynamic();
          std::fill(std::begin(particles), std::end(particles), base);
        }

        void spawn_at(const fan::vec3& position) {
          particles[current_index].start_particles();
          particles[current_index].set_position(position);
          current_index = (current_index + 1) % N;
        }
      };
    };
  }
}

#endif

// graphics event wrappers
export namespace fan::event {
  template <typename derived_t>
  struct callback_awaiter : fan::event::condition_awaiter<derived_t> {
    template<typename promise_t>
    void await_suspend(std::coroutine_handle<promise_t> h) {
      auto* callbacks = fan::graphics::ctx().update_callback;
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
        fan::graphics::ctx().update_callback->unlrec(node);
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

#if defined(FAN_2D)


export namespace fan {

  enum class ease_e {
    linear,
    sine,
    pulse,
    ease_in,
    ease_out
  };

  f32_t apply_ease(ease_e easing, f32_t t);

  template <typename T>
  struct transition_t {
    ~transition_t() {
      loop = false;
    }

    fan::event::task_t animate(std::function<void(const T&)> callback) {
      if (on_start) {
        on_start();
      }

      f32_t elapsed = 0.f;

      do {
        while (elapsed < duration) {
          f32_t t = std::fmod((elapsed / duration) + phase_offset, 1.f);
          t = fan::apply_ease(easing, t);
          callback(lerp(from, to, t));
          co_await fan::graphics::co_next_frame();
          elapsed += fan::graphics::get_window().m_delta_time;
        }

        elapsed = 0.f;
        co_await fan::graphics::co_next_frame();
      } while (loop);

      callback(to);

      if (on_end) {
        on_end();
      }
    }

    T from;
    T to;
    f32_t duration;
    f32_t phase_offset = 0.f;
    std::function<T(const T&, const T&, f32_t)> lerp;
    std::function<void()> on_start = {};
    std::function<void()> on_end = {};
    bool loop = false;
    ease_e easing = ease_e::sine;
  };

  template <typename T>
  struct auto_transition_t : transition_t<T> {
    using base_t = transition_t<T>;
    fan::event::task_t task;
    std::function<void(const T&)> callback;
    bool active = false;

    void setup_lerp() {
      if constexpr (requires (T a, T b, f32_t t) { a.lerp(b, t); }) {
        base_t::lerp = [](const T& a, const T& b, f32_t t) {
          return a.lerp(b, t);
        };
      }
      else {
        base_t::lerp = [](const T& a, const T& b, f32_t t) {
          return a + (b - a) * t;
        };
      }
    }

    void start(
      const T& from,
      const T& to,
      f32_t duration,
      std::function<void(const T&)> cb,
      ease_e easing = ease_e::pulse
    ) {
      if (active) {
        return;
      }

      base_t::from = from;
      base_t::to = to;
      base_t::duration = duration;
      base_t::phase_offset = fan::random::value(0.f, 1.f);
      base_t::loop = true;
      base_t::easing = easing;

      setup_lerp();

      callback = cb;
      task = base_t::animate(callback);
      active = true;
    }

    void start_once(
      const T& from,
      const T& to,
      f32_t duration,
      std::function<void(const T&)> cb
    ) {
      if (active) {
        return;
      }

      base_t::from = from;
      base_t::to = to;
      base_t::duration = duration;
      base_t::phase_offset = 0.f;
      base_t::loop = false;
      base_t::easing = ease_e::linear;

      setup_lerp();

      callback = cb;
      task = base_t::animate(callback);
      active = true;
    }

    void stop(const T& reset_to) {
      if (!active) {
        return;
      }

      task = {};
      callback(reset_to);
      active = false;
    }

    fan::event::task_t animate(std::function<void(const T&)> cb) {
      return base_t::animate(cb);
    }
  };

  using color_transition_t = transition_t<fan::color>;
  using auto_color_transition_t = auto_transition_t<fan::color>;
  using vec2_transition_t = transition_t<fan::vec2>;
  using auto_vec2_transition_t = auto_transition_t<fan::vec2>;

  auto_color_transition_t pulse_red(f32_t duration = 1.f);
  color_transition_t fade_out(f32_t duration);
  vec2_transition_t move_linear(const fan::vec2& from, const fan::vec2& to, f32_t duration);
  vec2_transition_t move_pingpong(const fan::vec2& from, const fan::vec2& to, f32_t duration);

}

// graphics awaiters
export namespace fan::graphics {
  struct animation_frame_awaiter : fan::event::callback_awaiter<animation_frame_awaiter> {
    animation_frame_awaiter(
      fan::graphics::shapes::shape_t* sprite_,
      const std::string& anim_,
      int frame_
    );

    bool check_condition() const;

    fan::graphics::shapes::shape_t* sprite = nullptr;
    std::string animation_name;
    int target_frame = 0;
  };
}

#endif