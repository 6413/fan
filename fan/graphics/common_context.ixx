module;

#include <fan/utility.h>

#include <fan/graphics/common_context_functions_declare.h>

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <source_location>
#include <functional>

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
#endif

export module fan.graphics.common_context;

export import fan.types.color;
export import fan.graphics.image_load;
export import fan.camera;

import fan.window;
import fan.window.input_action;
import fan.print;
import fan.utility;

#if defined(fan_gui)
  import fan.console;
  import fan.graphics.gui.text_logger;
#endif

export namespace fan {
  namespace graphics {
    enum image_format {
      undefined = -1,
      r8b8g8a8_unorm,
      b8g8r8a8_unorm,
      r8_unorm,
      rg8_unorm,
      rgb_unorm,
      rgba_unorm,
      bgr_unorm,
      r8_uint,
      r8g8b8a8_srgb,
      r11f_g11f_b10f,
      yuv420p,
      nv12,
    };
    constexpr uint8_t get_texture_amount(uint8_t format) {
      switch (format) {
      case undefined: {
        return 0;
      }
      case yuv420p: {
        return 3;
      }
      case nv12: {
        return 2;
      }
      default: {
        fan::throw_error("invalid format");
        return undefined;
      }
      }
    }
    enum image_sampler_address_mode {
      repeat,
      mirrored_repeat,
      clamp_to_edge,
      clamp_to_border,
      mirrored_clamp_to_edge,
    };
    enum image_filter {
      nearest,
      linear,
      nearest_mipmap_nearest,
      linear_mipmap_nearest,
      nearest_mipmap_linear,
      linear_mipmap_linear,
    };
    enum data_types {
      fan_unsigned_byte,
      fan_byte,
      fan_unsigned_int,
      fan_float,
    };
    struct image_load_properties_defaults {
      static constexpr uint32_t visual_output = repeat;
      static constexpr uint32_t internal_format = r8b8g8a8_unorm;
      static constexpr uint32_t format = r8b8g8a8_unorm;
      static constexpr uint32_t type = fan_unsigned_byte; // internal
      static constexpr uint32_t min_filter = linear;
      static constexpr uint32_t mag_filter = linear;
    };

    struct context_camera_t : fan::camera {
      fan::mat4 m_projection = fan::mat4(1);
      fan::mat4 m_view = fan::mat4(1);
      f32_t zfar = 1000.f;
      f32_t znear = 0.1f;

      union {
        struct {
          f32_t left;
          f32_t right;
          f32_t up;
          f32_t down;
        };
        fan::vec4 v;
      }coordinates;
    };

    struct context_viewport_t {
      fan::vec2 viewport_position;
      fan::vec2 viewport_size;
    };

    struct image_load_properties_t {
      uint32_t visual_output = image_load_properties_defaults::visual_output;
      uintptr_t internal_format = image_load_properties_defaults::internal_format;
      uintptr_t format = image_load_properties_defaults::format;
      uintptr_t type = image_load_properties_defaults::type;
      uintptr_t min_filter = image_load_properties_defaults::min_filter;
      uintptr_t mag_filter = image_load_properties_defaults::mag_filter;
    };

    struct image_data_t {
      fan::vec2 size;
      std::string image_path;
      image_load_properties_t image_settings;
      void* internal;
    };


    typedef uint8_t* (*this_offset_camera_list_t)(uint8_t* context);
    typedef uint8_t* (*this_offset_shader_list_t)(uint8_t* context);
    typedef uint8_t* (*this_offset_image_list_t)(uint8_t* context);
    typedef uint8_t* (*this_offset_viewport_list_t)(uint8_t* context);

    inline this_offset_camera_list_t get_camera_list;
    inline this_offset_shader_list_t get_shader_list;
    inline this_offset_image_list_t get_image_list;
    inline this_offset_viewport_list_t get_viewport_list;

    struct shader_data_t {
      std::string svertex, sfragment;
      std::unordered_map<std::string, std::string> uniform_type_table;
      void* internal;
    };

    constexpr f32_t znearfar = 0xffff;

    struct primitive_topology_t {
      enum {
        points,
        lines,
        line_strip,
        line_loop,
        triangles,
        triangle_strip,
        triangle_fan,
        lines_with_adjacency,
        line_strip_with_adjacency,
        triangles_with_adjacency,
        triangle_strip_with_adjacency,
      };
    };

  };
}

namespace bll_builds {
#include <fan/fan_bll_preset.h>
#define BLL_set_prefix camera_list
#define BLL_set_type_node uint8_t
#define BLL_set_NodeDataType fan::graphics::context_camera_t
#define BLL_set_Link 0
#define BLL_set_nrtra 1
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 0
#include <BLL/BLL.h>
  using camera_nr_t = camera_list_NodeReference_t;

#include <fan/fan_bll_preset.h>
#define BLL_set_prefix shader_list
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType fan::graphics::shader_data_t
#define BLL_set_Link 0
#define BLL_set_nrtra 1
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 0
#define bcontainer_set_StoreFormat 1
#include <BLL/BLL.h>
  using shader_nr_t = shader_list_NodeReference_t;

#include <fan/fan_bll_preset.h>
#define BLL_set_prefix image_list
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType fan::graphics::image_data_t
#define BLL_set_Link 0
#define BLL_set_nrtra 1
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 0
#define BLL_set_Usage 1
#include <BLL/BLL.h>
  using image_nr_t = image_list_NodeReference_t;

#include <fan/fan_bll_preset.h>
#define BLL_set_prefix viewport_list
#define BLL_set_type_node uint8_t
#define BLL_set_NodeDataType fan::graphics::context_viewport_t
#define BLL_set_Link 0
#define BLL_set_nrtra 1
#define BLL_set_AreWeInsideStruct 0
#include <BLL/BLL.h>
  using viewport_nr_t = viewport_list_NodeReference_t;
};

export namespace fan {
  namespace graphics {
    using camera_list_t = bll_builds::camera_list_t;
    using shader_list_t = bll_builds::shader_list_t;
    using image_list_t = bll_builds::image_list_t;
    using viewport_list_t = bll_builds::viewport_list_t;

    using camera_nr_t = bll_builds::camera_nr_t;
    using shader_nr_t = bll_builds::shader_nr_t;
    using image_nr_t = bll_builds::image_nr_t;
    using viewport_nr_t = bll_builds::viewport_nr_t;

    struct context_functions_t {
      context_build_shader_functions(context_typedef_func_ptr)
      context_build_image_functions(context_typedef_func_ptr)
      context_build_camera_functions(context_typedef_func_ptr)
      context_build_viewport_functions(context_typedef_func_ptr)
    };
    context_functions_t get_vk_context_functions();

    constexpr uint8_t get_channel_amount(uint8_t format) {
      switch (static_cast<image_format>(format)) {
      case image_format::undefined: return 0;

      case image_format::r8_unorm:
      case image_format::r8_uint: return 1;

      case image_format::rg8_unorm: return 2;

      case image_format::rgb_unorm: return 3;
      case image_format::bgr_unorm: return 3;

      case image_format::r8b8g8a8_unorm:
      case image_format::b8g8r8a8_unorm:
      case image_format::rgba_unorm:
      case image_format::r8g8b8a8_srgb: return 4;

      case image_format::r11f_g11f_b10f: return 3;

      case image_format::nv12: return 2;

      case image_format::yuv420p: return 3;

      default:
        fan::throw_error("Invalid format");
        return 0;
      }
    }


    constexpr std::array<fan::vec2ui, 4> get_image_sizes(uint8_t format, const fan::vec2ui& image_size) {
      using namespace fan::graphics;
      switch (format) {
      case  image_format::yuv420p: {
        return std::array<fan::vec2ui, 4>{image_size, image_size / 2, image_size / 2};
      }
      case  image_format::nv12: {
        return std::array<fan::vec2ui, 4>{image_size, fan::vec2ui{ image_size.x / 2, image_size.y / 2 }};
      }
      default: {
        fan::throw_error("invalid format");
        return std::array<fan::vec2ui, 4>{};
      }
      }
    }
    template <typename T>
    constexpr std::array<T, 4> get_image_properties(uint8_t format) {
      using namespace fan::graphics;
      std::array<T, 4> result{};

      switch (format) {
      case image_format::yuv420p:
        for (int i = 0; i < 3; ++i) {
          result[i].internal_format = fan::graphics::image_format::r8_unorm;
          result[i].format = fan::graphics::image_format::r8_unorm;
        }
        break;

      case image_format::nv12:
        result[0].internal_format = result[0].format = fan::graphics::image_format::r8_unorm;
        result[1].internal_format = result[1].format = fan::graphics::image_format::rg8_unorm;
        break;

      default:
        fan::throw_error("invalid format");
      }

      return result;
    }
  };
}


namespace bll_builds {
  #define BLL_set_SafeNext 1
  #define BLL_set_prefix update_callback
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType std::function<void(void*)>
  #define BLL_set_CPP_CopyAtPointerChange 1
  #include <BLL/BLL.h>

  
#if defined(fan_gui)
  #define BLL_set_SafeNext 1
  #define BLL_set_prefix gui_draw_cb
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType std::function<void()>
  #include <BLL/BLL.h>
  #endif
}

export namespace fan::graphics{
  using update_callback_t = bll_builds::update_callback_t;
  using update_callback_nr_t = bll_builds::update_callback_NodeReference_t;

  using gui_draw_cb_t = bll_builds::gui_draw_cb_t;
  using gui_draw_cb_nr_t = bll_builds::gui_draw_cb_NodeReference_t;
  
  bool gui_draw_cb_inric(gui_draw_cb_nr_t nr) {
    return bll_builds::gui_draw_cb_inric(nr);
  }
}

export namespace fan::graphics {
  using camera_t = fan::graphics::camera_nr_t;
  using shader_t = fan::graphics::shader_nr_t;
  using viewport_t = fan::graphics::viewport_nr_t;
  // image_t defined after render_context_handle_t

  struct render_view_t;

  
  struct lighting_t {
    static constexpr const char* ambient_name = "lighting_ambient";
    fan::vec3 ambient = fan::vec3(1, 1, 1);

    fan::vec3 start = ambient;
    fan::vec3 target = fan::vec3(1, 1, 1);
    f32_t duration = 0.5f; // seconds to reach target
    f32_t elapsed = 0.f;

    void set_target(const fan::vec3& t, f32_t d = 0.5f) {
      start = ambient;
      target = t;
      duration = d;
      elapsed = 0.0f;
    }

    void update(f32_t delta_time) {
      if (elapsed < duration) {
        elapsed += delta_time;
        f32_t t = std::min(elapsed / duration, 1.0f);
        ambient = fan::math::lerp(start, target, t);
      }
    }

    bool is_near(const fan::vec3& t, f32_t eps = 0.01f) const {
      return ambient.distance(t) < eps;
    }
    bool is_near_target(f32_t eps = 0.01f) const {
      return is_near(target, eps);
    }
  };

  struct render_context_handle_t {
    void set_context(context_functions_t& ctx, void* context) {
      render_functions = &ctx;
      render_context = context;
    }

    context_functions_t* operator->() { return render_functions; }
    operator void* () { return render_context; }

    uint8_t get_renderer() {
      return window->renderer;
    }

    context_functions_t* render_functions = nullptr;
    void* render_context = nullptr;

    // common data
    fan::graphics::image_nr_t default_texture;
    image_list_t* image_list = nullptr;
    shader_list_t* shader_list = nullptr;
    fan::window_t* window = nullptr;

    fan::graphics::render_view_t* orthographic_render_view = nullptr;
    fan::graphics::render_view_t* perspective_render_view = nullptr;

    update_callback_t* update_callback = nullptr;

    fan::window::input_action_t* input_action = nullptr;
    fan::console_t* console = nullptr;

    lighting_t* lighting = nullptr;

  #if defined(fan_gui)

    gui_draw_cb_t* gui_draw_cbs = nullptr;
    fan::graphics::gui::text_logger_t* text_logger = nullptr;
  #endif
  };

  thread_local render_context_handle_t g_render_context_handle;

  fan::window_t& get_window() {
    return *fan::graphics::g_render_context_handle.window;
  }

  render_context_handle_t& ctx() {
    return g_render_context_handle;
  }

  fan::graphics::render_view_t& get_orthographic_render_view() {
    return *ctx().orthographic_render_view;
  }
  fan::graphics::render_view_t& get_perspective_render_view() {
    return *ctx().perspective_render_view;
  }

  fan::graphics::image_data_t& image_get_data(fan::graphics::image_nr_t nr) {
		return (*ctx().image_list)[nr];
	}

  lighting_t& get_lighting() {
    return *ctx().lighting;
  }

  gui_draw_cb_t& get_gui_draw_cbs() {
    return *ctx().gui_draw_cbs;
  }
  
  struct image_t : fan::graphics::image_nr_t {
    using fan::graphics::image_nr_t::image_nr_t;
    // for no gloco access
    explicit image_t(bool) : fan::graphics::image_nr_t() {}
    image_t() : fan::graphics::image_nr_t(g_render_context_handle.default_texture) {}
    image_t(fan::graphics::image_nr_t image) : fan::graphics::image_nr_t(image) {

    }
    image_t(const fan::color& color) 
      : fan::graphics::image_nr_t(g_render_context_handle->image_create_color(g_render_context_handle, color)) {}
    image_t(const char* path, const std::source_location& callers_path = std::source_location::current())
      : image_t(std::string(path), callers_path) {
    }
    image_t(const std::string& path, const std::source_location& callers_path = std::source_location::current())
      : fan::graphics::image_nr_t(g_render_context_handle->image_load_path(g_render_context_handle, path, callers_path)) {}

    fan::vec2 get_size() const {
      return fan::graphics::image_get_data(*this).size;
    }
    operator fan::graphics::image_nr_t& () {
      return static_cast<fan::graphics::image_nr_t&>(*this);
    }
    operator const fan::graphics::image_nr_t& () const {
      return static_cast<const fan::graphics::image_nr_t&>(*this);
    }
  };

  fan::graphics::image_t get_default_texture() {
    return ctx().default_texture;
  }

  struct render_view_t {
    fan::graphics::camera_t camera;
    fan::graphics::viewport_t viewport;

    void create() {
      camera = g_render_context_handle->camera_create(g_render_context_handle);
      viewport = g_render_context_handle->viewport_create(g_render_context_handle);
    }

    void remove() {
      g_render_context_handle->camera_erase(g_render_context_handle, camera);
      g_render_context_handle->viewport_erase(g_render_context_handle, viewport);
    }

    void set(
      const fan::vec2& ortho_x, const fan::vec2& ortho_y,
      const fan::vec2& viewport_position, 
      const fan::vec2& viewport_size,
      const fan::vec2& window_size
    ) {
      g_render_context_handle->camera_set_ortho(g_render_context_handle, camera, ortho_x, ortho_y);
      g_render_context_handle->viewport_set(
        g_render_context_handle, viewport_position, viewport_size, window_size
      );
    }
  };

  fan::vec2 translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera) {
		auto v = g_render_context_handle->viewport_get(g_render_context_handle, viewport);
    auto c = g_render_context_handle->camera_get(g_render_context_handle, camera);

		fan::vec2 viewport_position = v.viewport_position;
		fan::vec2 viewport_size = v.viewport_size;

		f32_t l = c.coordinates.left;
		f32_t r = c.coordinates.right;
		f32_t t = c.coordinates.up;
		f32_t b = c.coordinates.down;

		fan::vec2 tp = p - viewport_position;
		fan::vec2 d = viewport_size;
		tp /= d;
		tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
		return tp;
	}

  fan::vec2 transform_position(const fan::vec2& p, fan::graphics::viewport_t viewport, fan::graphics::camera_t camera) {
    auto v = g_render_context_handle->viewport_get(g_render_context_handle, viewport);
    auto c = g_render_context_handle->camera_get(g_render_context_handle, camera);

    fan::vec2 viewport_position = v.viewport_position;
    fan::vec2 viewport_size = v.viewport_size;

    f32_t l = c.coordinates.left;
    f32_t r = c.coordinates.right;
    f32_t t = c.coordinates.up;
    f32_t b = c.coordinates.down;

    fan::vec2 tp = p - viewport_position;
    fan::vec2 d = viewport_size;
    tp /= d;
    tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
    tp += c.position;
    return tp;
  }
  fan::vec2 transform_position(const fan::vec2& p, const render_view_t& render_view) {
    return transform_position(p, render_view.viewport, render_view.camera);
  }
  fan::vec2 inverse_transform_position(
    const fan::vec2& p,
    fan::graphics::viewport_t viewport,
    fan::graphics::camera_t camera
  ) {
    auto v = g_render_context_handle->viewport_get(g_render_context_handle, viewport);
    auto c = g_render_context_handle->camera_get(g_render_context_handle, camera);

    fan::vec2 viewport_position = v.viewport_position;
    fan::vec2 viewport_size = v.viewport_size;

    f32_t l = c.coordinates.left;
    f32_t r = c.coordinates.right;
    f32_t t = c.coordinates.up;
    f32_t b = c.coordinates.down;

    fan::vec2 tp = p - c.position;

    f32_t u = (tp.x - l) / (r - l);
    f32_t vcoord = (tp.y - t) / (b - t);

    tp = fan::vec2(u, vcoord) * viewport_size;
    tp += viewport_position;

    return tp;
  }
  fan::vec2 inverse_transform_position(const fan::vec2& p, const render_view_t& render_view) {
    return inverse_transform_position(p, render_view.viewport, render_view.camera);
  }


  fan::vec2 get_mouse_position() {
    return fan::graphics::g_render_context_handle.window->get_mouse_position();
    //return get_mouse_position(default_camera->camera, default_camera->viewport); behaving oddly
  }
  fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport) {
    return fan::graphics::transform_position(get_mouse_position(), viewport, camera);
  }
  fan::vec2 get_mouse_position(const fan::graphics::render_view_t& render_view) {
    return get_mouse_position(render_view.camera, render_view.viewport);
  }
  struct icons_t {
    image_t play;
    image_t pause;
    image_t settings;
  }icons;
  struct tile_world_images {
    inline static fan::graphics::image_t dirt;
    inline static fan::graphics::image_t background;
  };
}

#if defined(fan_gui)
export namespace fan::graphics::gui {
  inline constexpr f32_t font_sizes[] = {
  4, 5, 6, 7, 8, 9, 10, 11, 12, 14,
  16, 18, 20, 22, 24, 28,
  32, 36, 48, 60, 72
  };
  ImFont* fonts[std::size(font_sizes)]{};
	ImFont* fonts_bold[std::size(font_sizes)]{};
}
#endif

export namespace fan {
  namespace window {

    fan::vec2 get_input_vector(
      const std::string& forward = "move_forward",
      const std::string& back = "move_back",
      const std::string& left = "move_left",
      const std::string& right = "move_right"
    ) {
      auto& ia = *fan::graphics::g_render_context_handle.input_action;
      fan::vec2 v(
        ia.is_action_down(right) - ia.is_action_down(left),
        ia.is_action_down(back) - ia.is_action_down(forward)
      );
      return v.length() > 0 ? v.normalized() : v;
    }
    fan::vec2 get_size() {
      return fan::graphics::g_render_context_handle.window->get_size();
    }
    void set_size(const fan::vec2& size) {
      fan::graphics::g_render_context_handle.window->set_size(size);
      fan::graphics::g_render_context_handle->viewport_set_nr(
        fan::graphics::g_render_context_handle,
        fan::graphics::g_render_context_handle.orthographic_render_view->viewport,
        fan::vec2(0, 0),
        size, 
        fan::window::get_size()
      );
      fan::graphics::g_render_context_handle->camera_set_ortho(
        fan::graphics::g_render_context_handle,
        fan::graphics::g_render_context_handle.orthographic_render_view->camera,
        fan::vec2(0, size.x),
        fan::vec2(0, size.y)
      );

      fan::graphics::g_render_context_handle->viewport_set_nr(
        fan::graphics::g_render_context_handle,
        fan::graphics::g_render_context_handle.perspective_render_view->viewport, 
        fan::vec2(0, 0), 
        size,
        fan::window::get_size()
      );
      fan::graphics::g_render_context_handle->camera_set_ortho(
        fan::graphics::g_render_context_handle,
        fan::graphics::g_render_context_handle.perspective_render_view->camera,
        fan::vec2(0, size.x),
        fan::vec2(0, size.y)
      );
    }

    fan::vec2 get_mouse_position() {
      return fan::graphics::get_mouse_position();
    }
    bool is_mouse_clicked(int button = fan::mouse_left) {
      return fan::graphics::g_render_context_handle.window->key_state(button) == (int)fan::mouse_state::press;
    }
    bool is_mouse_down(int button = fan::mouse_left) {
      int state = fan::graphics::g_render_context_handle.window->key_state(button);
      return
        state == (int)fan::mouse_state::press ||
        state == (int)fan::mouse_state::repeat;
    }
    bool is_mouse_released(int button = fan::mouse_left) {
      return fan::graphics::g_render_context_handle.window->key_state(button) == (int)fan::mouse_state::release;
    }
    fan::vec2 get_mouse_drag(int button = fan::mouse_left) {
      auto* win = fan::graphics::g_render_context_handle.window;
      if (is_mouse_down(button)) {
        if (win->drag_delta_start != fan::vec2(-1)) {
          return win->get_mouse_position() - win->drag_delta_start;
        }
      }
      return fan::vec2();
    }

    bool is_key_pressed(int key) {
      return fan::graphics::g_render_context_handle.window->key_state(key) == (int)fan::mouse_state::press;
    }
    bool is_key_down(int key) {
      int state = fan::graphics::g_render_context_handle.window->key_state(key);
      return
        state == (int)fan::mouse_state::press ||
        state == (int)fan::mouse_state::repeat;
    }
    bool is_key_released(int key) {
      return fan::graphics::g_render_context_handle.window->key_state(key) == (int)fan::mouse_state::release;
    }
  }
}


#undef context_typedef_func_ptr
#undef context_typedef_func_ptr2
#undef context_declare_func
#undef context_declare_func2
#undef context_build_shader_functions
#undef context_build_image_functions
#undef context_build_camera_functions
#undef context_build_viewport_functions