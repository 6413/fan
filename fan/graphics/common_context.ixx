module;

#include <fan/utility.h>

#include <fan/graphics/common_context_functions_declare.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <source_location>
#include <functional>
#include <cstdlib>
#include <chrono>
#include <type_traits>
#include <coroutine>

#include <sstream>
#include <fstream>

export module fan.graphics.common_context;

export import fan.types.color;
export import fan.graphics.image_load;
export import fan.camera;

import fan.io.file;
import fan.window;
import fan.window.input_action;
import fan.print;
import fan.utility;
import fan.types.compile_time_string;


#if defined(FAN_GUI)
  import fan.graphics.gui.types;
  import fan.console;
#endif

#define __dme_extend \
  static auto get_names_impl() { \
    std::array<fan::ct_string<256>, size()> a{}; \
    for (size_t i = 0; i < size(); i++) { \
      a[i] = fan::snake_to_title(items()[i]); \
    } \
    return a; \
  } \
  static auto get_names() { \
    static auto names = get_names_impl(); \
    std::array<const char*, size()> p{}; \
    for (size_t i = 0; i < size(); i++) \
      p[i] = names[i].c_str(); \
    return p; \
  }

namespace fan {
  #include <fan/types/dme.h>
}

export namespace fan {
  namespace graphics {
    struct image_format_e : __dme_inherit(image_format_e) {
      __dme(r8b8g8a8_unorm);
      __dme(b8g8r8a8_unorm);
      __dme(r8_unorm);
      __dme(rg8_unorm);
      __dme(rgb_unorm);
      __dme(rgba_unorm);
      __dme(bgr_unorm);
      __dme(r8_uint);
      __dme(r8g8b8a8_srgb);
      __dme(r11f_g11f_b10f);
      __dme(yuv420p);
      __dme(nv12);

      static constexpr uint8_t undefined = 255;
    }image_format;

    constexpr uint8_t get_texture_amount(uint8_t format) {
      switch (format) {
      case image_format_e::undefined: {
        return 0;
      }
      case image_format_e::yuv420p: {
        return 3;
      }
      case image_format_e::nv12: {
        return 2;
      }
      default: {
        fan::throw_error("invalid format");
        return image_format_e::undefined;
      }
      }
    }
    struct image_sampler_address_mode_e : __dme_inherit(image_sampler_address_mode_e) {
      __dme(repeat);
      __dme(mirrored_repeat);
      __dme(clamp_to_edge);
      __dme(clamp_to_border);
      __dme(mirrored_clamp_to_edge);
    }image_sampler_address_mode;
    struct image_filter_e : __dme_inherit(image_filter_e) {
      __dme(nearest);
      __dme(linear);
      __dme(nearest_mipmap_nearest);
      __dme(linear_mipmap_nearest);
      __dme(nearest_mipmap_linear);
      __dme(linear_mipmap_linear);
    } image_filter;
    enum data_types {
      fan_unsigned_byte,
      fan_byte,
      fan_unsigned_int,
      fan_float,
    };
    struct image_load_properties_defaults {
      static constexpr uint32_t visual_output = image_sampler_address_mode_e::repeat;
      static constexpr uint32_t internal_format = image_format_e::r8b8g8a8_unorm;
      static constexpr uint32_t format = image_format_e::r8b8g8a8_unorm;
      static constexpr uint32_t type = fan_unsigned_byte; // internal
      static constexpr uint32_t min_filter = image_filter_e::linear;
      static constexpr uint32_t mag_filter = image_filter_e::linear;
    };

    struct context_camera_t : fan::camera {
      fan::mat4 m_projection = fan::mat4(1);
      fan::mat4 m_view = fan::mat4(1);
      f32_t zfar = 1000.f;
      f32_t znear = 0.1f;
      f32_t zoom = 1.0f;

      union {
        struct {
          f32_t left;
          f32_t right;
          f32_t top;
          f32_t bottom;
        };
        fan::vec4 v;
      }coordinates;
      fan::vec4 original_coordinates = fan::vec4(0, 0, 0, 0);
    };

    struct context_viewport_t {
      fan::vec2 position;
      fan::vec2 size;
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

    struct shader_data_t {
      fan::ct_string<256> path_vertex, path_fragment;
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

    constexpr uint8_t get_channel_amount(uint32_t format) {
      switch (format) {
      case image_format_e::undefined: return 0;

      case image_format_e::r8_unorm:
      case image_format_e::r8_uint: return 1;

      case image_format_e::rg8_unorm: return 2;

      case image_format_e::rgb_unorm:
      case image_format_e::bgr_unorm: return 3;

      case image_format_e::r8b8g8a8_unorm:
      case image_format_e::b8g8r8a8_unorm:
      case image_format_e::rgba_unorm:
      case image_format_e::r8g8b8a8_srgb: return 4;

      case image_format_e::r11f_g11f_b10f: return 3;

      case image_format_e::nv12: return 2;

      case image_format_e::yuv420p: return 3;

      default:
        fan::throw_error("Invalid format");
        return 0;
      }
    }

    constexpr std::array<fan::vec2ui, 4> get_image_sizes(uint8_t format, const fan::vec2ui& image_size) {
      using namespace fan::graphics;
      switch (format) {
      case image_format_e::yuv420p:
      {
        return std::array<fan::vec2ui, 4>{image_size, image_size / 2, image_size / 2};
      }
      case image_format_e::nv12:
      {
        return std::array<fan::vec2ui, 4>{image_size, fan::vec2ui {image_size.x / 2, image_size.y / 2}};
      }
      default:
      {
        fan::throw_error("invalid format");
        return std::array<fan::vec2ui, 4>{};
      }
      }
    }

    template <typename T>
    constexpr std::array<T, 4> get_image_properties(uint8_t format) {
      using namespace fan::graphics;
      std::array<T, 4> result {};

      switch (format) {
      case image_format_e::yuv420p:
        for (int i = 0; i < 3; ++i) {
          result[i].internal_format = fan::graphics::image_format_e::r8_unorm;
          result[i].format = fan::graphics::image_format_e::r8_unorm;
        }
        break;

      case image_format_e::nv12:
        result[0].internal_format = result[0].format = fan::graphics::image_format_e::r8_unorm;
        result[1].internal_format = result[1].format = fan::graphics::image_format_e::rg8_unorm;
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

  
#if defined(FAN_GUI)
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

#if defined(FAN_GUI)
  using gui_draw_cb_t = bll_builds::gui_draw_cb_t;
  using gui_draw_cb_nr_t = bll_builds::gui_draw_cb_NodeReference_t;
  
  bool gui_draw_cb_inric(gui_draw_cb_nr_t nr) {
    return bll_builds::gui_draw_cb_inric(nr);
  }
#endif
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

    void set_target(const fan::vec3& t, f32_t d = 0.5f);
    void update(f32_t delta_time);
    bool is_near(const fan::vec3& t, f32_t eps = 0.01f) const;
    bool is_near_target(f32_t eps = 0.01f) const;
  };

  struct render_context_handle_t {
    void set_context(context_functions_t& ctx, void* context);
    context_functions_t* operator->();
    operator void* ();
    uint8_t get_renderer();

    context_functions_t* context_functions = nullptr;
    void* render_context = nullptr;

    // common data
    fan::graphics::image_nr_t default_texture;
    camera_list_t* camera_list = nullptr;
    shader_list_t* shader_list = nullptr;
    image_list_t* image_list = nullptr;
    viewport_list_t* viewport_list = nullptr;
    fan::window_t* window = nullptr;

    fan::graphics::render_view_t* orthographic_render_view = nullptr;
    fan::graphics::render_view_t* perspective_render_view = nullptr;

    update_callback_t* update_callback = nullptr;

    fan::window::input_action_t* input_action = nullptr;
  #if defined(FAN_GUI)
    fan::console_t* console = nullptr;
  #endif

    lighting_t* lighting = nullptr;

  #if defined(FAN_GUI)

    gui_draw_cb_t* gui_draw_cbs = nullptr;
    void* text_logger = nullptr;
  #endif
  };
  fan::window_t& get_window();
  render_context_handle_t& ctx();
  fan::graphics::render_view_t& get_orthographic_render_view();
  fan::graphics::render_view_t& get_perspective_render_view();
  fan::graphics::image_data_t& image_get_data(fan::graphics::image_nr_t nr);
  lighting_t& get_lighting();

#if defined(FAN_GUI)
  gui_draw_cb_t& get_gui_draw_cbs();
#endif

  struct image_t : fan::graphics::image_nr_t {
    using fan::graphics::image_nr_t::image_nr_t;
    // for no gloco access
    explicit image_t(bool);
    image_t();
    image_t(fan::graphics::image_nr_t image);
    image_t(const fan::color& color);
    image_t(const char* path, const std::source_location& callers_path = std::source_location::current());
    image_t(const std::string& path, const std::source_location& callers_path = std::source_location::current());

    image_t(const char* path, const fan::graphics::image_load_properties_t lp, const std::source_location& callers_path = std::source_location::current());
    image_t(const std::string& path, const fan::graphics::image_load_properties_t lp, const std::source_location& callers_path = std::source_location::current());


    fan::vec2 get_size() const;
    image_load_properties_t get_load_properties() const;
    std::string get_path() const;

    operator fan::graphics::image_nr_t& ();
    operator const fan::graphics::image_nr_t& () const;
    bool valid() const;
  };

  fan::graphics::image_t get_default_texture();

  struct render_view_t {
    fan::graphics::camera_t camera;
    fan::graphics::viewport_t viewport;
    render_view_t() = default;
    explicit render_view_t(bool);
    void create();
    void create_default(const fan::vec2& window_size, f32_t zoom = 1.f);
    void remove();
    void set(
      const fan::vec2& ortho_x, const fan::vec2& ortho_y,
      const fan::vec2& viewport_position,
      const fan::vec2& viewport_size,
      const fan::vec2& window_size
    );
    std::string debug_string();
  };

  fan::vec2 translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera);
  fan::vec2 screen_to_world(const fan::vec2& p, fan::graphics::viewport_t viewport, fan::graphics::camera_t camera);
  fan::vec2 screen_to_world(const fan::vec2& p, const render_view_t& render_view = *fan::graphics::ctx().orthographic_render_view);
  fan::vec2 world_to_screen(const fan::vec2& p, fan::graphics::viewport_t viewport, fan::graphics::camera_t camera);
  fan::vec2 world_to_screen(const fan::vec2& p, const render_view_t& render_view = *fan::graphics::ctx().orthographic_render_view);
  fan::vec2 get_mouse_position();
  fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport);
  fan::vec2 get_mouse_position(const fan::graphics::render_view_t& render_view);

  struct icons_t {
    image_t play;
    image_t pause;
    image_t settings;
  }icons;
  struct tile_world_images {
    inline static fan::graphics::image_t dirt;
    inline static fan::graphics::image_t background;
  };

  std::string read_shader(
    std::string_view path,
    const std::source_location& callers_path = std::source_location::current()
  );
}

export namespace fan {
  namespace actions {
    namespace groups {
      inline constexpr const char* system = "System";
      inline constexpr const char* movement = "Movement";
      inline constexpr const char* combat = "Combat";
      inline constexpr const char* debug = "Debug";
    }

    // System
    inline constexpr const char* toggle_settings = "Toggle Settings";
    inline constexpr const char* toggle_console = "Toggle Console";

    // Movement
    inline constexpr const char* move_forward = "Move Forward";
    inline constexpr const char* move_back = "Move Back";
    inline constexpr const char* move_left = "Move Left";
    inline constexpr const char* move_right = "Move Right";
    inline constexpr const char* move_up = "Move Up";

    // Combat
    inline constexpr const char* light_attack = "Light Attack";
    inline constexpr const char* block_attack = "Block Attack";

    // Debug
    constexpr const char* toggle_debug_physics = "Toggle Debug Physics";

    constexpr const char* toggle_debug_light_buffer = "Toggle Light Buffer";
    constexpr const char* recompile_shaders = "Recompile Shaders";
  }

  namespace window {
    fan::vec2 get_input_vector(
      const std::string& forward = fan::actions::move_forward,
      const std::string& back = fan::actions::move_back,
      const std::string& left = fan::actions::move_left,
      const std::string& right = fan::actions::move_right
    );
    fan::vec2 get_size();
    void set_size(const fan::vec2& size);
    fan::vec2 get_mouse_position();
    bool is_mouse_clicked(int button = fan::mouse_left);
    bool is_mouse_down(int button = fan::mouse_left);
    bool is_mouse_released(int button = fan::mouse_left);
    fan::vec2 get_mouse_drag(int button = fan::mouse_left);
    bool is_key_clicked(int key);
    bool is_key_down(int key);
    bool is_key_released(int key);
    bool is_gamepad_button_down(int key);
    bool is_gamepad_axis_active(int key);
    fan::vec2 get_current_gamepad_axis(int key);

    bool is_input_clicked(const std::string& name);
    bool is_input_down(const std::string& name);
    bool is_input_released(const std::string& name);
  }
}

export namespace fan::graphics {
  struct next_frame_awaiter {
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> handle) {
      get_pending().push_back(handle);
    }
    void await_resume() const noexcept {}
    static std::vector<std::coroutine_handle<>>& get_pending() {
      static std::vector<std::coroutine_handle<>> pending;
      return pending;
    }
    
  };
  next_frame_awaiter co_next_frame() {
    return {};
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