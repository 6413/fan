module;

#if defined (FAN_WINDOW)

#include <fan/utility.h>

#include <cstdint>
#include <coroutine>
#include <fan/graphics/common_context_functions_declare.h>

#endif

export module fan.graphics.common_context;

#if defined (FAN_WINDOW)

import std;

import fan.types;
import fan.types.color;
import fan.types.matrix;
import fan.types.compile_time_string;
import fan.camera;

import fan.io.file;

import fan.print.error;
import fan.memory;

import fan.window;
import fan.window.input;
import fan.window.input_action;
import fan.graphics.image_load;


#if defined(FAN_GUI)
  import fan.graphics.gui.types;
#endif

#define __dme_extend \
  static auto get_names_impl() { \
    std::array<fan::ct_string<256>, size()> a{}; \
    for (std::size_t i = 0; i < size(); i++) { \
      a[i] = fan::snake_to_title(items()[i]); \
    } \
    return a; \
  } \
  static auto get_names() { \
    static auto names = get_names_impl(); \
    std::array<const char*, size()> p{}; \
    for (std::size_t i = 0; i < size(); i++) \
      p[i] = names[i].c_str(); \
    return p; \
  }

namespace fan {
  #include <fan/types/dme.h>
}

export namespace fan {
  namespace graphics {
    struct image_format_e : __dme_inherit(image_format_e) {
      __dme(bgra);
      __dme(r8_unorm);
      __dme(r32_float);
      __dme(rg8_unorm);
      __dme(rgb_unorm);
      __dme(rgba_unorm);
      __dme(rgba);
      __dme(rgba8);
      __dme(bgr_unorm);
      __dme(r8_uint);
      __dme(r8g8b8a8_srgb);
      __dme(r11f_g11f_b10f);
      __dme(yuv420p);
      __dme(nv12);
      __dme(rgba32f);

      static constexpr std::uint8_t undefined = 255;
    }image_format;

    constexpr std::uint8_t get_texture_amount(std::uint8_t format) {
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
        fan::throw_error_impl("invalid format");
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
    enum data_type_e {
      fan_unsigned_byte,
      fan_byte,
      fan_unsigned_int,
      fan_float,
    };
    struct image_load_properties_defaults {
      static constexpr std::uint32_t visual_output = image_sampler_address_mode_e::repeat;
      static constexpr std::uint32_t internal_format = image_format_e::rgba;
      static constexpr std::uint32_t format = image_format_e::rgba;
      static constexpr std::uint32_t type = fan_unsigned_byte; // internal
      static constexpr std::uint32_t min_filter = image_filter_e::linear;
      static constexpr std::uint32_t mag_filter = image_filter_e::linear;
    };

    struct context_camera_t : fan::camera {
      fan::mat4 projection = fan::mat4(1);
      fan::mat4 view = fan::mat4(1);
      f32_t zfar = 1000.f;
      f32_t znear = 0.1f;
      f32_t zoom = 1.0f;
      f32_t fov = 90.f;
      f32_t speed = 1000.f;
      f32_t friction = 12.f;

      union {
        struct {
          f32_t left;
          f32_t right;
          f32_t top;
          f32_t bottom;
        };
        fan::vec4 v;
      }coordinates{};

      friend std::ostream& operator<<(std::ostream& os, const context_camera_t& c) {
        return os << std::format(
          "pos: {:.1f} {:.1f} {:.1f}\nyaw: {:.2f}  pitch: {:.2f}\nfov: {:.1f}  near: {:.2f}  far: {:.1f}  zoom: {:.2f}",
          c.position.x, c.position.y, c.position.z, c.yaw, c.pitch, c.fov, c.znear, c.zfar, c.zoom
        );
      }

      fan::vec4 original_coordinates = fan::vec4(0, 0, 0, 0);
    };

    struct context_viewport_t {
      fan::vec2 position;
      fan::vec2 size;
    };

    struct image_load_properties_t {
      std::uint32_t visual_output = image_load_properties_defaults::visual_output;
      std::uintptr_t internal_format = image_load_properties_defaults::internal_format;
      std::uintptr_t format = image_load_properties_defaults::format;
      std::uintptr_t type = image_load_properties_defaults::type;
      std::uintptr_t min_filter = image_load_properties_defaults::min_filter;
      std::uintptr_t mag_filter = image_load_properties_defaults::mag_filter;
    };

    struct image_data_t {
      fan::vec2 size;
      std::string image_path;
      image_load_properties_t image_settings;
      void* internal;
    };

    struct shader_data_t {
      fan::ct_string<256> path_vertex, path_fragment, path_compute;
      std::string svertex, sfragment, scompute;
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
#define BLL_set_type_node std::uint8_t
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
#define BLL_set_type_node std::uint16_t
#define BLL_set_NodeDataType fan::graphics::image_data_t
#define BLL_set_Link 0
#define BLL_set_nrtra 1
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 0
#define BLL_set_Usage 1
#define bcontainer_set_StoreFormat 1
#include <BLL/BLL.h>
  using image_nr_t = image_list_NodeReference_t;

#include <fan/fan_bll_preset.h>
#define BLL_set_prefix viewport_list
#define BLL_set_type_node std::uint8_t
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

    constexpr std::uint8_t get_channel_amount(std::uint32_t format) {
      switch (format) {
      case image_format_e::undefined: return 0;
      case image_format_e::r32_float:
      case image_format_e::r8_unorm:
      case image_format_e::r8_uint: return 1;

      case image_format_e::rg8_unorm: return 2;

      case image_format_e::rgb_unorm:
      case image_format_e::bgr_unorm: return 3;

      case image_format_e::rgba:
      case image_format_e::bgra:
      case image_format_e::rgba_unorm:
      case image_format_e::r8g8b8a8_srgb: 
      case image_format_e::rgba32f:return 4;

      case image_format_e::r11f_g11f_b10f: return 3;

      case image_format_e::nv12: return 2;

      case image_format_e::yuv420p: return 3;

      default:
        fan::throw_error_impl("Invalid format");
        return 0;
      }
    }

    constexpr std::array<fan::vec2ui, 4> get_image_sizes(std::uint8_t format, const fan::vec2ui& image_size) {
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
        fan::throw_error_impl("invalid format");
        return std::array<fan::vec2ui, 4>{};
      }
      }
    }

    template <typename T>
    constexpr std::array<T, 4> get_image_properties(std::uint8_t format) {
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
        fan::throw_error_impl("invalid format");
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
  struct shader_t : fan::graphics::shader_nr_t {
    using shader_nr_t::shader_nr_t;
    shader_t(shader_nr_t nr) : shader_nr_t(nr) {}
    template <typename T>
    void set_value(auto& engine, const std::string_view name, const T& val) {
      engine.shader_set_value(*this, name, val);
    }
    void use(auto& engine) const {
      engine.shader_use(*this);
    }
  };
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
    std::uint8_t get_renderer();

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
    void* console = nullptr;
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

  namespace image_presets {
    image_load_properties_t pixel_art() {
      image_load_properties_t props;
      props.visual_output = fan::graphics::image_sampler_address_mode_e::clamp_to_border;
      props.min_filter = image_filter_e::nearest;
      props.mag_filter = image_filter_e::nearest;
      return props;
    }

    image_load_properties_t pixel_art_repeat() {
      auto props = pixel_art();
      props.visual_output = fan::graphics::image_sampler_address_mode_e::repeat;
      return props;
    }

    image_load_properties_t smooth() {
      image_load_properties_t props;
      props.visual_output = fan::graphics::image_sampler_address_mode_e::clamp_to_border;
      props.min_filter = image_filter_e::linear;
      props.mag_filter = image_filter_e::linear;
      return props;
    }

    image_load_properties_t mipmapped() {
      image_load_properties_t props;
      props.min_filter = image_filter_e::linear_mipmap_linear;
      props.mag_filter = image_filter_e::linear;
      return props;
    }
  }


  struct image_t : fan::graphics::image_nr_t {
    explicit image_t(__empty_struct st);
    image_t();
    image_t(fan::graphics::image_nr_t image);
    image_t(const fan::color& color);
    image_t(fan::str_view_t path, const std::source_location& callers_path = std::source_location::current());
    image_t(fan::str_view_t path, const fan::graphics::image_load_properties_t lp, const std::source_location& callers_path = std::source_location::current());
    image_t(const char* path, const std::source_location& callers_path = std::source_location::current());
    image_t(const fan::image::info_t& info);
    image_t(const fan::image::info_t& info, const fan::graphics::image_load_properties_t& lp);
    image_t(fan::color* colors, const fan::vec2ui& size);
    image_t(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& lp);
    image_t(std::span<const fan::color> colors, const fan::vec2ui& size);
    image_t(const fan::vec2& size, std::uint32_t channels = 4, const image_load_properties_t& lp = image_presets::pixel_art());

    // for no gloco access
    static image_t invalid();

    fan::vec2 get_size() const;
    image_load_properties_t get_load_properties() const;
    std::string get_path() const;

    bool valid() const;

    void reload(const fan::image::info_t& info);
    void reload(const fan::image::info_t& info, const fan::graphics::image_load_properties_t& lp);
    void reload(const std::string& path, const std::source_location& callers_path = std::source_location::current());
    void reload(const std::string& path, const fan::graphics::image_load_properties_t& lp, const std::source_location& callers_path = std::source_location::current());
    void unload();
    void remove() {
      unload();
    }
    void update(const void* data, std::uint32_t channels = 4);
    void update(const std::vector<std::uint8_t>& data, std::uint32_t channels = 4);
    std::vector<std::uint8_t> get_pixel_data(int image_format, fan::vec2 uvp = 0, fan::vec2 uvs = 1) const;
    std::vector<std::uint8_t> read_pixels(const fan::vec2& uv_position = 0, const fan::vec2& uv_size = 1) const;
    void bind() const;
    void bind(std::uint32_t unit) const;
    void bind(std::uint32_t unit, std::uint32_t access, std::uint32_t format) const;
    void unbind() const;
    std::uint64_t get_handle() const;
    image_load_properties_t& get_settings();
    void set_settings(const fan::graphics::image_load_properties_t& settings);
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
    operator fan::graphics::camera_t&();
    operator fan::graphics::viewport_t&();
    fan::vec3 get_camera_position() const;
    void set_camera_position(fan::vec3 pos);
    context_camera_t& get_camera();
    context_viewport_t& get_viewport();
  };

  fan::vec2 translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera);
  fan::vec2 screen_to_world(const fan::vec2& p, fan::graphics::viewport_t viewport, fan::graphics::camera_t camera);
  fan::vec2 screen_to_world(const fan::vec2& p, const render_view_t& render_view = *fan::graphics::ctx().orthographic_render_view);
  fan::vec2 world_to_screen(const fan::vec2& p, fan::graphics::viewport_t viewport, fan::graphics::camera_t camera);
  fan::vec2 world_to_screen(const fan::vec2& p, const render_view_t& render_view = *fan::graphics::ctx().orthographic_render_view);
  fan::vec2 get_mouse_position();
  fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport);
  fan::vec2 get_mouse_position(const fan::graphics::render_view_t& render_view);
  fan::vec2 get_mouse_world_pos();

  struct icons_t {
    image_t play;
    image_t pause;
    image_t settings;
  }icons;
  struct tile_world_images_t {
    fan::graphics::image_t dirt;
    fan::graphics::image_t background;
  }tile_world_images;
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
    void add_input_action(const int* keys, std::size_t count, const std::string_view action_name);
    void add_input_action(std::initializer_list<int> keys, const std::string_view action_name);
    void add_input_action(int key, const std::string_view action_name);
    bool is_input_action_active(const std::string_view action_name, int pstate = fan::window::input_action_t::press);
    bool is_action_clicked(const std::string_view action_name);
    bool is_action_down(const std::string_view action_name);
    bool exists(const std::string_view action_name);

    fan::vec2 get_input_vector(
      const std::string& forward = fan::actions::move_forward,
      const std::string& back = fan::actions::move_back,
      const std::string& left = fan::actions::move_left,
      const std::string& right = fan::actions::move_right
    );
    fan::vec2 get_input_vector(fan::vec2 scalar);
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
    char get_char_pressed();
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

export namespace fan::graphics {
  fan::graphics::image_t image_create();
  std::uint64_t image_get_handle(fan::graphics::image_t nr);
  void image_erase(fan::graphics::image_t nr);
  void image_bind(fan::graphics::image_t nr);
  void image_unbind(fan::graphics::image_t nr);
  fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_t nr);
  void image_set_settings(fan::graphics::image_t nr, const fan::graphics::image_load_properties_t& settings);
  fan::graphics::image_t image_load(const fan::image::info_t& image_info);
  fan::graphics::image_t image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_t image_load(const std::string& path, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_t image_load(fan::color* colors, const fan::vec2ui& size);
  fan::graphics::image_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_t image_load(std::span<const fan::color> colors, const fan::vec2ui& size);
  void image_unload(fan::graphics::image_t nr);
  bool is_image_valid(fan::graphics::image_t nr);
  fan::graphics::image_t image_load_pixel_art(const std::string& path);
  fan::graphics::image_t image_load_smooth(const std::string& path);
  fan::graphics::image_t create_missing_texture();
  fan::graphics::image_t create_transparent_texture();
  void image_reload(fan::graphics::image_t nr, const fan::image::info_t& image_info);
  void image_reload(fan::graphics::image_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p);
  void image_reload(fan::graphics::image_t nr, const std::string& path, const std::source_location& callers_path = std::source_location::current());
  void image_reload(fan::graphics::image_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_t image_create(const fan::color& color);
  fan::graphics::image_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_t image_create(void* data, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p);
  std::vector<std::uint8_t> read_pixels(const fan::vec2& position, const fan::vec2& size);
  std::vector<std::uint8_t> read_pixels_from_image(fan::graphics::image_t nr, const fan::vec2& uv_position = 0, const fan::vec2& uv_size = 1);

  fan::graphics::shader_t shader_create();
  fan::graphics::shader_t shader_create(
    const std::string_view vertex_file_path,
    const fan::str_view_t vertex, 
    const std::string_view fragment_file_path,
    const fan::str_view_t fragment
  );
  void shader_erase(fan::graphics::shader_nr_t nr);
  void shader_use(fan::graphics::shader_nr_t nr);
  void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code);
  void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code);
  fan::graphics::shader_t compute_shader_create();

  fan::graphics::shader_t compute_shader_create(
    const std::string_view compute_file_path,
    const fan::str_view_t compute
  );

  void shader_set_compute(
    fan::graphics::shader_nr_t nr,
    const std::string_view file_path,
    const std::string& compute_code
  );

  void shader_dispatch_compute(
    fan::graphics::shader_nr_t nr,
    uint32_t x,
    uint32_t y,
    uint32_t z
  );
  bool shader_compile(fan::graphics::shader_nr_t nr);
  std::string read_shader(
    std::string_view path,
    const std::source_location& callers_path = std::source_location::current()
  );
  fan::graphics::shader_t get_sprite_shader(const std::string_view fragment_file_path, const fan::str_view_t fragment);

  fan::graphics::camera_nr_t camera_create();
  fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr = fan::graphics::get_orthographic_render_view());
  void camera_erase(fan::graphics::camera_nr_t nr);
  fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y);
  // Returns the raw translation offset of the camera matrix.
  // For an orthographic projection starting at (0,0), this represents the top-left corner.
  // For a symmetric projection (e.g., -width/2 to width/2), this represents the center.
  fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr);
  void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
  void camera_set_position(const fan::vec3& cp);
  // Returns the true world-space center of the camera's view,
  // regardless of how the projection matrix was initialized.
  fan::vec3 camera_get_center(fan::graphics::camera_nr_t nr = fan::graphics::get_orthographic_render_view());
  void camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
  void camera_set_center(const fan::vec3& cp);
  fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr);
  fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr = fan::graphics::get_orthographic_render_view());
  f32_t camera_get_zoom(fan::graphics::camera_nr_t nr);
  void camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom);
  void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y);
  void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size);
  void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset);
  void camera_follow(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed = 10);
  void camera_follow(const fan::vec2& target, f32_t move_speed = 10);
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

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////


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

export namespace fan::shader_paths {
  namespace gl {

    constexpr const char* blit_fs =               "shaders/opengl/2D/blit.fs";
    constexpr const char* blit_vs =               "shaders/opengl/2D/blit.vs";

    // 2D shapes
    constexpr const char* capsule_fs =               "shaders/opengl/2D/objects/capsule.fs";
    constexpr const char* capsule_vs =               "shaders/opengl/2D/objects/capsule.vs";
    constexpr const char* circle_fs =                "shaders/opengl/2D/objects/circle.fs";
    constexpr const char* circle_vs =                "shaders/opengl/2D/objects/circle.vs";
    constexpr const char* grid_fs =                  "shaders/opengl/2D/objects/grid.fs";
    constexpr const char* grid_vs =                  "shaders/opengl/2D/objects/grid.vs";
    constexpr const char* light_fs =                 "shaders/opengl/2D/objects/light.fs";
    constexpr const char* light_vs =                 "shaders/opengl/2D/objects/light.vs";
    constexpr const char* line_fs =                  "shaders/opengl/2D/objects/line.fs";
    constexpr const char* line_vs =                  "shaders/opengl/2D/objects/line.vs";
    constexpr const char* polygon_fs =               "shaders/opengl/2D/objects/polygon.fs";
    constexpr const char* polygon_vs =               "shaders/opengl/2D/objects/polygon.vs";
    constexpr const char* rectangle_fs =             "shaders/opengl/2D/objects/rectangle.fs";
    constexpr const char* rectangle_vs =             "shaders/opengl/2D/objects/rectangle.vs";
    constexpr const char* shadow_fs =                "shaders/opengl/2D/objects/shadow.fs";
    constexpr const char* shadow_vs =                "shaders/opengl/2D/objects/shadow.vs";
    constexpr const char* sprite_fs =                "shaders/opengl/2D/objects/sprite.fs";
    constexpr const char* sprite_vs =                "shaders/opengl/2D/objects/sprite.vs";
    constexpr const char* sprite_2_1_fs =            "shaders/opengl/2D/objects/sprite_2_1.fs";
    constexpr const char* sprite_2_1_vs =            "shaders/opengl/2D/objects/sprite_2_1.vs";
    constexpr const char* unlit_sprite_fs =          "shaders/opengl/2D/objects/unlit_sprite.fs";


    // 2D effects
    
    // 2D alpha shadow
    constexpr const char* alpha_shadow_quad_vs =     "shaders/opengl/2D/effects/alpha_shadow/quad.vs";
    constexpr const char* alpha_shadow_solid_fs =    "shaders/opengl/2D/effects/alpha_shadow/solid.fs";
    constexpr const char* alpha_shadow_occluder_fs = "shaders/opengl/2D/effects/alpha_shadow/occluder.fs";
    constexpr const char* alpha_shadow_radial_fs =   "shaders/opengl/2D/effects/alpha_shadow/radial.fs";
    constexpr const char* alpha_shadow_light_fs =    "shaders/opengl/2D/effects/alpha_shadow/light.fs";

    constexpr const char* clouds_fs =                "shaders/opengl/2D/effects/clouds.fs";
    constexpr const char* downsample_fs =            "shaders/opengl/2D/effects/downsample.fs";
    constexpr const char* downsample_vs =            "shaders/opengl/2D/effects/downsample.vs";
    constexpr const char* gradient_fs =              "shaders/opengl/2D/effects/gradient.fs";
    constexpr const char* gradient_vs =              "shaders/opengl/2D/effects/gradient.vs";
    constexpr const char* particles_fs =             "shaders/opengl/2D/effects/particles.fs";
    constexpr const char* particles_vs =             "shaders/opengl/2D/effects/particles.vs";
    constexpr const char* reflection_fs =            "shaders/opengl/2D/effects/reflection.fs";
    constexpr const char* reflection_vs =            "shaders/opengl/2D/effects/reflection.vs";
    constexpr const char* upsample_fs =              "shaders/opengl/2D/effects/upsample.fs";

    // Framebuffer / post-process
    constexpr const char* final_fs =                 "shaders/opengl/2D/effects/loco_fbo.fs";
    constexpr const char* final_vs =                 "shaders/opengl/2D/effects/loco_fbo.vs";

    // Unique
    constexpr const char* empty_fs =                 "shaders/empty.fs";
    constexpr const char* empty_vs =                 "shaders/empty.vs";
    constexpr const char* pixel_format_renderer_vs = "shaders/opengl/2D/objects/pixel_format_renderer.vs";
    constexpr const char* nv12_fs =                  "shaders/opengl/2D/objects/nv12.fs";
    constexpr const char* yuv420p_fs =               "shaders/opengl/2D/objects/yuv420p.fs";

  #if defined(FAN_3D)
    // 3D
    constexpr const char* line3d_fs =                "shaders/opengl/3D/objects/line.fs";
    constexpr const char* line3d_vs =                "shaders/opengl/3D/objects/line.vs";
    constexpr const char* rectangle3d_fs =           "shaders/opengl/3D/objects/rectangle.fs";
    constexpr const char* rectangle3d_vs =           "shaders/opengl/3D/objects/rectangle.vs";
    constexpr const char* model3d_fs =               "shaders/opengl/3D/objects/model.fs";
    constexpr const char* model3d_vs =               "shaders/opengl/3D/objects/model.vs";
  #endif

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

#endif