module;

#include <fan/utility.h>

#include <cstdint>

#if defined(fan_compiler_gcc)
  #define _GCC_MAX_ALIGN_T
  #define ____mbstate_t_defined
  #define _BITS_PTHREADTYPES_COMMON_H
#endif

// loco framebuffer is recommended, you cant see sprites without it, 
// since light uses framebuffer _t01. you could use unlit_sprite, if required
#define LOCO_FRAMEBUFFER
#if defined(FAN_VULKAN)
  #include <vulkan/vulkan.h>
#endif
#include <fan/event/types.h>

// +cuda
#if __has_include("cuda.h")
  #include <nvcuvid.h>
  //#define loco_cuda
#endif

export module fan.graphics.loco;

import std;

import fan.types;
import fan.types.color;
import fan.types.matrix;

import fan.utility; // engine_functions.h member_offset
import fan.memory;

import fan.event.types;
import fan.event;
import fan.time;
import fan.print.error;

import fan.window.input;

import fan.texture_pack.tp0;

import fan.graphics.common_context;
import fan.graphics.shapes;
import fan.graphics.image_load;
import fan.graphics.opengl.core;
import fan.graphics.input_subsystem;
import fan.graphics.culling;

import fan.noise;

#if defined(FAN_VULKAN)
  import fan.graphics.vulkan.core;
#endif

#if defined(FAN_2D)
  import fan.graphics.shapes.types;
#endif

import fan.console;

#if defined(FAN_GUI)
  import fan.graphics.gui.base;
  import fan.graphics.gui.text_logger;
  //import fan.graphics.gui.settings_menu;
#endif

#if defined(FAN_AUDIO)
  import fan.graphics.audio_subsystem;
#endif

import fan.physics.types;
#if defined(FAN_PHYSICS_2D)
  import fan.graphics.physics_subsystem;
#endif

#if defined(debug_shape_t)
  import fan.print;
#endif

import fan.math.intersection;

#if defined(loco_cuda)
export namespace fan {
  namespace cuda {
    void check_error(auto result) {
      if (result != CUDA_SUCCESS) {
        if constexpr (std::is_same_v<decltype(result), CUresult>) {
          const char* err_str = nullptr;
          cuGetErrorString(result, &err_str);
          fan::throw_error_impl("function failed with:" + std::to_string(result) + ", " + err_str);
        }
        else {
          fan::throw_error_impl("function failed with:" + std::to_string(result) + ", ");
        }
      }
    }
  }
}
export extern "C" {
  extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
}
#endif
// -cuda

export struct loco_t;

struct global_loco_t {
  loco_t* loco = nullptr;
  operator loco_t* () { return loco; }
  global_loco_t& operator=(loco_t* l) {
    loco = l;
    return *this;
  }
  loco_t* operator->() {
    return loco;
  }
};

export global_loco_t& gloco() {
  static global_loco_t loco;
  return loco;
}

export namespace fan::graphics {
  struct engine_init_t {
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix init_callback
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define BLL_set_type_node std::uint16_t
  #define BLL_set_NodeDataType std::function<void(loco_t*)>
  #define BLL_set_CPP_CopyAtPointerChange 1
  #include <BLL/BLL.h>

    using init_callback_nr_t = init_callback_NodeReference_t;
  };
  // cbs called every time engine opens
  engine_init_t::init_callback_t& get_engine_init_cbs() {
    static engine_init_t::init_callback_t engine_init_cbs;
    return engine_init_cbs;
  }

  std::uint32_t get_draw_mode(std::uint8_t internal_draw_mode);
}

export namespace fan {
  enum class stage_fade_mode_t {
    instant,
    crossfade,
    fade_in
  };
}

export namespace fan {
  struct stage_loader_t;

  inline struct gstage_t {
    stage_loader_t* loader;

    operator stage_loader_t*() {
      return loader;
    }
    gstage_t& operator=(stage_loader_t* l) {
      loader = l;
      return *this;
    }
    stage_loader_t* operator->() {
      return loader;
    }
  } gstage;

  struct stage_loader_t {
    #define BLL_set_Link 1
    #define BLL_set_declare_NodeReference 1
    #define BLL_set_declare_rest 0
    #include <fan/fan_bll_preset.h>
    #define BLL_set_AreWeInsideStruct 1
    #define BLL_set_prefix stage_list
    #define BLL_set_type_node std::uint16_t
    #define bcontainer_set_StoreFormat 1
    #define BLL_set_NodeData \
    fan::graphics::update_callback_nr_t update_nr; \
    fan::window_t::resize_callback_NodeReference_t resize_id; \
    void *stage;
    #include <BLL/BLL.h>

    #define bcontainer_set_StoreFormat 1
    #include <fan/fan_bll_preset.h>
    #define BLL_set_prefix cid_list
    #define BLL_set_type_node std::uint32_t
    #define BLL_set_NodeDataType fan::graphics::shape_t
    #define BLL_set_Link 1
    #define BLL_set_AreWeInsideStruct 1
    #include <BLL/BLL.h>

    #define BLL_set_declare_NodeReference 0
    #define BLL_set_declare_rest 1
    #include <fan/fan_bll_preset.h>
    #define BLL_set_prefix stage_list
    #define BLL_set_type_node std::uint16_t
    #define BLL_set_NodeData \
    fan::graphics::update_callback_nr_t update_nr; \
    fan::window_t::resize_callback_NodeReference_t resize_id; \
    void *stage;
    #define BLL_set_Link 1
    #define BLL_set_AreWeInsideStruct 1
    #include <BLL/BLL.h>

    using cid_nr_t = cid_list_NodeReference_t;

    struct nr_t : stage_list_NodeReference_t {
      using base_nr_t = stage_list_NodeReference_t;
      using base_nr_t::base_nr_t;

      nr_t() = default;
      nr_t(base_nr_t nr) : base_nr_t(nr) {}

      void erase() {
        #if FAN_DEBUG >= 2
        if (iic()) {
          fan::throw_error("double erase or uninitialized erase");
        }
        #endif
        gstage->close_stage(*this);
        sic();
      }
    };

    struct stage_open_properties_t {
      fan::graphics::render_view_t* render_view = &fan::graphics::get_orthographic_render_view();

      stage_loader_t::nr_t parent_id;
      std::uint32_t itToDepthMultiplier = 0x100;

      void* sod;
    };

    struct stage_common_t {
      using open_t = void(*)(void*, void*);
      using close_t = void(*)(void*);
      using window_resize_t = void(*)(void*);
      using update_t = void(*)(void*);

      open_t open;
      close_t close;
      window_resize_t window_resize;
      update_t update;
      stage_list_NodeReference_t stage_id;
      std::uint32_t it;
      cid_list_t cid_list;
      stage_loader_t::stage_list_NodeReference_t parent_id;
      fan::graphics::update_callback_nr_t update_nr;
    };

    using key_t = std::pair<void*, std::string>;

    struct pair_hasher_t {
      std::size_t operator()(const key_t& pair) const {
        return std::hash<decltype(pair.first)>()(pair.first) ^ std::hash<std::string>()(pair.second);
      }
    };

    struct pair_equal_t {
      bool operator()(const key_t& lhs, const key_t& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
      }
    };

    using cid_map_t = std::unordered_map<key_t, cid_nr_t, pair_hasher_t, pair_equal_t>;

    stage_loader_t() {
      gstage = this;
    }

    template <typename T>
    static consteval std::string_view get_type_name() {
      #if defined(__clang__)
      constexpr std::string_view p = "[T = ";
      constexpr std::string_view s = "]";
      constexpr std::string_view n = __PRETTY_FUNCTION__;
      #elif defined(__GNUC__)
      constexpr std::string_view p = "[with T = ";
      constexpr std::string_view s = "]";
      constexpr std::string_view n = __PRETTY_FUNCTION__;
      #elif defined(_MSC_VER)
      constexpr std::string_view p = "get_type_name<struct ";
      constexpr std::string_view s = ">(void)";
      constexpr std::string_view n = __FUNCSIG__;
      #else
      return "";
      #endif
      return n.substr(n.find(p) + p.size(), n.rfind(s) - (n.find(p) + p.size()));
    }

    template <typename T>
    nr_t open_stage(const stage_open_properties_t& op);

    template <typename T>
    nr_t open_stage() {
      return open_stage<T>(stage_open_properties_t());
    }

    template <typename T>
    T& get_data() {
      return *static_cast<T*>(stage_list[get<T>()].stage);
    }

    fan::graphics::shape_t& get_id(auto* stage_ptr, const std::string id) {
      auto found = cid_map.find(std::make_pair(stage_ptr, id));
      if (found == cid_map.end()) {
        fan::throw_error("invalid fetch for id - usage shape_{id}", id);
      }
      return stage_ptr->stage_common.cid_list[found->second];
    }

    void load_fgm(auto* stage, const stage_open_properties_t& op, const char* stage_name) {
      std::string filename = std::string("stages_runtime/") + stage_name + ".fgm";
      #define only_struct_data
      #include <fan/graphics/gui/stage_maker/loader_versions/1.h>
    }

    void close_stage(nr_t id);

    template <typename T>
    void restart_stage(nr_t& id) {
      if (id) {
        close_stage(id);
      }
      id = open_stage<T>();
    }

    template <typename T>
    nr_t open() {
      auto key = std::type_index(typeid(T));
      auto it = type_stage_map.find(key);
      nr_t old_nr;
      if (it != type_stage_map.end()) { old_nr = it->second; type_stage_map.erase(it); }
      nr_t nr = open_stage<T>();
      type_stage_map[key] = nr;
      if (old_nr) close_stage(old_nr);
      return nr;
    }

    template <typename T>
    void restart() {
      auto key = std::type_index(typeid(T));
      auto it = type_stage_map.find(key);
      if (it != type_stage_map.end()) {
        close_stage(it->second);
      }
      type_stage_map[key] = open_stage<T>();
    }

    template <typename T>
    void close() {
      auto key = std::type_index(typeid(T));
      auto it = type_stage_map.find(key);
      if (it == type_stage_map.end()) { return; }
      auto nr = it->second;
      type_stage_map.erase(it);
      close_stage(nr);
    }

    template <typename T>
    nr_t get() {
      auto it = type_stage_map.find(std::type_index(typeid(T)));
      if (it == type_stage_map.end()) { fan::throw_error_impl("stage not open"); }
      return it->second;
    }

    template <typename T>
    bool is_open() {
      return type_stage_map.contains(std::type_index(typeid(T)));
    }

    fan::event::task_t change_stage_impl(
      std::function<void()> close_cb,
      std::function<void()> open_cb,
      stage_fade_mode_t mode,
      f32_t duration,
      fan::color color
    );

    template <typename CurrentStageT, typename NextStageT>
    fan::event::task_t change_stage(
      stage_fade_mode_t mode = stage_fade_mode_t::crossfade,
      f32_t duration = 1.f,
      fan::color color = fan::colors::black
    ) {
      return change_stage_impl(
        [this] { close<CurrentStageT>(); },
        [this] { open<NextStageT>(); },
        mode, duration, color
      );
    }

    template <typename NextStageT>
    fan::event::task_t change_stage(
      nr_t old_stage_id,
      nr_t& out_new_stage_id,
      stage_fade_mode_t mode = stage_fade_mode_t::crossfade,
      f32_t duration = 1.f,
      fan::color color = fan::colors::black
    ) {
      return change_stage_impl(
        [this, old_stage_id] { close_stage(old_stage_id); },
        [this, &out_new_stage_id] { out_new_stage_id = open_stage<NextStageT>(); },
        mode, duration, color
      );
    }

    stage_list_t stage_list;
    cid_map_t cid_map;
    std::unordered_map<std::type_index, nr_t> type_stage_map;
    std::string current_stage_name;
    std::string previous_stage_name;
  };

  template <typename Derived>
  struct stage_t {
    using self_t = Derived;

    stage_t(const stage_loader_t::stage_open_properties_t& op = {}) {
      stage_common.open = [](void* ptr, void* sod) {
        if constexpr (requires { static_cast<Derived*>(ptr)->open(sod); }) {
          static_cast<Derived*>(ptr)->open(sod);
        }
      };
      stage_common.close = [](void* ptr) {
        if constexpr (requires { static_cast<Derived*>(ptr)->close(); }) {
          static_cast<Derived*>(ptr)->close();
        }
        delete static_cast<Derived*>(ptr);
      };
      stage_common.update = [](void* ptr) {
        if constexpr (requires { static_cast<Derived*>(ptr)->update(); }) {
          static_cast<Derived*>(ptr)->update();
        }
      };

      auto outside = static_cast<Derived*>(this);
      auto nr = gstage->stage_list.NewNodeLast();
      stage_common.stage_id = nr;
      stage_common.parent_id = op.parent_id;
      gstage->type_stage_map[std::type_index(typeid(Derived))] = nr;
      gstage->stage_list[nr].stage = outside;

      if (stage_common.stage_id.Prev(&gstage->stage_list) != gstage->stage_list.src) {
        stage_common.it = static_cast<stage_loader_t::stage_common_t*>(
          gstage->stage_list[stage_common.stage_id.Prev(&gstage->stage_list)].stage)->it + 1;
      } else {
        stage_common.it = 0;
      }

      stage_common.update_nr = gloco()->m_update_callback.NewNodeFirst();
      gstage->stage_list[stage_common.stage_id].update_nr = stage_common.update_nr;
      gloco()->m_update_callback[stage_common.update_nr] = [outside](void*) {
        outside->stage_common.update(outside);
      };

      auto resize_id = gloco()->window.add_resize_callback([outside](const auto& d) {
        std::printf("todo -- stage common window resize\n");
      });
      gstage->stage_list[stage_common.stage_id].resize_id = resize_id;
    }

    static constexpr std::string_view get_stage_name() {
      if constexpr (requires { Derived::stage_name; }) {
        return Derived::stage_name;
      } else {
        return stage_loader_t::get_type_name<Derived>();
      }
    }

    inline static loco_t* engine = nullptr;
    inline static fan::window_t* window = nullptr;
    inline static fan::physics::context_t* physics = nullptr;
    stage_loader_t::stage_common_t stage_common;
  };
}

export namespace fan {
  using stage_handle_t = stage_loader_t::nr_t;
}

export struct loco_t {
  using key_cb_t = fan::window_t::key_cb_t;
  using key_handle_t = fan::window_t::key_handle_t;
  using keys_handle_t = fan::window_t::keys_handle_t;
  using buttons_handle_t = fan::window_t::buttons_handle_t;
  using mouse_down_handle_t = fan::window_t::mouse_down_handle_t;
  using resize_handle_t = fan::window_t::resize_handle_t;
  using mouse_move_handle_t = fan::window_t::mouse_move_handle_t;

  using buttons_data_t = fan::window_t::buttons_data_t;
  using button_data_t = fan::window_t::button_data_t;
  using mouse_down_data_t = fan::window_t::mouse_down_data_t;
  using mouse_up_data_t = fan::window_t::mouse_up_data_t;
  using mouse_click_data_t = fan::window_t::mouse_click_data_t;
  using keys_data_t = fan::window_t::keys_data_t;
  using key_data_t = fan::window_t::key_data_t;
  using key_down_data_t = fan::window_t::key_down_data_t;
  using key_up_data_t = fan::window_t::key_up_data_t;
  using key_click_data_t = fan::window_t::key_click_data_t;
  using text_callback_handle_t = fan::window_t::text_callback_handle_t;

  using buttons_cb_t = fan::window_t::buttons_cb_t;

  struct properties_t {
    bool render_shapes_top = false;
    bool vsync = true;
    fan::vec2 window_position = -1;
    fan::vec2 window_size = -1;
    std::uint64_t window_flags = 0;
    int window_open_mode = fan::window_t::mode::windowed;
    std::uint8_t renderer = fan::window_t::renderer_t::opengl;
    std::uint8_t samples = 0;
    bool enable_bloom = true;
  }open_props;
  std::int32_t target_fps = 165; // must be changed from function
  bool init_gloco;
  fan::window_t& get_window();
  fan::window_t window; // destruct last

private:
  using shader_t = fan::graphics::shader_nr_t;
  using image_t = fan::graphics::image_nr_t;
  using camera_t = fan::graphics::camera_nr_t;
  using viewport_t = fan::graphics::viewport_nr_t;

  // shared impl for is_mouse_clicked/down and is_key_clicked/down
  bool key_state_is(int key, bool include_repeat);

public:

  std::uint8_t get_renderer();
  fan::graphics::shader_nr_t shader_create();
  fan::graphics::context_shader_t shader_get(fan::graphics::shader_nr_t nr);
  void shader_erase(fan::graphics::shader_nr_t nr);
  void shader_use(fan::graphics::shader_nr_t nr);
  void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code);
  void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code);
  void shader_set_compute(
    fan::graphics::shader_nr_t nr,
    const std::string_view file_path,
    const std::string& compute_code
  );
  bool shader_compile(fan::graphics::shader_nr_t nr);
  template <typename T>
  void shader_set_value(fan::graphics::shader_nr_t nr, const std::string_view name, const T& val) {
    if (0) {}
  #if defined(FAN_OPENGL)
    else if (window.renderer == fan::window_t::renderer_t::opengl) {
      context.gl.shader_set_value(nr, name, val);
    }
  #endif
    else if (window.renderer == fan::window_t::renderer_t::vulkan) {
      fan::throw_error_impl("todo");
    }
  }
  shader_t get_post_process_shader();
  template <typename T>
  void set_post_process(const std::string_view name, T value) {
  #if defined(LOCO_FRAMEBUFFER) && defined(FAN_OPENGL)
    if (window.renderer == fan::window_t::renderer_t::opengl) {
      shader_set_value(get_post_process_shader(), name, value);
    }
  #endif
  }
  #if defined(FAN_2D)
  void shader_set_camera(shader_t nr, camera_t camera_nr);
  fan::graphics::shader_nr_t shader_get_nr(std::uint16_t shape_type);
  fan::graphics::shader_list_t::nd_t& shader_get_data(std::uint16_t shape_type);
  fan::graphics::shader_list_t::nd_t& shader_get_data(fan::graphics::shader_t shader);
  void shader_set_paths(fan::graphics::shader_t shader, std::string_view vertex, std::string_view fragment);
  void shader_recompile_all();
  #endif

  f32_t* get_bloom_filter_radius_ptr();
  f32_t* get_bloom_threshold_ptr();
  f32_t* get_bloom_knee_ptr();
  fan::vec3* get_bloom_tint_ptr();
  void* get_framebuffer();

  fan::graphics::camera_list_t camera_list;
  fan::graphics::shader_list_t shader_list;
  fan::graphics::image_list_t image_list;
  fan::graphics::viewport_list_t viewport_list;

  std::vector<std::uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp = 0, fan::vec2 uvs = 1);
  fan::graphics::image_nr_t image_create();
  fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr);
  std::uint64_t image_get_handle(fan::graphics::image_nr_t nr);
  fan::graphics::image_data_t& image_get_data(fan::graphics::image_nr_t nr);
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
  fan::graphics::image_nr_t create_missing_texture();
  fan::graphics::image_nr_t create_transparent_texture();
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info);
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p);
  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path = std::source_location::current());
  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_nr_t image_create(const fan::color& color);
  fan::graphics::image_nr_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p);

  fan::graphics::camera_nr_t camera_create();
  fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr = fan::graphics::get_orthographic_render_view().camera);
  void camera_erase(fan::graphics::camera_nr_t nr);
  fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y);
  // Returns the raw translation offset of the camera matrix.
  // For an orthographic projection starting at (0,0), this represents the top-left corner.
  // For a symmetric projection (e.g., -width/2 to width/2), this represents the center.
  fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr = fan::graphics::get_orthographic_render_view().camera);
  void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
  void camera_set_position(const fan::vec3& cp);
  // Returns the true world-space center of the camera's view,
  // regardless of how the projection matrix was initialized.
  fan::vec3 camera_get_center(fan::graphics::camera_nr_t nr = fan::graphics::get_orthographic_render_view().camera);
  void camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
  void camera_set_center(const fan::vec3& cp);
  fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr);
  f32_t camera_get_zoom(fan::graphics::camera_nr_t nr = fan::graphics::get_orthographic_render_view().camera);
  void camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom);
  void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y);
  void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size);
  void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset);
  void camera_rotate(const fan::vec2& offset);
  void camera_follow(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed = 10);
  void camera_follow(const fan::vec2& target, f32_t move_speed = 10);

  fan::graphics::viewport_nr_t viewport_create();
  fan::graphics::viewport_nr_t viewport_create(const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr = fan::graphics::get_orthographic_render_view().viewport);
  void viewport_erase(fan::graphics::viewport_nr_t nr);
  fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr = fan::graphics::get_orthographic_render_view().viewport);
  fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr = fan::graphics::get_orthographic_render_view().viewport);
  void viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  void viewport_set_size(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_size);
  void viewport_set_position(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position);
  void viewport_zero(fan::graphics::viewport_nr_t nr);
  bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
  bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
  bool inside(const fan::graphics::render_view_t& render_view, const fan::vec2& position);
  bool is_mouse_inside(const fan::graphics::render_view_t& render_view);

  fan::graphics::context_functions_t context_functions;
  fan::graphics::context_t context;

  // unsafe
  loco_t(const loco_t&) = delete;
  loco_t& operator=(const loco_t&) = delete;
  loco_t(loco_t&&) = delete;
  loco_t& operator=(loco_t&&) = delete;

  void use();
  void camera_move(f32_t movement_speed = 1000.f, f32_t friction = 12);
  void camera_move(fan::graphics::context_camera_t& camera, f32_t movement_speed = 1000.f, f32_t friction = 12);

  #include "shaders.h"
  shaders_t shaders;

#if defined(FAN_OPENGL)
  struct opengl;
  opengl* gl = nullptr;
#endif

#if defined(FAN_VULKAN)
  struct vulkan {
  #include <fan/graphics/vulkan/engine_functions.h>

    fan::vulkan::context_t::descriptor_t d_attachments;
    fan::vulkan::context_t::pipeline_t post_process;
    VkResult image_error = VK_SUCCESS;

    fan::window_t::resize_handle_t window_resize_handle;
  }vk;
#endif

public:

  std::vector<std::function<void()>> m_pre_draw;
  std::vector<std::function<void()>> m_mid_draw;
  std::vector<std::function<void()>> m_mid_single_draw;
  template <typename T>
  void add_custom_draw(T&& cb) {
    m_mid_draw.emplace_back(std::forward<T>(cb));
  }
  template <typename T>
  void add_custom_single_draw(T&& cb) {
    m_mid_single_draw.emplace_back(std::forward<T>(cb));
  }
  std::vector<std::function<void()>> m_post_draw;

#if defined(FAN_2D)
  void add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s);
  std::uint32_t add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s);
  void remove_static_shape_draw(const fan::graphics::shapes::shape_t& s);
#endif

  static void generate_commands(loco_t* loco);

#if defined(FAN_2D)
  fan::vec2 world_min = fan::vec2(-10000);
  fan::vec2 cell_size = fan::vec2(256);
  fan::vec2i grid_size = fan::vec2i(256);
  bool init_culling = true;

  void culling_rebuild_grid();
  void rebuild_static_culling();
  bool culling_enabled() const;
  void set_culling_enabled(bool enabled);
  void get_culling_stats(std::uint32_t& visible, std::uint32_t& culled) const;
  void run_culling();
  void set_cull_padding(const fan::vec2& padding);
  bool is_visualizing_culling = false;
  void visualize_culling();
#endif

#if defined(FAN_VULKAN)
  static void check_vk_result(VkResult err);
#endif

#if defined(FAN_GUI)
  void init_gui();
  void destroy_gui();
#endif

  void bind_global_context();

  loco_t();
  loco_t(const properties_t& p);
  loco_t(std::function<void()> loop_fn);
  loco_t(std::function<void()> loop_fn, const properties_t& p);
  ~loco_t();

  void destroy();
  void close();

  void switch_renderer(std::uint8_t renderer);
  void shapes_draw();
  void process_shapes();
  void process_gui();
  void get_vram_usage(int* total_mem_MB, int* used_MB);

  struct time_monitor_t {
    void update(f32_t v);
    void reset();

    struct stats_t {
      f32_t avg_frame_time_s;
      f32_t min_frame_time_s;
      f32_t max_frame_time_s;

      f32_t avg_fps() const { return 1.0f / avg_frame_time_s; }
      f32_t min_fps() const { return 1.0f / max_frame_time_s; }
      f32_t max_fps() const { return 1.0f / min_frame_time_s; }
    };

    stats_t stats() const;

  #if defined(FAN_GUI)
    void plot(loco_t* loco, std::string_view label);
  #endif

    void toggle_pause() { paused = !paused; }

    std::vector<f32_t> buffer;
    std::deque<int> min_q;
    std::deque<int> max_q;
    f32_t sum = 0.0f;
    bool paused = false;
  };

  struct time_plot_scroll_t {
    int scroll_offset = 0;
    int view_size = 512;
  };

  std::vector<std::function<void()>> draw_end_cb;

  void process_render();
  bool should_close();

  bool process_frame();
  bool process_frame(const std::function<void()>& cb);
  bool process_frame(const std::function<void(f32_t delta_time)>& cb);
  void loop();
  void loop(const std::function<void()>& cb);
  void loop(const std::function<void(f32_t delta_time)>& cb);
  camera_t open_camera(const fan::vec2& x, const fan::vec2& y);
  camera_t open_camera_perspective(f32_t fov = 90.0f, f32_t zmin = 0.1f, f32_t zfar = 10000.f);
  fan::graphics::viewport_t open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  void set_viewport(fan::graphics::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  fan::vec2 get_input_vector(
    const std::string& forward = fan::actions::move_forward,
    const std::string& back = fan::actions::move_back,
    const std::string& left = fan::actions::move_left,
    const std::string& right = fan::actions::move_right
  );
  fan::vec2 get_input_vector(fan::vec2 scalar);
  fan::vec2 transform_matrix(const fan::vec2& position);
  fan::vec2 screen_to_ndc(const fan::vec2& screen_pos);
  fan::vec2 ndc_to_screen(const fan::vec2& ndc_position);
  void set_vsync(bool flag);
  void start_timer();
  static void idle_cb(void* idle_handle);
  void start_idle(bool start_idle = true);
  void update_timer_interval(bool idle = true);
  void set_target_fps(std::int32_t new_target_fps, bool idle = true);
  fan::graphics::context_t& get_context();
  fan::graphics::render_view_t render_view_create();
  fan::graphics::render_view_t render_view_create(
    const fan::vec2& ortho_x, const fan::vec2& ortho_y,
    const fan::vec2& viewport_position, const fan::vec2& viewport_size
  );

  using update_callback_handle_t = fan::graphics::update_callback_t::nr_t;
  // function callback parameter gives loco_t*
  update_callback_handle_t add_update_callback(std::function<void(void*)>&& cb);
  update_callback_handle_t add_update_callback_front(std::function<void(void*)>&& cb);
  void remove_update_callback(update_callback_handle_t handle);
  fan::graphics::update_callback_t m_update_callback;
  std::vector<std::function<void()>> single_queue;

  image_t default_texture;

  void load_engine_images();
  void unload_engine_images();

  fan::graphics::render_view_t orthographic_render_view;
  fan::graphics::render_view_t perspective_render_view;

#if defined(FAN_2D)
  fan::graphics::shapes shapes;
#endif

  void set_window_name(const std::string& name);
  void set_window_icon(const fan::image::info_t& info);
  void set_window_icon(const fan::graphics::image_t& image);

  fan::time::timer start_time;
  f32_t time = 0;

  bool idle_init = false;
  void* idle_handle = nullptr;
  bool timer_init = false;
  void* timer_handle = nullptr;

  std::function<void(f32_t delta_time)> main_loop; // bad, but forced

#define FORWARD_CB_TO_WINDOW(NAME, HANDLE, CBDATA_NAME) \
    HANDLE on_##NAME(int arg, CBDATA_NAME cb) { \
      return window.on_##NAME(arg, std::move(cb)); \
    }

#define FORWARD_CB_TO_WINDOW_NOARG(NAME, HANDLE, CBDATA_NAME) \
    using CBDATA_NAME = fan::window_t::CBDATA_NAME; \
    using NAME##_data_t = fan::window_t::NAME##_data_t; \
    HANDLE on_##NAME(CBDATA_NAME cb) { \
      return window.on_##NAME(std::move(cb)); \
    }

  FORWARD_CB_TO_WINDOW(mouse_click, buttons_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(mouse_down, mouse_down_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(mouse_up, buttons_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(key_click, key_handle_t, key_cb_t);
  FORWARD_CB_TO_WINDOW(key_down, key_handle_t, key_cb_t);
  FORWARD_CB_TO_WINDOW(key_up, key_handle_t, key_cb_t);

  FORWARD_CB_TO_WINDOW_NOARG(mouse_move, mouse_move_handle_t, mouse_move_cb_t);
  FORWARD_CB_TO_WINDOW_NOARG(resize, resize_handle_t, resize_cb_t);

#if defined(FAN_2D)
  void debug_draw_light_buffer();

  // clears shapes after drawing, good for debug draw, not best for performance
  std::vector<fan::graphics::shapes::shape_t> immediate_render_list;
  std::unordered_map<std::uint32_t, fan::graphics::shapes::shape_t> static_render_list;
#endif

  fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport) const;
  fan::vec2 get_mouse_position(const fan::graphics::render_view_t& render_view = fan::graphics::get_orthographic_render_view()) const;
  fan::vec2 get_raw_mouse_position() const;
  fan::vec2 translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera);
  fan::vec2 translate_position(const fan::vec2& p);

  bool is_mouse_clicked(int button = fan::mouse_left);
  bool is_mouse_down(int button = fan::mouse_left);
  bool is_mouse_released(int button = fan::mouse_left);
  fan::vec2 get_mouse_drag(int button = fan::mouse_left);
  bool is_key_clicked(int key);
  bool is_key_down(int key);
  bool is_key_released(int key);

  // input action wrappers
  bool is_active(std::string_view action_name, int pstate = fan::window::input_action_t::press);
  bool is_toggled(std::string_view action_name);
  bool is_toggled(int key);
  bool is_toggled(std::initializer_list<int> keys);
  bool is_clicked(std::string_view action_name);
  bool is_down(std::string_view action_name);
  bool is_released(std::string_view action_name);

#if defined(FAN_2D)
  void shape_open(
    std::uint16_t shape_type,
    std::size_t sizeof_vi,
    std::size_t sizeof_ri,
    fan::graphics::shape_gl_init_list_t shape_shader_locations,
    fan::graphics::shader_t shader,
    fan::graphics::shaper_t::ShapeRenderDataSize_t instance_count = 1,
    bool instanced = true
  );
#endif

  fan::graphics::shader_t get_sprite_shader(const std::string_view fragment_file_path, const std::string& fragment);

  std::string      get_renderer_string();
  std::string_view get_platform_string();
  std::string_view get_build_string();
  std::string_view get_physics_string();

#if defined(FAN_GUI)
  void toggle_console();
  void toggle_console(bool active);
#endif

#if defined(FAN_OPENGL)
  fan::graphics::texture_pack_t texture_pack;
#endif

  fan::graphics::image_load_properties_t default_noise_image_properties();
  fan::graphics::image_t create_noise_image(const fan::vec2& size);
  fan::graphics::image_t create_noise_image(const fan::vec2& size, int seed);
  fan::graphics::image_t create_noise_image(const fan::vec2& size, const std::vector<std::uint8_t>& data);
  fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const;
  fan::vec2 convert_mouse_to_ndc() const;
  fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view);
  fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view);

#if defined(loco_cuda)
  struct cuda_textures_t {
    void close(loco_t* loco, fan::graphics::shapes::shape_t& cid);
    void resize(loco_t* loco, fan::graphics::shapes::shape_t& id, std::uint8_t format, fan::vec2ui size);
    cudaArray_t& get_array(std::uint32_t index_t);
    bool inited = false;
    struct graphics_resource_t {
      void open(int texture_id);
      void close();
      void map();
      void unmap();
      cudaGraphicsResource_t resource = nullptr;
      cudaArray_t cuda_array = nullptr;
    };
    graphics_resource_t wresources[4];
  };
#endif

  struct renderer_state_t {
    fan::color clear_color = {0.f, 0.f, 0.f, 1.f};
    fan::graphics::lighting_t lighting;
  #if defined(FAN_2D)
    bool force_line_draw = false;
  #endif
    bool render_shapes_top = false;
    // -1 no reload, opengl = 0 etc
    std::uint8_t reload_renderer_to = -1;
  } renderer_state;

  fan::graphics::lighting_t& get_lighting()          { return renderer_state.lighting; }
  bool&                      get_render_shapes_top() { return renderer_state.render_shapes_top; }
  fan::color&                get_clear_color() { return renderer_state.clear_color; }
  void                       set_clear_color(const fan::color& color) { get_clear_color() = color; }

  
  struct timing_t {
    fan::time::timer shape_draw_timer;
    fan::time::timer gui_draw_timer;
    f64_t shape_draw_time_s = 0;
    f64_t gui_draw_time_s = 0;
    fan::time::timer frame_timer {true};
    f64_t target_frame_time = 0.0;
    f64_t accumulated_time = 0.0;
    bool timer_enabled = true;
    bool vsync = false;
  } timing;

  bool&  get_vsync()       { return timing.vsync; }
  f64_t& get_delta_time()  { return window.m_delta_time; }

  // input
  fan::graphics::input_subsystem_t input;
  fan::window::input_action_t& get_input_action() { return input.input_action; }

  fan::stage_loader_t stage_loader;

  template <typename T>
  fan::stage_handle_t stage_open() { return stage_loader.open<T>(); }

  template <typename T>
  void stage_close() { stage_loader.close<T>(); }

  template <typename T>
  void stage_restart() { stage_loader.restart<T>(); }

  template <typename CurrentStageT, typename NextStageT>
  void stage_change( 
    fan::stage_fade_mode_t mode = fan::stage_fade_mode_t::crossfade,
    f32_t duration = 1.f,
    fan::color color = fan::colors::black) 
  { 
    fan::event::add_awaitable(stage_loader.change_stage<CurrentStageT, NextStageT>(mode, duration, color)); 
  }

  template <typename CurrentStageT, typename NextStageT>
  void stage_change( 
    fan::stage_handle_t old_stage_id,
    fan::stage_handle_t& out_new_stage_id,
    fan::stage_fade_mode_t mode = fan::stage_fade_mode_t::crossfade,
    f32_t duration = 1.f,
    fan::color color = fan::colors::black)
  { 
    fan::event::add_awaitable(stage_loader.change_stage<CurrentStageT, NextStageT>(
      old_stage_id, out_new_stage_id, mode, duration, color)); 
  }

  template <typename T>
  T& stage_get() { return stage_loader.get_data<T>(); }

  template <typename T>
  bool is_stage_open() { return stage_loader.is_open<T>(); }

  struct gui_state_t {
#if defined(FAN_GUI)
    void* settings_menu = nullptr;
    std::future<void> font_future;
#endif
    fan::console_t console;
  #if defined(FAN_GUI)
    fan::time::timer fps_timer;
    std::uint32_t frame_count = 0;
    std::uint32_t last_fps = 0;
    bool render_console = false;
    bool render_debug_memory = false;
    bool show_fps = false;
    bool render_settings_menu = 0;
    bool allow_docking = true;
    bool gui_initialized = false;
    fan::graphics::gui::text_logger_t text_logger;
    fan::graphics::gui_draw_cb_t gui_draw_cb;
    time_monitor_t frame_monitor;
    time_monitor_t shape_monitor;
    time_monitor_t gui_monitor;
    time_plot_scroll_t time_plot_scroll;
    bool enable_overlay = true;
  #endif
  } gui;

#if defined(FAN_GUI)
  void*                                 get_settings_menu()        { return gui.settings_menu; }
  bool&                                 get_render_settings_menu() { return gui.render_settings_menu; }
  bool&                                 get_show_fps()             { return gui.show_fps; }
  bool&                                 get_allow_docking()        { return gui.allow_docking; }
  bool&                                 get_enable_overlay()       { return gui.enable_overlay; }
#endif
  fan::console_t&                       get_console()              { return gui.console; }

#if defined(FAN_AUDIO)
  fan::graphics::audio_subsystem_t audio;
#endif

  #if defined(FAN_PHYSICS_2D)
    fan::graphics::physics_subsystem_t physics;
    void update_physics(bool flag);
    fan::physics::context_t& get_physics_context() { return physics.context; }
  #endif

  fan::graphics::image_t get_color_buffer(int idx);

#if defined(FAN_2D)
  void camera_move_to(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view);
  void camera_move_to(const fan::graphics::shapes::shape_t& shape);
  void camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view);
  void camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape);
  bool shader_update_fragment(std::uint16_t shape_type, const std::string_view fragment_file_path, const std::string& fragment);
#endif
}; // loco_t

template <typename T>
inline fan::stage_loader_t::nr_t fan::stage_loader_t::open_stage(const stage_open_properties_t& op) {
  if (!T::engine)  T::engine = gloco();
  if (!T::window)  T::window = &gloco()->window;
  if (!T::physics) T::physics = &gloco()->get_physics_context();

  previous_stage_name = current_stage_name;
  if constexpr (requires { T::stage_name; }) {
    current_stage_name = T::stage_name;
  } else {
    current_stage_name = get_type_name<T>();
  }

  T* stage = new T{op};
  stage->stage_common.open(stage, op.sod);
  return stage->stage_common.stage_id;
}

fan::event::task_t fan::stage_loader_t::change_stage_impl(
  std::function<void()> close_cb,
  std::function<void()> open_cb,
  fan::stage_fade_mode_t mode,
  f32_t duration,
  fan::color color
) {
  if (mode == fan::stage_fade_mode_t::instant || duration <= 0.f) {
    close_cb();
    open_cb();
    co_return;
  }
  fan::graphics::shape_t overlay{fan::graphics::shapes::rectangle_t::properties_t{
    .position = fan::vec3(0, 0, 0xfffe),
    .size = fan::vec2(99999),
    .color = color.set_alpha(mode == fan::stage_fade_mode_t::fade_in ? 1.f : 0.f),
    .blending = true,
  }};
  f32_t half = (mode == fan::stage_fade_mode_t::crossfade) ? duration / 2.f : duration;
  if (mode == fan::stage_fade_mode_t::crossfade) {
    for (f32_t t = 0.f; t < half; t += gloco()->get_delta_time()) {
      overlay.set_color(color.set_alpha(t / half));
      co_await fan::graphics::co_next_frame();
    }
  }
  close_cb();
  open_cb();
  for (f32_t t = 0.f; t < half; t += gloco()->get_delta_time()) {
    overlay.set_color(color.set_alpha(1.f - t / half));
    co_await fan::graphics::co_next_frame();
  }
}

void fan::stage_loader_t::close_stage(nr_t id) {
  auto* sc = (stage_common_t*)stage_list[id].stage;
  auto update_nr = stage_list[id].update_nr;
  gloco()->m_update_callback.unlrec(update_nr);
  sc->close(stage_list[id].stage);
  stage_list.unlrec(id);
}

#if defined(FAN_VULKAN)
#include <fan/graphics/vulkan/uniform_block.h>
#endif

#if defined(FAN_GUI)
namespace fan {
  namespace graphics {
    using texture_packe0 = fan::graphics::texture_pack::internal_t;
    using ti_t = fan::graphics::texture_pack::ti_t;
  }
}
#endif

export namespace fan::graphics {
  using engine_t = loco_t;
  void shader_set_camera(fan::graphics::shader_t nr, fan::graphics::camera_t camera_nr);
}

export namespace fan {
  template <typename Derived>
  struct frame_task_t {
    using self_t = Derived;

    frame_task_t() {
      update_nr = gloco()->m_update_callback.NewNodeFirst();
      gloco()->m_update_callback[update_nr] = [this](void*) {
        if constexpr (requires{static_cast<Derived*>(this)->update(); }) {
          static_cast<Derived*>(this)->update();
        }
      };
    }

    ~frame_task_t() {
      if (update_nr) {
        gloco()->m_update_callback.unlrec(update_nr);
        update_nr.sic();
      }
    }

    fan::graphics::update_callback_nr_t update_nr;
  };
}