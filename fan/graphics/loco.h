#pragma once

#include <fan/graphics/loco_settings.h>
#include <fan/graphics/file_dialog.h>

// remove
#define loco_box2d

#if defined(loco_box2d)
  #include <fan/physics/b2_integration.hpp>
#endif

#define loco_opengl
#define loco_framebuffer
#define loco_post_process
#define loco_vfi
#define loco_physics

#include <uv.h>
#undef min
#undef max
//
#include <fan/window/window.h>
#include <fan/graphics/opengl/gl_core.h>
#include <fan/io/file.h>

#if defined(loco_imgui)
#include <fan/imgui/imgui.h>
#include <fan/imgui/imgui_impl_opengl3.h>
#include <fan/imgui/imgui_impl_glfw.h>
#include <fan/imgui/imgui_neo_sequencer.h>
#include <fan/imgui/implot.h>
#endif

#include <fan/physics/collision/rectangle.h>

#include <fan/graphics/algorithm/FastNoiseLite.h>

#if defined(loco_imgui)
#include <fan/graphics/console.h>
#endif

#if defined(loco_json)

#include <fan/io/json_impl.h>

struct loco_t;

namespace fan {
  using namespace nlohmann;
}

namespace nlohmann {

  template <typename T>
  struct nlohmann::adl_serializer<fan::vec2_wrap_t<T>> {
    static void to_json(nlohmann::json& j, const fan::vec2_wrap_t<T>& v) {
      j = nlohmann::json{ v.x, v.y };
    }
    static void from_json(const nlohmann::json& j, fan::vec2_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
    }
  };

  template <typename T>
  struct nlohmann::adl_serializer<fan::vec3_wrap_t<T>> {
    static void to_json(nlohmann::json& j, const fan::vec3_wrap_t<T>& v) {
      j = nlohmann::json{ v.x, v.y, v.z };
    }
    static void from_json(const nlohmann::json& j, fan::vec3_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
      v.z = j[2].get<T>();
    }
  };

  template <typename T>
  struct nlohmann::adl_serializer<fan::vec4_wrap_t<T>> {
    static void to_json(nlohmann::json& j, const fan::vec4_wrap_t<T>& v) {
      j = nlohmann::json{ v.x, v.y, v.z, v.w };
    }
    static void from_json(const nlohmann::json& j, fan::vec4_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
      v.z = j[2].get<T>();
      v.w = j[3].get<T>();
    }
  };

  template <> struct adl_serializer<fan::color> {
    static void to_json(json& j, const fan::color& c) {
      j = json{ c.r, c.g, c.b, c.a };
    }
    static void from_json(const json& j, fan::color& c) {
      c.r = j[0];
      c.g = j[1];
      c.b = j[2];
      c.a = j[3];
    }
  };
}

namespace fan {
  struct json_stream_parser_t {
    std::string buf;
  
    struct parsed_result {
      bool success;
      fan::json value;
      std::string error;
    };

    [[nodiscard]] 
    std::pair<size_t, size_t> find_next_json_bounds(std::string_view s, size_t pos = 0) const noexcept;

    std::vector<parsed_result> process(std::string_view chunk);

    void clear() noexcept { buf.clear(); }
  };
}

#endif

#include <fan/tp/tp0.h>

#define loco_line
#define loco_rectangle
#define loco_sprite
#define loco_light
#define loco_circle
#define loco_responsive_text
#define loco_universal_image_renderer


#if defined(loco_cuda)

// +cuda
#include "cuda_runtime.h"
#include <cuda.h>
#include <nvcuvid.h>


namespace fan {
  namespace cuda {
    void check_error(auto result) {
      if (result != CUDA_SUCCESS) {
        if constexpr (std::is_same_v<decltype(result), CUresult>) {
          const char* err_str = nullptr;
          cuGetErrorString(result, &err_str);
          fan::throw_error("function failed with:" + std::to_string(result) + ", " + err_str);
        }
        else {
          fan::throw_error("function failed with:" + std::to_string(result) + ", ");
        }
      }
    }
  }
}

extern "C" {
  extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource** resource, fan::opengl::GLuint image, fan::opengl::GLenum target, unsigned int flags);
}

#endif
// -cuda

//#define debug_shape_t

struct loco_t;

// to set new loco use gloco = new_loco;
struct global_loco_t {

  loco_t* loco = nullptr;

  operator loco_t* ();
  global_loco_t& operator=(loco_t* l);
  loco_t* operator->() {
    return loco;
  }
};
// might crash if pch or lib is built with extern/inline so if its different, 
// it will crash in random places
#if defined(fan_compiler_clang)
  inline global_loco_t gloco;
#elif defined(fan_compiler_msvc)
  inline global_loco_t gloco;
#endif

namespace fan {
  void printcl(auto&&... values);
  void printclh(int highlight = 0, auto&&... values);
}

struct loco_t : fan::opengl::context_t {

  static uint8_t* A_resize(void* ptr, uintptr_t size);

  static constexpr uint32_t MaxElementPerBlock = 0x1000;

  struct shape_gl_init_t {
    std::pair<int, const char*> index;
    uint32_t size;
    uint32_t type; // for example GL_FLOAT
    uint32_t stride;
    void* pointer;
  };

  #define shaper_set_MaxMaxElementPerBlock 0x1000
  #define shaper_set_fan 1
  // sizeof(image_t) == 2
  static_assert(sizeof(loco_t::image_t) != 2, "update shaper_set_MaxKeySize");
  #define shaper_set_MaxKeySize 2 * 30
  #include <fan/graphics/shaper.h>

  template<
    typename... Ts,
    uintptr_t s = (sizeof(Ts) + ...)
  >static constexpr shaper_t::ShapeID_t shape_add(
    shaper_t::ShapeTypeIndex_t sti,
    const auto& rd,
    const auto& d,
    Ts... args
  ) {
    struct structarr_t {
      uint8_t p[s];
      uint8_t& operator[](uintptr_t i) {
        return p[i];
      }
    };
    structarr_t a;
    uintptr_t i = 0;
    ([&](auto arg) {
      __MemoryCopy(&arg, &a[i], sizeof(arg));
      i += sizeof(arg);
      }(args), ...);
    constexpr uintptr_t count = (!!(sizeof(Ts) + 1) + ...);
    static_assert(count % 2 == 0);
    constexpr uintptr_t last_sizeof = (static_cast<uintptr_t>(0), ..., sizeof(Ts));
    uintptr_t LastKeyOffset = s - last_sizeof - 1;
    gloco->shaper.PrepareKeysForAdd(&a, LastKeyOffset);
    return gloco->shaper.add(sti, &a, s, &rd, &d);
  }


  loco_t(const loco_t&) = delete;
  loco_t& operator=(const loco_t&) = delete;
  loco_t(loco_t&&) = delete;
  loco_t& operator=(loco_t&&) = delete;

  struct shape_type_t {
    enum {
      invalid = -1,
      // render order
      // make sure shape.open() has same order - TODO remove shape.open - use shape_functions[i].open
      button,
      sprite = 1,
      text,
      hitbox,
      line,
      mark,
      rectangle,
      light,
      unlit_sprite,
      circle,
      capsule,
      grid,
      vfi,
      particles,
      universal_image_renderer,
      gradient,
      light_end,
      shader_shape,
      rectangle3d,
      line3d,
      last
    };
  };

  struct kp {
    enum {
      light,
      common,
      vfi,
      texture,
    };
  };

  static constexpr const char* shape_names[] = {
    "button",
    "sprite",
    "text",
    "hitbox",
    "line",
    "mark",
    "rectangle",
    "light",
    "unlit_sprite",
    "circle",
    "grid",
    "vfi",
    "particles",
  };

#if defined (loco_imgui)
  using console_t = fan::console_t;
#endif

  using blending_t = uint8_t;
  using depth_t = uint16_t;

  void use();

  using camera_t = fan::opengl::context_t::camera_nr_t;
  void camera_move(fan::opengl::context_t::camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction);

  uint32_t fb_vao;
  uint32_t fb_vbo;
  void render_final_fb();
  void initialize_fb_vaos(uint32_t& vao, uint32_t& vbo);

  using texture_packe0 = fan::graphics::texture_packe0;

  using viewport_t = fan::opengl::context_t::viewport_nr_t;
  using image_t = fan::opengl::context_t::image_nr_t;

  using shader_t = fan::opengl::context_t::shader_nr_t;

  struct shape_t;

  #include <fan/graphics/opengl/texture_pack.h>

  using push_back_cb = shape_t (*)(void*);
  using set_position2_cb = void (*)(shape_t*, const fan::vec2&);
  // depth
  using set_position3_cb = void (*)(shape_t*, const fan::vec3&);
  using set_size_cb = void (*)(shape_t*, const fan::vec2&);
  using set_size3_cb = void (*)(shape_t*, const fan::vec3&);

  using get_position_cb = fan::vec3 (*)(shape_t*);
  using get_size_cb = fan::vec2 (*)(shape_t*);
  using get_size3_cb = fan::vec3 (*)(shape_t*);

  using set_rotation_point_cb = void (*)(shape_t*, const fan::vec2&);
  using get_rotation_point_cb = fan::vec2 (*)(shape_t*);

  using set_color_cb = void (*)(shape_t*, const fan::color&);
  using get_color_cb = fan::color (*)(shape_t*);

  using set_angle_cb = void (*)(shape_t*, const fan::vec3&);
  using get_angle_cb = fan::vec3 (*)(shape_t*);

  using get_tc_position_cb = fan::vec2 (*)(shape_t*);
  using set_tc_position_cb = void (*)(shape_t*, const fan::vec2&);

  using get_tc_size_cb = fan::vec2 (*)(shape_t*);
  using set_tc_size_cb = void (*)(shape_t*, const fan::vec2&);

  using load_tp_cb = bool(*)(shape_t*, loco_t::texturepack_t::ti_t*);

  using get_grid_size_cb = fan::vec2 (*)(shape_t*);
  using set_grid_size_cb = void (*)(shape_t*, const fan::vec2&);

  using get_camera_cb = loco_t::camera_t (*)(shape_t*);
  using set_camera_cb = void (*)(shape_t*, loco_t::camera_t);

  using get_viewport_cb = loco_t::viewport_t (*)(shape_t*);
  using set_viewport_cb = void (*)(shape_t*, loco_t::viewport_t);


  using get_image_cb = loco_t::image_t(*)(shape_t*);
  using set_image_cb = void (*)(shape_t*, loco_t::image_t);

  using get_parallax_factor_cb = f32_t (*)(shape_t*);
  using set_parallax_factor_cb = void (*)(shape_t*, f32_t);
  using get_rotation_vector_cb = fan::vec3 (*)(shape_t*);
  using get_flags_cb = uint32_t (*)(shape_t*);
  using set_flags_cb = void(*)(shape_t*, uint32_t);
  //
  using get_radius_cb = f32_t (*)(shape_t*);
  using get_src_cb = fan::vec3 (*)(shape_t*);
  using get_dst_cb = fan::vec3 (*)(shape_t*);
  using get_outline_size_cb = f32_t (*)(shape_t*);
  using get_outline_color_cb = fan::color (*)(shape_t*);

  using reload_cb = void (*)(shape_t*, uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter); 

  using draw_cb = void (*)(uint8_t draw_range);

  using set_line_cb = void (*)(shape_t*, const fan::vec2&, const fan::vec2&);
  using set_line3_cb = void (*)(shape_t*, const fan::vec3&, const fan::vec3&);

  struct functions_t {
    push_back_cb push_back;

    get_position_cb get_position;
    set_position2_cb set_position2;
    set_position3_cb set_position3;

    get_size_cb get_size;
    get_size3_cb get_size3;
    set_size_cb set_size;
    set_size3_cb set_size3;

    get_rotation_point_cb get_rotation_point;
    set_rotation_point_cb set_rotation_point;

    get_color_cb get_color;
    set_color_cb set_color;

    get_angle_cb get_angle;
    set_angle_cb set_angle;

    get_tc_position_cb get_tc_position;
    set_tc_position_cb set_tc_position;

    get_tc_size_cb get_tc_size;
    set_tc_size_cb set_tc_size;

    load_tp_cb load_tp;

    get_grid_size_cb get_grid_size;
    set_grid_size_cb set_grid_size;

    get_camera_cb get_camera;
    set_camera_cb set_camera;

    get_viewport_cb get_viewport;
    set_viewport_cb set_viewport;

    get_image_cb get_image;
    set_image_cb set_image;

    get_parallax_factor_cb get_parallax_factor;
    set_parallax_factor_cb set_parallax_factor;
    get_rotation_vector_cb get_rotation_vector;


    get_flags_cb get_flags;
    set_flags_cb set_flags;

    get_radius_cb get_radius;
    get_src_cb get_src;
    get_dst_cb get_dst;
    get_outline_size_cb get_outline_size;
    get_outline_color_cb get_outline_color;

    reload_cb reload;

    draw_cb draw;

    set_line_cb set_line;
    set_line3_cb set_line3;
  };

  template <typename T, typename T2>
  static T2& get_render_data(shape_t* shape, T2 T::* attribute) {
    shaper_t::ShapeRenderData_t* data = gloco->shaper.GetRenderData(*shape);
    return ((T*)data)->*attribute;
  }

  template <typename T, typename T2, typename T3>
  static void modify_render_data_element(shape_t* shape, T2 T::* attribute, const T3& value) {
    shaper_t::ShapeRenderData_t* data = gloco->shaper.GetRenderData(*shape);

    if ((gloco->opengl.major > 3) || (gloco->opengl.major == 3 && gloco->opengl.minor >= 3)) {
      ((T*)data)->*attribute = value;
      gloco->shaper.ElementIsPartiallyEdited(
        gloco->shaper.GetSTI(*shape),
        gloco->shaper.GetBLID(*shape),
        gloco->shaper.GetElementIndex(*shape),
        fan::member_offset(attribute),
        sizeof(T3)
      );
    }
    else {
      for (int i = 0; i < 6; ++i) {
        auto& v = ((T*)data)[i];
        ((T*)&v)->*attribute = value;
        gloco->shaper.ElementIsPartiallyEdited(
          gloco->shaper.GetSTI(*shape),
          gloco->shaper.GetBLID(*shape),
          gloco->shaper.GetElementIndex(*shape),
          fan::member_offset(attribute) + sizeof(T) * i,
          sizeof(T3)
        );
      }
    }
  };

  template <typename T>
  static functions_t get_functions();

#pragma pack(push, 1)

#define st(name, inside) \
  template <bool cond> \
  struct CONCAT(name, _cond) { \
    template <typename T> \
    using d = typename fan::type_or_uint8_t<cond>::template d<T>; \
    inside \
  }; \
  using name = CONCAT(name, _cond)<1>; \
  struct CONCAT(_, name) : CONCAT(name, _cond<0>) {};

  using multitexture_image_t = std::array<loco_t::image_t, 30>;

  struct kps_t {
    st(light_t,
      d<uint8_t> genre;
      d<loco_t::viewport_t> viewport;
      d<loco_t::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
    );
    st(common_t,
      d<depth_t> depth;
      d<blending_t> blending;
      d<loco_t::viewport_t> viewport;
      d<loco_t::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
    );
    st(vfi_t,
      d<uint8_t> filler = 0;
    );
    st(texture_t,
      d<depth_t> depth;
      d<blending_t> blending;
      d<loco_t::image_t> image;
      d<loco_t::viewport_t> viewport;
      d<loco_t::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
    );
    // for universal_image_renderer
    // struct texture4_t {
    //   blending_t blending;
    //   depth_t depth;
    //   loco_t::image_t image; // 4 - 1
    //   loco_t::viewport_t viewport;
    //   loco_t::camera_t camera;
    //   shaper_t::ShapeTypeIndex_t ShapeType;
    // };
  };

#undef st
#pragma pack(pop)

  struct shape_info_t {
    functions_t functions;
  };

private:
  std::vector<shape_info_t> shape_info_list;
public:

  std::vector<fan::function_t<void()>> m_pre_draw;
  std::vector<fan::function_t<void()>> m_post_draw;

  struct properties_t {
    bool vsync = true;
    fan::vec2 window_size = -1;
    uint64_t window_flags = 0;
  };

  uint64_t start_time = 0;

  void init_framebuffer();

  loco_t();
  loco_t(const properties_t& p);
  ~loco_t();

  void draw_shapes();
  void process_frame();

  bool should_close();
  void should_close(int flag);

  bool process_loop(const fan::function_t<void()>& lambda = [] {});
  void loop(const fan::function_t<void()>& lambda);

  loco_t::camera_t open_camera(const fan::vec2 & x, const fan::vec2 & y);
  loco_t::camera_t open_camera_perspective(f32_t fov = 90.0f);
  
  loco_t::viewport_t open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size);

  void set_viewport(loco_t::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size);

  // for checking whether you set depth or no
  struct position3_t : public fan::vec3 {
    using fan::vec3::vec3;
    using fan::vec3::operator=;
    position3_t& operator=(const position3_t& p) {
      fan::vec3::operator=(p);
      return *this;
    }
  };


  //
  fan::vec2 transform_matrix(const fan::vec2& position);

  fan::vec2 screen_to_ndc(const fan::vec2& screen_pos);

  fan::vec2 ndc_to_screen(const fan::vec2& ndc_position);
  //

  void set_vsync(bool flag);
  void start_timer();
  void start_idle();
  void update_timer_interval();
  void set_target_fps(int32_t fps);

  //-----------------------------gui-----------------------------

#if defined(loco_imgui)
protected:
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix imgui_draw_cb
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType fan::function_t<void()>
#include <BLL/BLL.h>
public:

  using imgui_draw_cb_nr_t = imgui_draw_cb_NodeReference_t;
  imgui_draw_cb_t m_imgui_draw_cb;

  struct imgui_element_nr_t : loco_t::imgui_draw_cb_nr_t {
    using base_t = loco_t::imgui_draw_cb_nr_t;

    imgui_element_nr_t() = default;

    imgui_element_nr_t(const imgui_element_nr_t& nr);

    imgui_element_nr_t(imgui_element_nr_t&& nr);
    ~imgui_element_nr_t();


    imgui_element_nr_t& operator=(const imgui_element_nr_t& id);

    imgui_element_nr_t& operator=(imgui_element_nr_t&& id);

    void init();

    bool is_invalid() const;

    void invalidate_soft();

    void invalidate();

    inline void set(const auto& lambda) {
      gloco->m_imgui_draw_cb[*this] = lambda;
    }
  };

  struct imgui_element_t : imgui_element_nr_t {
    imgui_element_t() = default;
    imgui_element_t(const auto& lambda) {
      imgui_element_nr_t::init();
      imgui_element_nr_t::set(lambda);
    }
  };

#if !defined(__INTELLISENSE__)
#define fan_imgui_dragfloat_named(name, variable, speed, m_min, m_max) \
  [&] <typename T5>(T5& var) -> bool{ \
    fan::string label(name); \
    if constexpr(std::is_same_v<f32_t, T5>)  { \
      return ImGui::DragFloat(label.c_str(), &var, (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec2, T5>)  { \
      return ImGui::DragFloat2(label.c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec3, T5>)  { \
      return ImGui::DragFloat3(label.c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec4, T5>)  { \
      return ImGui::DragFloat4(label.c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::quat, T5>)  { \
      return ImGui::DragFloat4(label.c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::color, T5>)  { \
      return ImGui::DragFloat4(label.c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else {\
      fan::throw_error_impl(); \
      return 0; \
    } \
  }(variable)

#endif

#define fan_imgui_dragfloat(variable, speed, m_min, m_max) \
    fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, m_min, m_max)


#define fan_imgui_dragfloat1(variable, speed) \
    fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, 0, 0)

  struct imgui_fs_var_t {
    loco_t::imgui_element_t ie;

    imgui_fs_var_t() = default;

    template <typename T>
    imgui_fs_var_t(
      loco_t::shader_t shader_nr,
      const fan::string& var_name,
      T initial_ = 0,
      f32_t speed = 1,
      f32_t min = -100000,
      f32_t max = 100000
    );
  };

  static const char* item_getter1(const std::vector<std::string>& items, int index) {
    if (index >= 0 && index < (int)items.size()) {
      return items[index].c_str();
    }
    return "N/A";
  }

  void set_imgui_viewport(loco_t::viewport_t viewport);

#endif
  //-----------------------------gui-----------------------------

  fan::opengl::context_t& get_context();

  struct camera_impl_t {

    camera_impl_t() = default;
    loco_t::camera_t camera;
    loco_t::viewport_t viewport;
  };

  struct input_action_t {
    enum {
      none = -1,
      release = (int)fan::keyboard_state::release,
      press = (int)fan::keyboard_state::press,
      repeat = (int)fan::keyboard_state::repeat,
      press_or_repeat
    };

    struct action_data_t {
      static constexpr int max_keys_per_action = 5;
      int keys[max_keys_per_action]{};
      uint8_t count = 0;
      static constexpr int max_keys_combos = 5;
      int key_combos[max_keys_combos]{};
      uint8_t combo_count = 0;
    };

    void add(const int* keys, std::size_t count, std::string_view action_name);
    void add(int key, std::string_view action_name);
    void add(std::initializer_list<int> keys, std::string_view action_name);

    void add_keycombo(std::initializer_list<int> keys, std::string_view action_name);

    bool is_active(std::string_view action_name, int state = loco_t::input_action_t::press);
    bool is_action_clicked(std::string_view action_name);
    bool is_action_down(std::string_view action_name);

    std::unordered_map<std::string_view, action_data_t> input_actions;
  }input_action;

protected:
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix update_callback
  #include _FAN_PATH(fan_bll_preset.h)
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::function_t<void(loco_t*)>
  #include <BLL/BLL.h>
public:

  using update_callback_nr_t = update_callback_NodeReference_t;

  update_callback_t m_update_callback;

  std::vector<fan::function_t<void()>> single_queue;

  image_t default_texture;

  camera_impl_t orthographic_camera;
  camera_impl_t perspective_camera;

  fan::window_t window;
  uv_idle_t idle_handle;
  uv_timer_t timer_handle;
  private:
  int32_t target_fps = 165; // must be changed from function
  bool timer_enabled = target_fps > 0;
public:
  fan::function_t<void()> main_loop; // bad, but forced

  f64_t& delta_time = window.m_delta_time;

  std::vector<functions_t> shape_functions;

  // needs continous buffer
  std::vector<shaper_t::BlockProperties_t> BlockProperties;

  shaper_t shaper;
  
#if defined(loco_box2d)
  fan::physics::context_t physics_context{{}};
  struct physics_update_data_t {
    shaper_t::ShapeID_t shape_id;
    b2BodyId body_id;
    void* cb;
  };
  using shape_physics_update_cb = void(*)(const physics_update_data_t& data);
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix physics_update_cbs
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType physics_update_data_t
  #include <BLL/BLL.h>
  physics_update_cbs_t::nr_t add_physics_update(const physics_update_data_t& cb_data);
  void remove_physics_update(physics_update_cbs_t::nr_t nr);
  physics_update_cbs_t shape_physics_update_cbs;
#endif

#pragma pack(push, 1)

  struct Key_e {
    enum : shaper_t::KeyTypeIndex_t{
      light,
      light_end,
      blending,
      depth,
      image,
      viewport,
      camera,
      ShapeType,
      filler
    };
  };

#pragma pack(pop)

  fan::vec2 get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport);
  fan::vec2 get_mouse_position();

  static fan::vec2 translate_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera);
  fan::vec2 translate_position(const fan::vec2& p);

  struct shape_t : shaper_t::ShapeID_t{
    shape_t() {
      sic();
    }
    shape_t(shaper_t::ShapeID_t&& s) {
      NRI = s.NRI;
      s.sic();
    }
    shape_t(const shaper_t::ShapeID_t& s) : shape_t() {

      if (s.iic()) {
        return;
      }

      {
        auto sti = gloco->shaper.GetSTI(s);

        // alloc can be avoided inside switch
        uint8_t* KeyPack = new uint8_t[gloco->shaper.GetKeysSize(s)];
        gloco->shaper.WriteKeys(s, KeyPack);


        auto _vi = gloco->shaper.GetRenderData(s);
        auto vlen = gloco->shaper.GetRenderDataSize(sti);
        uint8_t* vi = new uint8_t[vlen];
        std::memcpy(vi, _vi, vlen);

        auto _ri = gloco->shaper.GetData(s);
        auto rlen = gloco->shaper.GetDataSize(sti);
        uint8_t* ri = new uint8_t[rlen];
        std::memcpy(ri, _ri, rlen);

        *this = gloco->shaper.add(
          sti, 
          KeyPack,
          gloco->shaper.GetKeysSize(s),
          vi, 
          ri
        );
#if defined(debug_shape_t)
        fan::print("+", NRI);
#endif
        delete[] KeyPack;
        delete[] vi;
        delete[] ri;
      }
    }

    template <typename T>
    requires requires(T t) { typename T::type_t; }
    shape_t(const T& properties) : shape_t() {
      if constexpr (std::is_same_v<T, light_t::properties_t>) {
        *this = gloco->light.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, line_t::properties_t>) {
        *this = gloco->line.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, rectangle_t::properties_t>) {
        *this = gloco->rectangle.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, sprite_t::properties_t>) {
        *this = gloco->sprite.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, unlit_sprite_t::properties_t>) {
        *this = gloco->unlit_sprite.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, circle_t::properties_t>) {
        if constexpr (fan_has_variable(loco_t, circle)) {
          *this = gloco->circle.push_back(properties);
        }
      }
      else if constexpr (std::is_same_v<T, capsule_t::properties_t>) {
        *this = gloco->capsule.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, grid_t::properties_t>) {
        *this = gloco->grid.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::vfi_t::common_shape_properties_t>) {
        *this = gloco->vfi.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::particles_t::properties_t>) {
        *this = gloco->particles.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::universal_image_renderer_t::properties_t>) {
        *this = gloco->universal_image_renderer.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::gradient_t::properties_t>) {
        *this = gloco->gradient.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::shader_shape_t::properties_t>) {
        *this = gloco->shader_shape.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::rectangle3d_t::properties_t>) {
        *this = gloco->rectangle3d.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, line3d_t::properties_t>) {
        *this = gloco->line3d.push_back(properties);
      }
      else {
        fan::throw_error("failed to find correct shape", typeid(T).name());
      }
#if defined(debug_shape_t)
      fan::print("+", NRI);
#endif
    }
    shape_t(shape_t&& s) : shape_t(std::move(*dynamic_cast<shaper_t::ShapeID_t*>(&s))) {

    }
    shape_t(const shape_t& s) : shape_t(*dynamic_cast<const shaper_t::ShapeID_t*>(&s)) {
      //NRI = s.NRI;
    }
    shape_t& operator=(const shape_t& s) {
      if (iic() == false) {
        remove();
      }
      if (s.iic()) {
        return *this;
      }
      if (this != &s) {
        {
          auto sti = gloco->shaper.GetSTI(s);

          // alloc can be avoided inside switch
          uint8_t* KeyPack = new uint8_t[gloco->shaper.GetKeysSize(s)];
          gloco->shaper.WriteKeys(s, KeyPack);


          auto _vi = gloco->shaper.GetRenderData(s);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);

          auto _ri = gloco->shaper.GetData(s);
          auto rlen = gloco->shaper.GetDataSize(sti);
          uint8_t* ri = new uint8_t[rlen];
          std::memcpy(ri, _ri, rlen);

          *this = gloco->shaper.add(
            sti,
            KeyPack,
            gloco->shaper.GetKeysSize(s),
            vi,
            ri
          );
#if defined(debug_shape_t)
          fan::print("+", NRI);
#endif

          delete[] KeyPack;
          delete[] vi;
          delete[] ri;
        }
        //fan::print("i dont know what to do");
        //NRI = s.NRI;
      }
      return *this;
    }
    shape_t& operator=(shape_t&& s) {
      if (iic() == false) {
        remove();
      }
      if (s.iic()) {
        return *this;
      }

      if (this != &s) {
        NRI = s.NRI;
        s.sic();
      }
      return *this;
    }
    ~shape_t() {
      remove();
    }

    void remove() {
      if (iic()) {
        return;
      }
#if defined(debug_shape_t)
      fan::print("-", NRI);
#endif
      if (get_shape_type() == loco_t::shape_type_t::vfi) {
        gloco->vfi.erase(*this);
      }
      else {
        gloco->shaper.remove(*this);
      }
      sic();
    }

    void erase() {
      remove();
    }

    // many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t
    uint16_t get_shape_type() {
      return gloco->shaper.GetSTI(*this);
    }

    template <typename T>
    void set_position(const fan::vec2_wrap_t<T>& position) {
      gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_position2(this, position);
    }

    void set_position(const fan::vec3& position);

    fan::vec3 get_position();

    void set_size(const fan::vec2& size);
    void set_size3(const fan::vec3& size);

    fan::vec2 get_size();
    fan::vec3 get_size3();

    void set_rotation_point(const fan::vec2& rotation_point);

    fan::vec2 get_rotation_point();

    void set_color(const fan::color& color);

    fan::color get_color();

    void set_angle(const fan::vec3& angle);

    fan::vec3 get_angle();

    fan::vec2 get_tc_position();
    void set_tc_position(const fan::vec2& tc_position);

    fan::vec2 get_tc_size();
    void set_tc_size(const fan::vec2& tc_size);

    bool load_tp(loco_t::texturepack_t::ti_t* ti);
    loco_t::texturepack_t::ti_t get_tp();
    bool set_tp(loco_t::texturepack_t::ti_t* ti);

    fan::vec2 get_grid_size();
    void set_grid_size(const fan::vec2& grid_size);

    loco_t::camera_t get_camera();
    void set_camera(loco_t::camera_t camera);
    loco_t::viewport_t get_viewport();
    void set_viewport(loco_t::viewport_t viewport);

    loco_t::image_t get_image();
    void set_image(loco_t::image_t image);

    f32_t get_parallax_factor();
    void set_parallax_factor(f32_t parallax_factor);

    fan::vec3 get_rotation_vector();

    uint32_t get_flags();
    void set_flags(uint32_t flag);

    f32_t get_radius();
    fan::vec3 get_src();
    fan::vec3 get_dst();
    f32_t get_outline_size();
    fan::color get_outline_color();

    void reload(uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR);
    void reload(uint8_t format, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR);

    void set_line(const fan::vec2& src, const fan::vec2& dst);

  private:
  };


  struct light_t {

    shaper_t::KeyTypeIndex_t shape_type = shape_type_t::light;
    static constexpr int kpi = kp::light;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t parallax_factor;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 rotation_vector;
      uint32_t flags = 0;
      fan::vec3 angle;
    };;
    struct ri_t {

    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_parallax_factor"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{{2, "in_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{4, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{5, "in_rotation_vector"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector))},
      shape_gl_init_t{{6, "in_flags"}, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{7, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

    struct properties_t {
      using type_t = light_t;

      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      uint32_t flags = 0;
      fan::vec3 angle = 0;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);
  }light;

  struct line_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::line;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::color color;
      fan::vec3 src;
      fan::vec3 dst;
    };
    struct ri_t {

    };

#pragma pack(pop)

  inline static std::vector<shape_gl_init_t> locations = {
    shape_gl_init_t{{0, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
    shape_gl_init_t{{1, "in_src"}, 3, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
    shape_gl_init_t{{2, "in_dst"}, 3, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
  };

    struct properties_t {
      using type_t = line_t;

      fan::color color = fan::colors::white;
      fan::vec3 src;
      fan::vec3 dst;

      bool blending = false;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);

  }line;

  struct rectangle_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::rectangle;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::color outline_color;
      fan::vec3 angle;
    };
    struct ri_t {
      
    };

#pragma pack(pop)

    inline static  std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{3, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{4, "in_outline_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color))},
      shape_gl_init_t{{5, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

    struct properties_t {
      using type_t = rectangle_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::color color = fan::colors::white;
      fan::color outline_color = color;
      bool blending = false;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);

  }rectangle;

  //----------------------------------------------------------


  struct sprite_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::sprite;
    static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t parallax_factor;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 angle;
      uint32_t flags;
      fan::vec2 tc_position;
      fan::vec2 tc_size;
      f32_t seed;
    };
    struct ri_t {
      // main image + light buffer + 30
      std::array<loco_t::image_t, 30> images;
    };

#pragma pack(pop)

  inline static std::vector<shape_gl_init_t> locations = {
    shape_gl_init_t{{0, "in_positio"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
    shape_gl_init_t{{1, "in_parallax_factor"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
    shape_gl_init_t{{2, "in_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
    shape_gl_init_t{{3, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
    shape_gl_init_t{{4, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
    shape_gl_init_t{{5, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
    shape_gl_init_t{{6, "in_flags"}, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
    shape_gl_init_t{{7, "in_tc_position"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
    shape_gl_init_t{{8, "in_tc_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
    shape_gl_init_t{{9, "in_seed"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
  };

    struct properties_t {
      using type_t = sprite_t;

      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      uint32_t flags = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      f32_t seed = 0;

      bool load_tp(loco_t::texturepack_t::ti_t* ti) {
        auto& im = *ti->image;
        image = im;
        auto& img = gloco->image_get_data(im);
        tc_position = ti->position / img.size;
        tc_size = ti->size / img.size;
        return 0;
      }

      bool blending = false;

      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }sprite;

  struct unlit_sprite_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::unlit_sprite;
    static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t parallax_factor;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 angle;
      uint32_t flags;
      fan::vec2 tc_position;
      fan::vec2 tc_size;
      f32_t seed = 0;
    };
    struct ri_t {
      // main image + light buffer + 30
      std::array<loco_t::image_t, 30> images;
    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_parallax_factor"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{{2, "in_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{4, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{5, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
      shape_gl_init_t{{6, "in_flags"}, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{7, "in_tc_position"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shape_gl_init_t{{8, "in_tc_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
      shape_gl_init_t{{9, "in_seed"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
    };

    struct properties_t {
      using type_t = unlit_sprite_t;

      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      int flags = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      f32_t seed = 0;

      bool blending = false;

      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      bool load_tp(loco_t::texturepack_t::ti_t* ti) {
        auto& im = *ti->image;
        image = im;
        auto& img = gloco->image_get_data(im);
        tc_position = ti->position / img.size;
        tc_size = ti->size / img.size;
        return 0;
      }
    };

    shape_t push_back(const properties_t& properties);

  }unlit_sprite;

  struct text_t {

    struct vi_t {

    };

    struct ri_t {

    };

    struct properties_t {
      using type_t = text_t;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      fan::vec3 position;
      f32_t outline_size = 1;
      fan::vec2 size;
      fan::vec2 tc_position;
      fan::color color = fan::colors::white;
      fan::color outline_color;
      fan::vec2 tc_size;
      fan::vec3 angle = 0;

      fan::string text;
    };

    shape_t push_back(const properties_t& properties);
  }text;

  struct circle_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::circle;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t radius;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 rotation_vector;
      fan::vec3 angle;
      uint32_t flags;
    };
    struct ri_t {

    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
      shape_gl_init_t{{1, "in_radius"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
      shape_gl_init_t{{2, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
      shape_gl_init_t{{3, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
      shape_gl_init_t{{4, "in_rotation_vector"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector)) },
      shape_gl_init_t{{5, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
      shape_gl_init_t{{6, "in_flags"}, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))}
    };

    struct properties_t {
      using type_t = circle_t;

      fan::vec3 position = 0;
      f32_t radius = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      fan::vec3 angle = 0;
      uint32_t flags = 0;

      bool blending = false;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    loco_t::shape_t push_back(const circle_t::properties_t& properties);

  }circle;

  struct capsule_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::capsule;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      fan::vec2 center0;
      fan::vec2 center1;
      f32_t radius;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 rotation_vector;
      fan::vec3 angle;
      uint32_t flags;
      fan::color outline_color;
    };
    struct ri_t {

    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
      shape_gl_init_t{{1, "in_center0"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, center0)) },
      shape_gl_init_t{{2, "in_center1"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, center1)) },
      shape_gl_init_t{{3, "in_radius"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
      shape_gl_init_t{{4, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
      shape_gl_init_t{{5, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
      shape_gl_init_t{{6, "in_rotation_vector"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector)) },
      shape_gl_init_t{{7, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
      shape_gl_init_t{{8, "in_flags"}, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{9, "in_outline_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color)) },
    };

    struct properties_t {
      using type_t = capsule_t;

      fan::vec3 position = 0;
      fan::vec2 center0 = 0;
      fan::vec2 center1 = {0, 1.f};
      f32_t radius = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::color outline_color = color;
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      fan::vec3 angle = 0;
      uint32_t flags = 0;

      bool blending = true;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };
    loco_t::shape_t push_back(const capsule_t::properties_t& properties);
  }capsule;

  struct grid_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::grid;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      fan::vec2 size;
      fan::vec2 grid_size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 angle;
    };
    struct ri_t {
      
    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, size)},
      shape_gl_init_t{{2, "in_grid_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, grid_size)},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, rotation_point)},
      shape_gl_init_t{{4, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, color)},
      shape_gl_init_t{{5, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, angle)},
    };

    struct properties_t {
      using type_t = grid_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 grid_size;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = 0;

      bool blending = false;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);
  }grid;


  struct particles_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::particles;
    static constexpr int kpi = kp::texture;

    inline static std::vector<shape_gl_init_t> locations = {};
     
#pragma pack(push, 1)

    struct vi_t {
      
    };

    struct shapes_e {
      enum {
        circle,
        rectangle
      };
    };

    struct ri_t {

      fan::vec3 position;
      fan::vec2 size;
      fan::color color;

      uint64_t begin_time;
      uint64_t alive_time;
      uint64_t respawn_time;
      uint32_t count;
      fan::vec2 position_velocity;
      fan::vec3 angle_velocity;
      f32_t begin_angle;
      f32_t end_angle;

      fan::vec3 angle;

      fan::vec2 gap_size;
      fan::vec2 max_spread_size;
      fan::vec2 size_velocity;

      uint32_t shape;

      bool blending;
    };
#pragma pack(pop)

    struct properties_t {
      using type_t = particles_t;

      fan::vec3 position = 0;
      fan::vec2 size = 100;
      fan::color color = fan::colors::red;

      uint64_t begin_time;
      uint64_t alive_time = (uint64_t)1e+9;
      uint64_t respawn_time = 0;
      uint32_t count = 10;
      fan::vec2 position_velocity = 130;
      fan::vec3 angle_velocity = fan::vec3(0, 0, 0);
      f32_t begin_angle = 0;
      f32_t end_angle = fan::math::pi * 2;

      fan::vec3 angle = 0;

      fan::vec2 gap_size = 1;
      fan::vec2 max_spread_size = 100;
      fan::vec2 size_velocity = 1;

      uint32_t shape = shapes_e::circle;

      bool blending = true;

      loco_t::image_t image = gloco->default_texture;
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }particles;

  struct universal_image_renderer_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::universal_image_renderer;
    static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
    };
    struct ri_t {
      loco_t::image_t images_rest[3]; // 3 + 1 (pk)
      uint8_t format = fan::pixel_format::undefined;
    };

#pragma pack(pop)

  inline static std::vector<shape_gl_init_t> locations = {
    shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
    shape_gl_init_t{{1, "in_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
    shape_gl_init_t{{2, "in_tc_position"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
    shape_gl_init_t{{3, "in_tc_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))}
  };

    struct properties_t {
      using type_t = universal_image_renderer_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;

      bool blending = false;

      loco_t::image_t images[4] = {
        gloco->default_texture,
        gloco->default_texture,
        gloco->default_texture,
        gloco->default_texture
      };
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }universal_image_renderer;

  struct gradient_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::gradient;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      fan::vec2 size;
      fan::vec2 rotation_point;
      // top left, top right
      // bottom left, bottom right
      fan::color color[4];
      fan::vec3 angle;
    };
    struct ri_t {

    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{3, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 0)},
      shape_gl_init_t{{4, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 1)},
      shape_gl_init_t{{5, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 2)},
      shape_gl_init_t{{6, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 3)},
      shape_gl_init_t{{7, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

    struct properties_t {
      using type_t = gradient_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::color color[4] = {
        fan::random::color(),
        fan::random::color(),
        fan::random::color(),
        fan::random::color()
      };
      bool blending = false;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);

  }gradient;

  struct shader_shape_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::shader_shape;
    static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t parallax_factor;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 angle;
      uint32_t flags;
      fan::vec2 tc_position;
      fan::vec2 tc_size;
      f32_t seed;
    };
    struct ri_t {
      // main image + light buffer + 30
      std::array<loco_t::image_t, 30> images;
    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_parallax_factor"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{{2, "in_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{4, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{5, "in_angle"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
      shape_gl_init_t{{6, "in_flags"}, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{7, "in_tc_position"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shape_gl_init_t{{8, "in_tc_size"}, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
      shape_gl_init_t{{9, "in_seed"}, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
    };

    struct properties_t {
      using type_t = shader_shape_t;

      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      uint32_t flags = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      f32_t seed = 0;
      loco_t::shader_t shader;
      bool blending = false;

      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }shader_shape;

  struct rectangle3d_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::rectangle3d;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      fan::vec3 size;
      fan::color color;
      fan::vec3 angle;
    };
    struct ri_t {

    };

#pragma pack(pop)

    inline static  std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t,  position)},
      shape_gl_init_t{{1, "in_size"}, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))}
    };

    struct properties_t {
      using type_t = rectangle_t;

      fan::vec3 position = 0;
      fan::vec3 size = 0;
      fan::color color = fan::colors::white;
      bool blending = false;
      fan::vec3 angle = 0;

      loco_t::camera_t camera = gloco->perspective_camera.camera;
      loco_t::viewport_t viewport = gloco->perspective_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);

  }rectangle3d;

  struct line3d_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::line3d;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::color color;
      fan::vec3 src;
      fan::vec3 dst;
    };
    struct ri_t {

    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_color"}, 4, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
      shape_gl_init_t{{1, "in_src"}, 3, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
      shape_gl_init_t{{2, "in_dst"}, 3, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
    };

    struct properties_t {
      using type_t = line_t;

      fan::color color = fan::colors::white;
      fan::vec3 src;
      fan::vec3 dst;

      bool blending = false;

      loco_t::camera_t camera = gloco->perspective_camera.camera;
      loco_t::viewport_t viewport = gloco->perspective_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);

  }line3d;

  //-------------------------------------shapes-------------------------------------

  template <typename T>
  inline void shape_open(T* shape, const fan::string& vertex, const fan::string& fragment, loco_t::shaper_t::ShapeRenderDataSize_t instance_count = 1) {
    auto& context = gloco->get_context();

    loco_t::shader_t shader = context.shader_create();

    context.shader_set_vertex(shader,
      context.read_shader(vertex)
    );

    context.shader_set_fragment(shader,
      context.read_shader(fragment)
    );

    context.shader_compile(shader);

    gloco->shaper.AddShapeType(
      shape->shape_type,
      {
        .MaxElementPerBlock = (loco_t::shaper_t::MaxElementPerBlock_t)MaxElementPerBlock,
        .RenderDataSize = (decltype(loco_t::shaper_t::BlockProperties_t::RenderDataSize))(sizeof(typename T::vi_t) * instance_count),
        .DataSize = sizeof(typename T::ri_t),
        .locations = T::locations,
        .shader = shader
      }
    );

    loco_t::functions_t functions = loco_t::get_functions<typename T::vi_t>();
    gloco->shape_functions.push_back(functions);
  }


#if defined(loco_sprite)
  loco_t::shader_t create_sprite_shader(const fan::string& fragment);
#endif


#if defined(loco_vfi)
  #include <fan/graphics/gui/vfi.h>
#endif
  vfi_t vfi;

//#if defined(loco_texture_pack)
//#endif

#if defined(loco_post_process)
  #include <fan/graphics/opengl/2D/effects/blur.h>
    blur_t blur[1];
  #include <fan/graphics/opengl/2D/effects/bloom.h>
  bloom_t bloom;
#endif

  fan::color clear_color = { 0.10f, 0.10f, 0.131f, 1.f };

#if defined(loco_framebuffer)
#if defined(loco_opengl)

  fan::opengl::core::framebuffer_t m_framebuffer;
  fan::opengl::core::renderbuffer_t m_rbo;
  loco_t::image_t color_buffers[4];
  loco_t::shader_t m_fbo_final_shader;

#endif
#endif

  struct lighting_t {
    static constexpr const char* ambient_name = "lighting_ambient";
    fan::vec3 ambient = fan::vec3(1, 1, 1);
  }lighting;

  //gui
#if defined(loco_imgui)
  fan::console_t console;
  bool render_console = false;
  bool toggle_fps = false;

  ImFont* fonts[6];
#endif
  //gui

  std::vector<uint8_t> create_noise_image_data(const fan::vec2& image_size, int seed = fan::random::value_i64(0, ((uint32_t)-1) / 2));

  loco_t::image_t create_noise_image(const fan::vec2& image_size);
  loco_t::image_t create_noise_image(const fan::vec2& image_size, const std::vector<uint8_t>& noise_data);
  static fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position, const fan::vec2i& window_size) {
    return fan::vec2((2.0f * mouse_position.x) / window_size.x - 1.0f, 1.0f - (2.0f * mouse_position.y) / window_size.y);
  }
  fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const {
    return convert_mouse_to_ndc(mouse_position, gloco->window.get_size());
  }
  fan::vec2 convert_mouse_to_ndc() const {
    return convert_mouse_to_ndc(gloco->get_mouse_position(), gloco->window.get_size());
  }
  static fan::ray3_t convert_mouse_to_ray(const fan::vec2i& mouse_position, const fan::vec2& screen_size, const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view);
  fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view);
  fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view);
  static bool is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size);



#if defined(loco_cuda)

  struct cuda_textures_t {

    cuda_textures_t() {
      inited = false;
    }
    ~cuda_textures_t() {
    }
    void close(loco_t* loco, loco_t::shape_t& cid) {
      loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)gloco->shaper.GetData(cid);
      uint8_t image_amount = fan::pixel_format::get_texture_amount(ri.format);
      for (uint32_t i = 0; i < image_amount; ++i) {
        wresources[i].close();
        gloco->image_unload(ri.images_rest[i]);
      }
    }

    void resize(loco_t* loco, loco_t::shape_t& id, uint8_t format, fan::vec2ui size, uint32_t filter = loco_t::image_filter::linear) {
      id.reload(format, size, filter);
      auto& ri = *(universal_image_renderer_t::ri_t*)gloco->shaper.GetData(id);
      auto vi_image = id.get_image();
      uint8_t image_amount = fan::pixel_format::get_texture_amount(format);
      if (inited == false) {
        // purge cid's images here
        // update cids images
        for (uint32_t i = 0; i < image_amount; ++i) {
          // a bit bad from fan side
          if (i == 0) {
            wresources[i].open(gloco->image_get(vi_image));
          }
          else {
            wresources[i].open(gloco->image_get(ri.images_rest[i - 1]));
          }
        }
        inited = true;
      }
      else {

        if (gloco->image_get_data(vi_image).size == size) {
          return;
        }

        // update cids images
        for (uint32_t i = 0; i < fan::pixel_format::get_texture_amount(ri.format); ++i) {
          wresources[i].close();
        }

        id.reload(format, size, filter);

        for (uint32_t i = 0; i < image_amount; ++i) {
          if (i == 0) {
            wresources[i].open(gloco->image_get(vi_image));
          }
          else {
            wresources[i].open(gloco->image_get(ri.images_rest[i - 1]));
          }
        }
      }
    }

    cudaArray_t& get_array(uint32_t index) {
      return wresources[index].cuda_array;
    }

    struct graphics_resource_t {
      void open(int texture_id) {
        fan::cuda::check_error(cudaGraphicsGLRegisterImage(&resource, texture_id, fan::opengl::GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
        map();
      }
      void close() {
        unmap();
        fan::cuda::check_error(cudaGraphicsUnregisterResource(resource));
        resource = nullptr;
      }
      void map() {
        fan::cuda::check_error(cudaGraphicsMapResources(1, &resource, 0));
        fan::cuda::check_error(cudaGraphicsSubResourceGetMappedArray(&cuda_array, resource, 0, 0));
        fan::print("+", resource);
      }
      void unmap() {
        fan::print("-", resource);
        fan::cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
        //fan::cuda::check_error(cudaGraphicsResourceSetMapFlags(resource, 0));
      }
      //void reload(int texture_id) {
      //  close();
      //  open(texture_id);
      //}
      cudaGraphicsResource_t resource = nullptr;
      cudaArray_t cuda_array = nullptr;
    };

    bool inited = false;
    graphics_resource_t wresources[4];
  };

#endif
};

// user friendly functions
/***************************************/
namespace fan {
  namespace graphics {

    using vfi_t = loco_t::vfi_t;

    using engine_t = loco_t;
    using image_t = loco_t::image_t;

#if defined(loco_imgui)
    using imgui_element_t = loco_t::imgui_element_t;

    void text(const std::string& text, const fan::vec2& position = 0, const fan::color& color = fan::colors::white);
    void text_bottom_right(const std::string& text, const fan::color& color = fan::colors::white, const fan::vec2& offset = 0);
#endif

    using camera_impl_t = loco_t::camera_impl_t;
    using camera_t = camera_impl_t;

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
            .rotation_vector = p.rotation_vector,
            .flags = p.flags,
            .angle = p.angle
          ));
      }
    };

    #if defined(loco_line)

      struct line_properties_t {
        camera_impl_t* camera = &gloco->orthographic_camera;
        fan::vec3 src = fan::vec3(0, 0, 0);
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
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
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

    // a bit bad because if sprite_t::properties or vi change need to update here
    struct sprite_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;
      f32_t parallax_factor = 0;
      bool blending = false;
      uint32_t flags = 0;
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
      fan::vec3 position = fan::vec3(0, 0, 0);
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
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t radius = 32.f;
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
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
            .color = p.color,
            .blending = p.blending,
            .flags = p.flags
          ));
      }
    };
#endif

    struct capsule_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
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

    struct grid_properties_t {
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
            .position = p.position,
            .size = p.size,
            .grid_size = p.grid_size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .angle = p.angle
          ));
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


#if defined(loco_vfi)

    // for line
    fan::line3 get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index);

    // REQUIRES to be allocated by new since lambda captures this
    // also container that it's stored in, must not change pointers
    template <typename T>
    struct vfi_root_custom_t {
      void create_highlight() {
        fan::vec3 op = children[0].get_position();
        fan::vec2 os = children[0].get_size();
        loco_t::camera_t c = children[0].get_camera();
        loco_t::viewport_t v = children[0].get_viewport();
        fan::graphics::camera_t cam;
        cam.camera = c;
        cam.viewport = v;
        for (std::size_t j = 0; j < highlight.size(); ++j) {
          for (std::size_t i = 0; i < highlight[0].size(); ++i) {
            fan::line3 line = get_highlight_positions(op, os, i);
            highlight[j][i] = fan::graphics::line_t{ {
              .camera = &cam,
              .src = line[0],
              .dst = line[1],
              .color = fan::color(1, 0.5, 0, 1)
            } };
          }
        }
      }
      void disable_highlight() {
        fan::vec3 op = children[0].get_position();
        fan::vec2 os = children[0].get_size();
        for (std::size_t j = 0; j < highlight.size(); ++j) {
          for (std::size_t i = 0; i < highlight[j].size(); ++i) {
            fan::line3 line = get_highlight_positions(op, os, i);
            if (highlight[j][i].iic() == false) {
              highlight[j][i].set_line(0, 0);
            }
          }
        }
      }

      void set_root(const loco_t::vfi_t::properties_t& p) {
        fan::graphics::vfi_t::properties_t in = p;
        in.shape_type = loco_t::vfi_t::shape_t::rectangle;
        in.shape.rectangle->viewport = p.shape.rectangle->viewport;
        in.shape.rectangle->camera = p.shape.rectangle->camera;
        in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) -> int {
          if (d.key == fan::key_c &&
            (d.keyboard_state == fan::keyboard_state::press ||
              d.keyboard_state == fan::keyboard_state::repeat)) {
            this->resize = true;
            return user_cb(d);
          }
          this->resize = false;
          return 0;
          };
        in.mouse_button_cb = [this, user_cb = p.mouse_button_cb](const auto& d) -> int {
          if (g_ignore_mouse) {
            return 0;
          }

          if (d.button != fan::mouse_left) {
            return 0;
          }
          if (d.button_state != fan::mouse_state::press) {
            this->move = false;
            moving_object = false;
            d.flag->ignore_move_focus_check = false;
              if (previous_click_position == d.position) {
                for (auto it = selected_objects.begin(); it != selected_objects.end(); ) {
                    (*it)->disable_highlight();
                    if (*it != this) {
                      it = selected_objects.erase(it);
                    } else {
                      ++it;
                    }
                  }
              }
            return 0;
          }
          if (d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
            return 0;
          }

          if (previous_focus && previous_focus != this) {
            for (std::size_t i = 0; i < previous_focus->highlight[0].size(); ++i) {
              if (previous_focus->highlight[0][i].iic() == false) {
                previous_focus->highlight[0][i].set_line(0, 0);
              }
            }
          }
          //selected_objects.clear();
          if (std::find(selected_objects.begin(), selected_objects.end(), this) == selected_objects.end()) {
            selected_objects.push_back(this);
          }
          //selected_objects.push_back(this);
          create_highlight();
          previous_focus = this;

          if (move_and_resize_auto) {
            previous_click_position = d.position;
            d.flag->ignore_move_focus_check = true;
            this->move = true;
            moving_object = true;
            this->click_offset = get_position() - d.position;
            
            gloco->vfi.set_focus_keyboard(d.vfi->focus.mouse);
          }
          return user_cb(d);
          };
        in.mouse_move_cb = [this, user_cb = p.mouse_move_cb](const auto& d) -> int {
          if (g_ignore_mouse) {
            return 0;
          }

          if (move_and_resize_auto) {
            if (this->resize && this->move) {
              fan::vec2 new_size = (d.position - get_position());
              static constexpr fan::vec2 min_size(10, 10);
              new_size.clamp(min_size);
              this->set_size(new_size.x);
              fan::vec3 op = children[0].get_position();
              fan::vec2 os = children[0].get_size();
              for (std::size_t j = 0; j < highlight.size(); ++j) {
                for (std::size_t i = 0; i < highlight[j].size(); ++i) {
                  fan::line3 line = get_highlight_positions(op, os, i);
                  if (highlight[j][i].iic() == false) {
                    highlight[j][i].set_line(line[0], line[1]);
                  }
                }
              }
              if (previous_focus && previous_focus != this) {
                for (std::size_t i = 0; i < previous_focus->highlight[0].size(); ++i) {
                  if (previous_focus->highlight[0][i].iic() == false) {
                    previous_focus->highlight[0][i].set_line(0, 0);
                  }
                }
                previous_focus = this;
              }
              return user_cb(d);
            }
            else if (this->move) {
              fan::vec3 p = get_position();
              p = fan::vec3(d.position + click_offset, p.z);
              p.x = std::round(p.x / 32.0f) * 32.0f;
              p.y = std::round(p.y / 32.0f) * 32.0f;
              this->set_position(p);
              return user_cb(d);
            }
          }
          else {
            return user_cb(d);
          }
          return 0;
          };
        vfi_root = in;
      }
      void push_child(const loco_t::shape_t& shape) {
        children.push_back({ shape });
      }
      fan::vec3 get_position() {
        return vfi_root.get_position();
      }

      static void update_highlight_position(vfi_root_custom_t<T>* instance) {
        fan::vec3 op = instance->children[0].get_position();
        fan::vec2 os = instance->children[0].get_size();
        for (std::size_t j = 0; j < instance->highlight.size(); ++j) {
          for (std::size_t i = 0; i < instance->highlight[j].size(); ++i) {
            fan::line3 line = get_highlight_positions(op, os, i);
            if (instance->highlight[j][i].iic() == false) {
              instance->highlight[j][i].set_line(line[0], line[1]);
            }
          }
        }
      }

      void set_position(const fan::vec3& position) {
        fan::vec2 root_pos = vfi_root.get_position();
        fan::vec2 offset = position - root_pos;
        vfi_root.set_position(fan::vec3(root_pos + offset, position.z));

        for (auto& child : children) {
          child.set_position(fan::vec3(fan::vec2(child.get_position()) + offset, position.z));
        }
        update_highlight_position(this);

        if (previous_focus && previous_focus != this) {
          for (std::size_t i = 0; i < previous_focus->highlight[0].size(); ++i) {
            if (previous_focus->highlight[0][i].iic() == false) {
              previous_focus->highlight[0][i].set_line(0, 0);
            }
          }
          previous_focus = this;
        }

        for (auto* i : selected_objects) {
          if (i == this) {
            continue;
          }
          fan::vec2 root_pos = i->vfi_root.get_position();
          i->vfi_root.set_position(fan::vec3(root_pos + offset, position.z));

          for (auto& child : i->children) {
            child.set_position(fan::vec3(fan::vec2(child.get_position()) + offset, position.z));
          }
          update_highlight_position(i);
        }
      }
      fan::vec2 get_size() {
        return vfi_root.get_size();
      }
      void set_size(const fan::vec2& size) {
        fan::vec2 root_pos = vfi_root.get_size();
        fan::vec2 offset = size - root_pos;
        vfi_root.set_size(root_pos + offset);
        for (auto& child : children) {
          child.set_size(fan::vec2(child.get_size()) + offset);
        }
      }

      fan::color get_color() {
        if (children.size()) {
          return children[0].get_color();
        }
        return fan::color(1);
      }
      void set_color(const fan::color& color) {
        for (auto& child : children) {
          child.set_color(color);
        }
      }

      inline static bool g_ignore_mouse = false;
      inline static bool moving_object = false;

      fan::vec2 click_offset = 0;
      fan::vec2 previous_click_position;
      bool move = false;
      bool resize = false;

      bool move_and_resize_auto = true;

      loco_t::shape_t vfi_root;
      struct child_data_t : loco_t::shape_t, T {

      };
      std::vector<child_data_t> children;

      inline static std::vector<vfi_root_custom_t<T>*> selected_objects;

      inline static vfi_root_custom_t<T>* previous_focus = nullptr;

      // 4 lines for square
      std::vector<std::array<loco_t::shape_t, 4>> highlight{ 1 };
    };

    using vfi_root_t = vfi_root_custom_t<__empty_struct>;


    template <typename T>
    struct vfi_multiroot_custom_t {
      void push_root(const loco_t::vfi_t::properties_t& p) {
        loco_t::vfi_t::properties_t in = p;
        in.shape_type = loco_t::vfi_t::shape_t::rectangle;
        in.shape.rectangle->viewport = p.shape.rectangle->viewport;
        in.shape.rectangle->camera = p.shape.rectangle->camera;
        in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) -> int {
          if (d.key == fan::key_c &&
            (d.keyboard_state == fan::keyboard_state::press ||
              d.keyboard_state == fan::keyboard_state::repeat)) {
            this->resize = true;
            return 0;
          }
          this->resize = false;
          return user_cb(d);
          };
        in.mouse_button_cb = [this, root_reference = vfi_root.empty() ? 0 : vfi_root.size() - 1, user_cb = p.mouse_button_cb](const auto& d) -> int {
          if (g_ignore_mouse) {
            return 0;
          }

          if (d.button != fan::mouse_left) {
            return user_cb(d);
          }

          if (d.button_state == fan::mouse_state::press && move_and_resize_auto) {
            this->move = true;
            gloco->vfi.focus.method.mouse.flags.ignore_move_focus_check = true;
          }
          else if (d.button_state == fan::mouse_state::release && move_and_resize_auto) {
            this->move = false;
            gloco->vfi.focus.method.mouse.flags.ignore_move_focus_check = false;
          }

          if (d.button_state == fan::mouse_state::release) {
            for (auto& root : vfi_root) {
              auto position = root->get_position();
              auto p = fan::vec3(fan::vec2(position), position.z);
              if (grid_size.x > 0) {
                p.x = floor(p.x / grid_size.x) * grid_size.x + grid_size.x / 2;
              }
              if (grid_size.y > 0) {
                p.y = floor(p.y / grid_size.y) * grid_size.y + grid_size.y / 2;
              }
              root->set_position(p);
            }
            for (auto& child : children) {
              auto position = child.get_position();
              auto p = fan::vec3(fan::vec2(position), position.z);
              if (grid_size.x > 0) {
                p.x = floor(p.x / grid_size.x) * grid_size.x + grid_size.x / 2;
              }
              if (grid_size.y > 0) {
                p.y = floor(p.y / grid_size.y) * grid_size.y + grid_size.y / 2;
              }
              child.set_position(p);
            }
          }
          if (d.button_state != fan::mouse_state::press) {
            return user_cb(d);
          }
          if (d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
            return user_cb(d);
          }

          if (move_and_resize_auto) {
            this->click_offset = get_position(root_reference) - d.position;
            gloco->vfi.set_focus_keyboard(d.vfi->focus.mouse);
          }
          return user_cb(d);
          };
        in.mouse_move_cb = [this, root_reference = vfi_root.empty() ? 0 : vfi_root.size() - 1, user_cb = p.mouse_move_cb](const auto& d) -> int {
          if (g_ignore_mouse) {
            return 0;
          }

          if (move_and_resize_auto) {
            if (this->resize && this->move) {
              return user_cb(d);
            }
            else if (this->move) {
              fan::vec3 p = get_position(root_reference);
              p = fan::vec3(d.position + click_offset, p.z);
              this->set_position(root_reference, p);
              return user_cb(d);
            }
          }
          else {
            return user_cb(d);
          }
          return 0;
          };
        vfi_root.push_back(std::make_unique<loco_t::shape_t>(in));
      }
      void push_child(const loco_t::shape_t& shape) {
        children.push_back({ shape });
      }
      fan::vec3 get_position(uint32_t index) {
        return vfi_root[index]->get_position();
      }
      void set_position(uint32_t root_reference, const fan::vec3& position) {
        fan::vec2 root_pos = vfi_root[root_reference]->get_position();
        fan::vec2 offset = position - root_pos;
        for (auto& root : vfi_root) {
          auto p = fan::vec3(fan::vec2(root->get_position()) + offset, position.z);
          root->set_position(fan::vec3(p.x, p.y, p.z));
        }
        for (auto& child : children) {
          auto p = fan::vec3(fan::vec2(child.get_position()) + offset, position.z);
          child.set_position(p);
        }
      }

      inline static bool g_ignore_mouse = false;

      fan::vec2 click_offset = 0;
      bool move = false;
      bool resize = false;
      fan::vec2 grid_size = 0;

      bool move_and_resize_auto = true;

      std::vector<std::unique_ptr<loco_t::shape_t>> vfi_root;
      struct child_data_t : loco_t::shape_t, T {

      };
      std::vector<child_data_t> children;
    };

    using vfi_multiroot_t = vfi_multiroot_custom_t<__empty_struct>;

  #endif
//#endif

    fan::vec2 get_mouse_position(const fan::graphics::camera_t& camera);
  }
}

// Imgui extensions
#if defined(loco_imgui)
namespace ImGui {
  IMGUI_API void Image(loco_t::image_t img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), const ImVec4& tint_col = ImVec4(1, 1, 1, 1), const ImVec4& border_col = ImVec4(0, 0, 0, 0));
  IMGUI_API bool ImageButton(const std::string& str_id, loco_t::image_t img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0, 0, 0, 0), const ImVec4& tint_col = ImVec4(1, 1, 1, 1));

  bool ToggleButton(const std::string& str, bool* v);
  bool ToggleImageButton(const std::string& char_id, loco_t::image_t image, const ImVec2& size, bool* toggle);
  
  void DrawTextBottomRight(const char* text, uint32_t reverse_yoffset = 0);


  template <std::size_t N>
  bool ToggleImageButton(const std::array<loco_t::image_t, N>& images, const ImVec2& size, int* selectedIndex)
  {
    f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y -  ImGui::GetStyle().FramePadding.y / 2;
    
    bool clicked = false;
    bool pushed = false;

    for (std::size_t i = 0; i < images.size(); ++i) {
      ImVec4 tintColor = ImVec4(0.2, 0.2, 0.2, 1.0);
      if (*selectedIndex == i) {
        tintColor = ImVec4(0.2, 0.2, 0.2, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, tintColor);
        pushed = true;
      }
      /*if (ImGui::IsItemHovered()) {
        tintColor = ImVec4(1, 1, 1, 1.0f);
      }*/
      ImGui::SetCursorPosY(y_pos);
      if (ImGui::ImageButton("##toggle_image_button" + std::to_string(i) + std::to_string((uint64_t)&clicked), images[i], size)) {
        *selectedIndex = i;
        clicked = true;
      }
      if (pushed) {
        ImGui::PopStyleColor();
        pushed = false;
      }

      ImGui::SameLine();
    }

    return clicked;
  }


  ImVec2 GetPositionBottomCorner(const char* text = "", uint32_t reverse_yoffset = 0);

}
// Imgui extensions

#include <fan/io/directory.h>

namespace fan {
  namespace graphics {
    struct imgui_content_browser_t {
      struct file_info_t {
        std::string filename;
        std::filesystem::path some_path; //?
        std::wstring item_path;
        bool is_directory;
        loco_t::image_t preview_image;
        //std::string 
      };

      std::vector<file_info_t> directory_cache;

      loco_t::image_t icon_arrow_left = gloco->image_load("images_content_browser/arrow_left.webp");
      loco_t::image_t icon_arrow_right = gloco->image_load("images_content_browser/arrow_right.webp");

      loco_t::image_t icon_file = gloco->image_load("images_content_browser/file.webp");
      loco_t::image_t icon_directory = gloco->image_load("images_content_browser/folder.webp");

      loco_t::image_t icon_files_list = gloco->image_load("images_content_browser/files_list.webp");
      loco_t::image_t icon_files_big_thumbnail = gloco->image_load("images_content_browser/files_big_thumbnail.webp");


      std::wstring asset_path = L"./";

      std::filesystem::path current_directory;
      enum viewmode_e {
        view_mode_list,
        view_mode_large_thumbnails,
      };
      viewmode_e current_view_mode = view_mode_list;
      float thumbnail_size = 128.0f;
      f32_t padding = 16.0f;
      std::string search_buffer;

      imgui_content_browser_t();
      void update_directory_cache();
      void render();
      void render_large_thumbnails_view();
      void render_list_view();
      void handle_item_interaction(const file_info_t& file_info);
      // [](const std::filesystem::path& path) {}
      void receive_drag_drop_target(auto receive_func) {
        ImGui::Dummy(ImGui::GetContentRegionAvail());

        if (ImGui::BeginDragDropTarget()) {
          if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CONTENT_BROWSER_ITEM")) {
            const wchar_t* path = (const wchar_t*)payload->Data;
            receive_func(std::filesystem::path(path));
            //fan::print(std::filesystem::path(path));
          }
          ImGui::EndDragDropTarget();
        }
      }
    };
  }
}
#endif

void init_imgui();

void shape_keypack_traverse(loco_t::shaper_t::KeyTraverse_t& KeyTraverse, fan::opengl::context_t& context);

#if defined(loco_json)
namespace fan {
  namespace graphics {
    bool shape_to_json(loco_t::shape_t& shape, fan::json* json);

    bool json_to_shape(const fan::json& in, loco_t::shape_t* shape);

    bool shape_serialize(loco_t::shape_t& shape, fan::json* out);
  }
}

namespace fan {

  namespace graphics {
    bool shape_to_bin(loco_t::shape_t& shape, std::string* str);

    bool bin_to_shape(const std::string& str, loco_t::shape_t* shape, uint64_t& offset);

    bool shape_serialize(loco_t::shape_t& shape, std::string* out);

    struct shape_deserialize_t {
      struct {
        // json::iterator doesnt support union
        // i dont want to use variant either so i accept few extra bytes
        json::const_iterator it;
        uint64_t offset = 0;
      }data;
      bool init = false;

      bool iterate(const fan::json& json, loco_t::shape_t* shape) {
        if (init == false) {
          data.it = json.cbegin();
          init = true; 
        }
        if (data.it == json.cend()) {
          return 0;
        }
        if (json.type() == fan::json::value_t::object) {
          json_to_shape(json, shape);
          return 0;
        }
        else {
          json_to_shape(*data.it, shape);
          ++data.it;
        }
        return 1;
      }

      bool iterate(const std::string& str, loco_t::shape_t* shape) {
        if (str.empty()) {
          return 0;
        }
        else if (data.offset >= str.size()) {
          return 0;
        }
        bin_to_shape(str, shape, data.offset);
        return 1;
      }
    };
  }
}

#endif

#if defined (loco_imgui)
void fan::printcl(auto&&... values) {
  ([&](const auto& value) {
    std::ostringstream oss;
    oss << value;
    gloco->console.print(oss.str() + " ", 0);
    }(values), ...);
  gloco->console.print("\n", 0);
}

void fan::printclh(int highlight, auto&&... values) {
  ([&](const auto& value) {
    std::ostringstream oss;
    oss << value;
    gloco->console.print(oss.str() + " ", highlight);
    }(values), ...);
  gloco->console.print("\n", highlight);
}
#endif
#include <fan/graphics/collider.h>
#if defined(loco_box2d)
  #include <fan/graphics/physics_shapes.hpp>
#endif