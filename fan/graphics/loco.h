#pragma once

#define loco_audio

#include <fan/types/types.h>
#if defined(loco_audio)
  #ifndef _INCLUDE_TOKEN
    #define _INCLUDE_TOKEN(p0, p1) <p0/p1>
  #endif
  #ifndef WITCH_INCLUDE_PATH
    #define WITCH_INCLUDE_PATH WITCH
  #endif
  #define WITCH_PRE_is_not_allowed
  #include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH,WITCH.h)
  #include <fan/audio/audio.h>
  
#endif

#include <fan/ev/ev.h>

#include <fan/graphics/file_dialog.h>

#include <fan/types/lazy_compiler_devs.h>

#if defined(fan_physics)
  #include <fan/physics/b2_integration.hpp>
#endif

#define loco_opengl
#define loco_framebuffer
#define loco_post_process
#define loco_vfi

#define loco_physics

//
#include <fan/window/window.h>
#include <fan/io/file.h>

#include <fan/graphics/types.h>

#if defined(fan_gui)
#include <fan/imgui/imgui.h>
#include <fan/imgui/imgui_impl_opengl3.h>
#if defined(loco_vulkan)
#include <fan/imgui/imgui_impl_vulkan.h>
#endif
#include <fan/imgui/imgui_impl_glfw.h>
#include <fan/imgui/imgui_neo_sequencer.h>
#include <fan/imgui/implot.h>
#endif

#include <fan/physics/collision/rectangle.h>

#include <fan/graphics/algorithm/FastNoiseLite.h>

#include <fan/graphics/opengl/core.h>
#if defined(loco_vulkan)
#include <fan/graphics/vulkan/core.h>
#endif

#undef camera_list
#undef shader_list
#undef image_list
#undef viewport_list

#if defined(fan_gui)
  #include <fan/graphics/console.h>
#endif

#if defined(fan_json)

#include <fan/io/json_impl.h>

struct loco_t;

// shaper
#include <variant>

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
  extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
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
inline global_loco_t gloco;

namespace fan {
  namespace graphics {

    struct engine_init_t {
      #define BLL_set_SafeNext 1
      #define BLL_set_AreWeInsideStruct 1
      #define BLL_set_prefix init_callback
      #include <fan/fan_bll_preset.h>
      #define BLL_set_Link 1
      #define BLL_set_type_node uint16_t
      #define BLL_set_NodeDataType fan::function_t<void(loco_t*)>
      #define BLL_set_CPP_CopyAtPointerChange 1
      #include <BLL/BLL.h>

      using init_callback_nr_t = init_callback_NodeReference_t;
    };

    // cbs called every time engine opens
    inline engine_init_t::init_callback_t engine_init_cbs;

    uint32_t get_draw_mode(uint8_t internal_draw_mode);

    namespace gui {
      bool render_blank_window(const std::string& name);
    }
    using context_shader_init_t = std::variant<
      fan::opengl::context_t::shader_t
#if defined(loco_vulkan)
      ,fan::vulkan::context_t::shader_t
#endif
    >;
    struct context_shader_t : context_shader_init_t {
      using context_shader_init_t::variant;
    };
    using context_image_init_t = std::variant<
      fan::opengl::context_t::image_t
  #if defined(loco_vulkan)
      ,fan::vulkan::context_t::image_t
#endif
    >;
    struct context_image_t : context_image_init_t {
      using context_image_init_t::variant;
    };
    struct context_t {
      context_t() {}
      ~context_t() {}
      union {
        fan::opengl::context_t gl;
  #if defined(loco_vulkan)
        fan::vulkan::context_t vk;
#endif
      };
    };
  }
}

//#include <fan/graphics/vulkan/ssbo.h>
struct loco_t {
  using renderer_t = fan::window_t::renderer_t;
  uint8_t get_renderer() {
    return window.renderer;
  }
  
  using shader_t = fan::graphics::shader_nr_t;
  using image_t = fan::graphics::image_nr_t;
  using camera_t = fan::graphics::camera_nr_t;
  using viewport_t = fan::graphics::viewport_nr_t;
  using image_load_properties_t = fan::graphics::image_load_properties_t;

  using image_sampler_address_mode = fan::graphics::image_sampler_address_mode;

  fan::graphics::shader_nr_t shader_create();
  fan::graphics::context_shader_t shader_get(fan::graphics::shader_nr_t nr);
  void shader_erase(fan::graphics::shader_nr_t nr);
  void shader_use(fan::graphics::shader_nr_t nr);
  void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code);
  void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code);
  bool shader_compile(fan::graphics::shader_nr_t nr);
  template <typename T>
  void shader_set_value(fan::graphics::shader_nr_t nr, const std::string& name, const T& val) {
    if (window.renderer == renderer_t::opengl) {
      context.gl.shader_set_value(nr, name, val);
    }
    else if (window.renderer == renderer_t::vulkan) {
      fan::throw_error("todo");
    }
  }
  void shader_set_camera(shader_t nr, camera_t camera_nr) {
    if (window.renderer == renderer_t::opengl) {
      context.gl.shader_set_camera(nr, camera_nr);
    }
#if defined(loco_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      fan::throw_error("todo");
    }
#endif
  }

  fan::graphics::camera_list_t camera_list;
  fan::graphics::shader_list_t shader_list;
  fan::graphics::image_list_t image_list;
  fan::graphics::viewport_list_t viewport_list;

  std::unique_ptr<uint8_t[]> image_get_pixel_data(fan::graphics::image_nr_t nr, GLenum format, fan::vec2 uvp = 0, fan::vec2 uvs = 1) {
    fan::throw_error("");
    return {};
  }

  fan::graphics::image_nr_t image_create();
  fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr);
  
  uint64_t image_get_handle(fan::graphics::image_nr_t nr);
  fan::graphics::image_data_t& image_get_data(fan::graphics::image_nr_t nr);
  void image_erase(fan::graphics::image_nr_t nr);
  void image_bind(fan::graphics::image_nr_t nr);
  void image_unbind(fan::graphics::image_nr_t nr);
  fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr);
  void image_set_settings(fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings);
  fan::graphics::image_nr_t image_load(const fan::image::image_info_t& image_info);
  fan::graphics::image_nr_t image_load(const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_nr_t image_load(const fan::string& path);
  fan::graphics::image_nr_t image_load(const fan::string& path, const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size);
  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p);
  void image_unload(fan::graphics::image_nr_t nr);
  fan::graphics::image_nr_t create_missing_texture();
  fan::graphics::image_nr_t create_transparent_texture();
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::image_info_t& image_info);
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p);
  void image_reload(fan::graphics::image_nr_t nr, const std::string& image_info);
  void image_reload(fan::graphics::image_nr_t nr, const std::string& image_info, const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_nr_t image_create(const fan::color& color);
  fan::graphics::image_nr_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p);


  fan::graphics::camera_nr_t camera_create();
  fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr);
  void camera_erase(fan::graphics::camera_nr_t nr);
  fan::graphics::camera_nr_t camera_open(const fan::vec2& x, const fan::vec2& y);
  fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr);
  void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
  fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr);
  void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y);
  void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size);
  void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset);


  fan::graphics::viewport_nr_t viewport_create();
  fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr);
  void viewport_erase(fan::graphics::viewport_nr_t nr);
  fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr);
  fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr);
  void viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size, const fan::vec2& window_size);
  void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size, const fan::vec2& window_size);
  void viewport_zero(fan::graphics::viewport_nr_t nr);
  bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
  bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position);

  fan::graphics::context_functions_t context_functions;
  fan::graphics::context_t context;


  static fan::string read_shader(const fan::string& path) {
    fan::string code;
    fan::io::file::read(path, &code);
    return code;
  }

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
  static_assert(sizeof(loco_t::image_t) == 2, "update shaper_set_MaxKeySize");
  #define shaper_set_MaxKeySize 2 * 30
  /*
  * void _ShapeTypeChange(
    ShapeTypeIndex_t sti,
    KeyPackSize_t keypack_size,
    uint8_t *keypack,
    MaxElementPerBlock_t element_count,
    const void *old_renderdata,
    const void *old_data,
    void *new_renderdata,
    void *new_data
  ){
  */
  // will die if renderer has different sizes of structs
  #define shaper_set_ShapeTypeChange \
    __builtin_memcpy(new_renderdata, old_renderdata, element_count * get_loco()->shaper.GetRenderDataSize(sti)); \
    __builtin_memcpy(new_data, old_data, element_count * get_loco()->shaper.GetDataSize(sti));
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
      __builtin_memcpy(&a[i], &arg, sizeof(arg));
      i += sizeof(arg);
      }(args), ...);

    constexpr uintptr_t count = (!!(sizeof(Ts) + 1) + ...);
    static_assert(count % 2 == 0);
    constexpr uintptr_t last_sizeof = (static_cast<uintptr_t>(0), ..., sizeof(Ts));
    uintptr_t LastKeyOffset = s - last_sizeof - 1;
    gloco->shaper.PrepareKeysForAdd(&a, LastKeyOffset);
    return gloco->shaper.add(sti, &a, s, &rd, &d);
  }

  // unsafe
  //loco_t(const loco_t&) = delete;
  //loco_t& operator=(const loco_t&) = delete;
  //loco_t(loco_t&&) = delete;
  //loco_t& operator=(loco_t&&) = delete;

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
      polygon,
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

#if defined (fan_gui)
  using console_t = fan::console_t;
#endif

  using blending_t = uint8_t;
  using depth_t = uint16_t;

  void use();

  void camera_move(fan::graphics::context_camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction);

  using texture_packe0 = fan::graphics::texture_packe0;

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

  using get_image_data_cb = fan::graphics::image_data_t&(*)(shape_t*);

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

    get_image_data_cb get_image_data;

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

  #pragma pack(push, 1)

  struct vertex_t {
    fan::vec3 position;
    fan::color color;
  };

  struct polygon_vertex_t {
    fan::vec3 position;
    fan::color color;
    fan::vec3 offset;
    fan::vec3 angle;
    fan::vec2 rotation_point;
  };

  #pragma pack(pop)

  #if defined(loco_opengl)
  // opengl namespace
  struct opengl {
    #include <fan/graphics/opengl/engine_functions.h>
    #include <fan/graphics/opengl/2D/effects/blur.h>

    blur_t blur;

    fan::opengl::core::framebuffer_t m_framebuffer;
    fan::opengl::core::renderbuffer_t m_rbo;
    loco_t::image_t color_buffers[4];
    loco_t::shader_t m_fbo_final_shader;

    GLenum blend_src_factor = GL_SRC_ALPHA;
    GLenum blend_dst_factor = GL_ONE_MINUS_SRC_ALPHA;

    uint32_t fb_vao;
    uint32_t fb_vbo;

    #undef loco
  }gl;
#endif

#if defined(loco_vulkan)
  struct vulkan {
    #include <fan/graphics/vulkan/engine_functions.h>

    fan::vulkan::context_t::descriptor_t d_attachments;
    fan::vulkan::context_t::pipeline_t post_process;
    VkResult image_error = VK_SUCCESS;
  }vk;
#endif

  template <typename T, typename T2>
  static T2& get_render_data(shape_t* shape, T2 T::* attribute) {
    shaper_t::ShapeRenderData_t* data = shape->GetRenderData(gloco->shaper);
    return ((T*)data)->*attribute;
  }

  template <typename T, typename T2, typename T3, typename T4>
  static void modify_render_data_element_arr(shape_t* shape, T2 T::* attribute, std::size_t i, auto T4::*arr_member, const T3& value) {
    shaper_t::ShapeRenderData_t* data = shape->GetRenderData(gloco->shaper);

    // remove gloco
    if (gloco->window.renderer == renderer_t::opengl) {
      gloco->gl.modify_render_data_element_arr(shape, data, attribute, i, arr_member, value);
    }
    #if defined(loco_vulkan)
    else if (gloco->window.renderer == renderer_t::vulkan) {
      (((T*)data)->*attribute)[i].*arr_member = value;
      auto& data = gloco->shaper.ShapeList[*shape];
      gloco->shaper.ElementIsPartiallyEdited(
        data.sti,
        data.blid,
        data.ElementIndex,
        fan::member_offset(attribute),
        sizeof(T3)
      );
    }
    #endif
  }

  template <typename T, typename T2, typename T3>
  static void modify_render_data_element(shape_t* shape, T2 T::* attribute, const T3& value) {
    shaper_t::ShapeRenderData_t* data = shape->GetRenderData(gloco->shaper);

    // remove gloco
    if (gloco->window.renderer == renderer_t::opengl) {
      gloco->gl.modify_render_data_element(shape, data, attribute, value);
    }
#if defined(loco_vulkan)
    else if (gloco->window.renderer == renderer_t::vulkan) {
      ((T*)data)->*attribute = value;
      auto& data = gloco->shaper.ShapeList[*shape];
      gloco->shaper.ElementIsPartiallyEdited(
        data.sti,
        data.blid,
        data.ElementIndex,
        fan::member_offset(attribute),
        sizeof(T3)
      );
    }
#endif
  }

  template <typename T>
  static functions_t get_functions();

#pragma pack(push, 1)

#define st(name, viewport_inside) \
  template <bool cond> \
  struct CONCAT(name, _cond) { \
    template <typename T> \
    using d = typename fan::type_or_uint8_t<cond>::template d<T>; \
    viewport_inside \
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
      d<uint8_t> draw_mode;
      d<uint32_t> vertex_count;
    );
    st(common_t,
      d<depth_t> depth;
      d<blending_t> blending;
      d<loco_t::viewport_t> viewport;
      d<loco_t::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
      d<uint8_t> draw_mode;
      d<uint32_t> vertex_count;
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
      d<uint8_t> draw_mode;
      d<uint32_t> vertex_count;
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
    bool render_shapes_top = false;
    bool vsync = true;
    fan::vec2 window_size = -1;
    uint64_t window_flags = 0;
    uint8_t renderer = renderer_t::opengl;
  };

  uint64_t start_time = 0;

  // -1 no reload, opengl = 0 etc
  uint8_t reload_renderer_to = -1;
  #if defined(fan_gui)
  void load_fonts(auto& fonts, ImGuiIO& io, const std::string& name, f32_t font_size = 4);

  void init_imgui();
  void destroy_imgui();
  bool enable_overlay = false;
#endif
  void init_framebuffer();

  loco_t();
  loco_t(const properties_t& p);
  ~loco_t();

  void destroy();
  void close();

  // for renderer switch
  // input loco_t::renderer_t::
  void switch_renderer(uint8_t renderer);

  void draw_shapes();
  void process_shapes();
  void process_gui();
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

  fan::graphics::context_t& get_context() {
    return context;
  }

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

    void edit(int key, std::string_view action_name);

    void add_keycombo(std::initializer_list<int> keys, std::string_view action_name);

    bool is_active(std::string_view action_name, int state = loco_t::input_action_t::press);
    bool is_action_clicked(std::string_view action_name);
    bool is_action_down(std::string_view action_name);
    bool exists(std::string_view action_name);
    void insert_or_assign(int key, std::string_view action_name);

    std::unordered_map<std::string_view, action_data_t> input_actions;
  }input_action;

  static fan::vec2 transform_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera);

protected:
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix update_callback
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::function_t<void(loco_t*)>
  #define BLL_set_CPP_CopyAtPointerChange 1
  #include <BLL/BLL.h>
public:

  using update_callback_nr_t = update_callback_NodeReference_t;

  update_callback_t m_update_callback;

  std::vector<fan::function_t<void()>> single_queue;

  image_t default_texture;

  camera_impl_t orthographic_camera;
  camera_impl_t perspective_camera;

  fan::window_t window;
  bool idle_init = false;
  uv_idle_t idle_handle;
  bool timer_init = false;
  uv_timer_t timer_handle;

  int32_t target_fps = 165; // must be changed from function
  bool timer_enabled = target_fps > 0;

  fan::function_t<void()> main_loop; // bad, but forced

  f64_t delta_time = window.m_delta_time;

  std::vector<functions_t> shape_functions;

  // needs continous buffer
  std::vector<shaper_t::BlockProperties_t> BlockProperties;

  shaper_t shaper;
  
#if defined(fan_physics)
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
      filler,
      draw_mode,
      vertex_count
    };
  };

#pragma pack(pop)

  fan::vec2 get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport);
  fan::vec2 get_mouse_position();

  static fan::vec2 translate_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera);
  fan::vec2 translate_position(const fan::vec2& p);

  bool is_mouse_clicked(int button = fan::mouse_left);
  bool is_mouse_down(int button = fan::mouse_left);
  bool is_mouse_released(int button = fan::mouse_left);
  fan::vec2 get_mouse_drag(int button = fan::mouse_left);

  bool is_key_pressed(int key);
  bool is_key_down(int key);
  bool is_key_released(int key);

  struct shape_t : shaper_t::ShapeID_t{
    using shaper_t::ShapeID_t::ShapeID_t;
    shape_t();
    shape_t(shaper_t::ShapeID_t&& s);
    shape_t(const shaper_t::ShapeID_t& s);

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
      else if constexpr (std::is_same_v<T, polygon_t::properties_t>) {
        *this = gloco->polygon.push_back(properties);
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
    shape_t(shape_t&& s);
    shape_t(const shape_t& s);
    shape_t& operator=(const shape_t& s);
    shape_t& operator=(shape_t&& s);
    ~shape_t();

    void remove();

    void erase();

    // many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t
    uint16_t get_shape_type() const;

    template <typename T>
    void set_position(const fan::vec2_wrap_t<T>& position) {
      gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_position2(this, position);
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
    fan::graphics::image_data_t& get_image_data();

    std::array<loco_t::image_t, 30> get_images();
    void set_images(const std::array<loco_t::image_t, 30>& images);

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

    void reload(uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter = fan::graphics::image_filter::linear);
    void reload(uint8_t format, const fan::vec2& image_size, uint32_t filter = fan::graphics::image_filter::linear);

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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{5, "in_rotation_vector"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector))},
      shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{7, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    });

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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_color"}, 4, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
      shape_gl_init_t{{1, "in_src"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
      shape_gl_init_t{{2, "in_dst"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
    });

    struct properties_t {
      using type_t = line_t;

      fan::color color = fan::colors::white;
      fan::vec3 src;
      fan::vec3 dst;

      bool blending = false;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::lines;
      uint32_t vertex_count = 2;
    };


    shape_t push_back(const properties_t& properties);

  }line;

  struct rectangle_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::rectangle;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t pad;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::color outline_color;
      fan::vec3 angle;
      f32_t pad2;
    };
    struct ri_t {
      
    };

#pragma pack(pop)

    // accounts padding
    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 4, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{4, "in_outline_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color))},
      shape_gl_init_t{{5, "in_angle"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
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
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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
      fan::vec3 pad;
    };
    struct ri_t {
      // main image + light buffer + 30
      std::array<loco_t::image_t, 30> images; // what about tc_pos and tc_size
    };

#pragma pack(pop)

  inline static auto locations = std::to_array({
    shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
    shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
    shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
    shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
    shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
    shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
    shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
    shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
    shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
    shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
  });

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
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
      shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
      shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
    });

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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;

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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
      shape_gl_init_t{{1, "in_radius"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
      shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
      shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
      shape_gl_init_t{{4, "in_rotation_vector"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector)) },
      shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
      shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))}
    });

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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
      shape_gl_init_t{{1, "in_center0"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, center0)) },
      shape_gl_init_t{{2, "in_center1"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, center1)) },
      shape_gl_init_t{{3, "in_radius"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
      shape_gl_init_t{{4, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
      shape_gl_init_t{{5, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
      shape_gl_init_t{{6, "in_rotation_vector"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector)) },
      shape_gl_init_t{{7, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
      shape_gl_init_t{{8, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{9, "in_outline_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color)) },
    });

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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
    };
    loco_t::shape_t push_back(const capsule_t::properties_t& properties);
  }capsule;

  
#pragma pack(push, 1)

  struct polygon_t {
    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::polygon;
    static constexpr int kpi = kp::common;


    // vertex
    struct vi_t {
      
    };
    struct ri_t {
      uint32_t buffer_size = 0;
      fan::opengl::core::vao_t vao;
      fan::opengl::core::vbo_t vbo;
    };

#pragma pack(pop)

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, position)) },
      shape_gl_init_t{{1, "in_color"}, 4, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, color)) },
      shape_gl_init_t{{2, "in_offset"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, offset)) },
      shape_gl_init_t{{3, "in_angle"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, angle)) },
      shape_gl_init_t{{4, "in_rotation_point"}, 2, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, rotation_point)) },
    });

    struct properties_t {
      using type_t = polygon_t;
      fan::vec3 position = 0;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;
      std::vector<vertex_t> vertices;
      bool blending = true;
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 0;
    };
    loco_t::shape_t push_back(const properties_t& properties);
  }polygon;

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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, size)},
      shape_gl_init_t{{2, "in_grid_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, grid_size)},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, rotation_point)},
      shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, color)},
      shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, angle)},
    });

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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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
      std::array<loco_t::image_t, 3> images_rest; // 3 + 1 (pk)
      uint8_t format = fan::pixel_format::undefined;
    };

#pragma pack(pop)

  inline static auto locations = std::to_array({
    shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
    shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
    shape_gl_init_t{{2, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
    shape_gl_init_t{{3, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))}
  });

    struct properties_t {
      using type_t = universal_image_renderer_t;

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
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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
      std::array<fan::color, 4> color;
      fan::vec3 angle;
    };
    struct ri_t {

    };

#pragma pack(pop)

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 0)},
      shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 1)},
      shape_gl_init_t{{5, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 2)},
      shape_gl_init_t{{6, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 3)},
      shape_gl_init_t{{7, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    });

    struct properties_t {
      using type_t = gradient_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      std::array<fan::color, 4> color = {
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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
      shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
      shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
    });

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
      bool blending = true;

      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t,  position)},
      shape_gl_init_t{{1, "in_size"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))}
    });

    struct properties_t {
      using type_t = rectangle_t;

      fan::vec3 position = 0;
      fan::vec3 size = 0;
      fan::color color = fan::colors::white;
      bool blending = false;
      fan::vec3 angle = 0;

      loco_t::camera_t camera = gloco->perspective_camera.camera;
      loco_t::viewport_t viewport = gloco->perspective_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 36;
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

    inline static auto locations = std::to_array({
      shape_gl_init_t{{0, "in_color"}, 4, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
      shape_gl_init_t{{1, "in_src"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
      shape_gl_init_t{{2, "in_dst"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
    });

    struct properties_t {
      using type_t = line_t;

      fan::color color = fan::colors::white;
      fan::vec3 src;
      fan::vec3 dst;

      bool blending = false;

      loco_t::camera_t camera = gloco->perspective_camera.camera;
      loco_t::viewport_t viewport = gloco->perspective_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::lines;
      uint32_t vertex_count = 2;
    };

    shape_t push_back(const properties_t& properties);

  }line3d;

  //-------------------------------------shapes-------------------------------------

  template <typename T>
  inline void shape_open(T* shape, const fan::string& vertex, const fan::string& fragment, loco_t::shaper_t::ShapeRenderDataSize_t instance_count = 1, bool instanced = true) {    
    loco_t::shader_t shader = shader_create();

    shader_set_vertex(shader,
      read_shader(vertex)
    );

    shader_set_fragment(shader,
      read_shader(fragment)
    );
    
    shader_compile(shader);

    decltype(loco_t::shaper_t::BlockProperties_t::renderer) data{loco_t::shaper_t::BlockProperties_t::gl_t{}};

    if (window.renderer == renderer_t::opengl) {
      loco_t::shaper_t::BlockProperties_t::gl_t d;
      d.locations = decltype(loco_t::shaper_t::BlockProperties_t::gl_t::locations)(std::begin(T::locations), std::end(T::locations));
      d.shader = shader;
      d.instanced = instanced;
      data = d;
    }
#if defined(loco_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      loco_t::shaper_t::BlockProperties_t::vk_t vk;

      // 2 for rect instance, upv
      static constexpr auto vulkan_buffer_count = 3;
      decltype(vk.shape_data.m_descriptor)::properties_t rectp;
      // image
      //uint32_t ds_offset = 3;
      auto& shaderd = *(fan::vulkan::context_t::shader_t*)gloco->context_functions.shader_get(&gloco->context.vk, shader);
      uint32_t ds_offset = 2;
      vk.shape_data.open(gloco->context.vk, 1);
      vk.shape_data.allocate(gloco->context.vk, 0xffffff);
      
      std::array<fan::vulkan::write_descriptor_set_t, vulkan_buffer_count> ds_properties{ {{0}} };
      {
        ds_properties[0].binding = 0;
        ds_properties[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ds_properties[0].flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        ds_properties[0].range = VK_WHOLE_SIZE;
        ds_properties[0].buffer = vk.shape_data.common.memory[gloco->get_context().vk.current_frame].buffer;
        ds_properties[0].dst_binding = 0;

        ds_properties[1].binding = 1;
        ds_properties[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ds_properties[1].flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        ds_properties[1].buffer = shaderd.projection_view_block.common.memory[gloco->get_context().vk.current_frame].buffer;
        ds_properties[1].range = shaderd.projection_view_block.m_size;
        ds_properties[1].dst_binding = 1;

        VkDescriptorImageInfo imageInfo{};
        auto img = std::get<fan::vulkan::context_t::image_t>(gloco->image_get(gloco->default_texture));
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = img.image_view;
        imageInfo.sampler = img.sampler;

        ds_properties[2].use_image = 1;
        ds_properties[2].binding = 2;
        ds_properties[2].dst_binding = 2;
        ds_properties[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ds_properties[2].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
        for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
          ds_properties[ds_offset].image_infos[i] = imageInfo;
        }

        //imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        //imageInfo.imageView = gloco->get_context().vk.postProcessedColorImageViews[0].image_view;
        //imageInfo.sampler = sampler;

        //imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        //ds_properties[ds_offset + 1].use_image = 1;
        //ds_properties[ds_offset + 1].binding = 4;
        //ds_properties[ds_offset + 1].dst_binding = 4;
        //ds_properties[ds_offset + 1].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        //ds_properties[ds_offset + 1].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
        //for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
        //  ds_properties[ds_offset + 1].image_infos[i] = imageInfo;
        //}
      }
      
      vk.shape_data.open_descriptors(gloco->context.vk, {ds_properties.begin(), ds_properties.end()});
      vk.shape_data.m_descriptor.update(context.vk, 3, 0);
      fan::vulkan::context_t::pipeline_t p;
      fan::vulkan::context_t::pipeline_t::properties_t pipe_p;
      VkPipelineColorBlendAttachmentState attachment = fan::vulkan::get_default_color_blend();
      pipe_p.color_blend_attachment_count = 1;
      pipe_p.color_blend_attachment = &attachment;
      pipe_p.shader = shader;
      pipe_p.descriptor_layout = &vk.shape_data.m_descriptor.m_layout;
      pipe_p.descriptor_layout_count = /*vulkan_buffer_count*/1;
      pipe_p.push_constants_size = sizeof(fan::vulkan::context_t::push_constants_t);
      p.open(context.vk, pipe_p);
      vk.pipeline = p;
      data = vk;
    }
#endif

    shaper_t::BlockProperties_t bp;
    bp.MaxElementPerBlock = (loco_t::shaper_t::MaxElementPerBlock_t)MaxElementPerBlock;
    bp.RenderDataSize = (decltype(loco_t::shaper_t::BlockProperties_t::RenderDataSize))(sizeof(typename T::vi_t) * instance_count);
    bp.DataSize = sizeof(typename T::ri_t);
    bp.renderer = data;

    gloco->shaper.SetShapeType(
      shape->shape_type,
      bp
    );

    loco_t::functions_t functions = loco_t::get_functions<typename T::vi_t>();
    gloco->shape_functions.push_back(functions);
  }


#if defined(loco_sprite)
  loco_t::shader_t get_sprite_vertex_shader(const fan::string& fragment);
#endif


#if defined(loco_vfi)
  #include <fan/graphics/gui/vfi.h>
#endif
  vfi_t vfi;

//#if defined(loco_texture_pack)
//#endif

  fan::color clear_color = { 
    /*0.10f, 0.10f, 0.131f, 1.f */
    0.f, 0.f, 0.f, 1.f
  };

  struct lighting_t {
    static constexpr const char* ambient_name = "lighting_ambient";
    fan::vec3 ambient = fan::vec3(1, 1, 1);
  }lighting;

  //gui
#if defined(fan_gui)
  fan::console_t console;
  bool render_console = false;
  bool toggle_fps = false;
  bool render_settings_menu = 0;

  ImFont* fonts[6];
  ImFont* fonts_bold[6];

  #include <fan/graphics/gui/settings_menu.h>
  settings_menu_t settings_menu;
#endif
  bool render_shapes_top = false;
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


  void printclnn(auto&&... values) {
#if defined (fan_gui)
    ([&](const auto& value) {
      std::ostringstream oss;
      oss << value;
      console.print(oss.str() + " ", 0);
      }(values), ...);
#endif
  }
  void printcl(auto&&... values) {
#if defined(fan_gui)
    printclnn(values...);
    console.print("\n", 0);
#endif
  }

  void printclnnh(int highlight, auto&&... values) {
#if defined(fan_gui)
    ([&](const auto& value) {
      std::ostringstream oss;
      oss << value;
      console.print(oss.str() + " ", highlight);
    }(values), ...);
#endif
  }

  void printclh(int highlight, auto&&... values) {
#if defined(fan_gui)
    printclnnh(highlight, values...);
    console.print("\n", highlight);
#endif
  }


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

        if (gloco->image_get(vi_image).size == size) {
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

    cudaArray_t& get_array(uint32_t index_t) {
      return wresources[index_t].cuda_array;
    }

    struct graphics_resource_t {
      void open(int texture_id) {
        fan::cuda::check_error(cudaGraphicsGLRegisterImage(&resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
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

#if defined(loco_audio)
  fan::system_audio_t system_audio;
  fan::audio_t audio;

  fan::audio_t::piece_t piece_hover, piece_click;
#endif
};

namespace fan {
  namespace graphics {
    using vfi_t = loco_t::vfi_t;

    using engine_t = loco_t;
    using image_t = loco_t::image_t;
    using camera_impl_t = loco_t::camera_impl_t;
    using camera_t = camera_impl_t;
    using viewport_t = loco_t::viewport_t;

    // fan_track_allocations() must be called in global scope before calling this function
    void render_allocations_plot();
    void add_input_action(const int* keys, std::size_t count, std::string_view action_name);
    void add_input_action(std::initializer_list<int> keys, std::string_view action_name);
    void add_input_action(int key, std::string_view action_name);
    bool is_input_action_active(std::string_view action_name, int pstate = loco_t::input_action_t::press);

    fan::vec2 get_mouse_position();
    fan::vec2 get_mouse_position(const fan::graphics::camera_t& camera);
    fan::vec2 get_mouse_position(
      const loco_t::camera_t& camera, 
      const loco_t::viewport_t& viewport
    );

    void text_partial_render(const std::string& text, size_t render_pos, f32_t wrap_width, f32_t line_spacing = 0);
  }
}


namespace fan {
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

inline bool init_fan_track_opengl_print = []() {
  fan_opengl_track_print = [](std::string func_name, uint64_t elapsed){
    fan::printclnnh(fan::graphics::highlight_e::text, func_name + ":");
    fan::printclh(fan::graphics::highlight_e::warning,  fan::to_string(elapsed / 1e+6) + "ms");
  };
  return 1;
}();

#if defined(fan_json)
  namespace fan {
    namespace graphics {
      bool shape_to_json(loco_t::shape_t& shape, fan::json* json);

      bool json_to_shape(const fan::json& in, loco_t::shape_t* shape);

      bool shape_serialize(loco_t::shape_t& shape, fan::json* out);
    }
  }

  namespace fan {

    namespace graphics {
      bool shape_to_bin(loco_t::shape_t& shape, std::vector<uint8_t>* data);

      bool bin_to_shape(const std::vector<uint8_t>& in, loco_t::shape_t* shape, uint64_t& offset);

      bool shape_serialize(loco_t::shape_t& shape, std::vector<uint8_t>* out);

      struct shape_deserialize_t {
        struct {
          // json::iterator doesnt support union
          // i dont want to use variant either so i accept few extra bytes
          json::const_iterator it;
          uint64_t offset = 0;
        }data;
        bool init = false;

        bool iterate(const fan::json& json, loco_t::shape_t* shape);
        bool iterate(const std::vector<uint8_t>& bin_data, loco_t::shape_t* shape);
      };
    }
  }

#endif


#include <fan/graphics/collider.h>

//vk

#if defined(loco_vulkan)
  #include <fan/graphics/vulkan/uniform_block.h>
  #include <fan/graphics/vulkan/memory.h>
#endif

inline bool fan__init_list = []{
  fan::graphics::get_camera_list = [](uint8_t* context) -> uint8_t* {
    auto ptr = OFFSETLESS(context, loco_t, context);
    return (uint8_t*)&ptr->camera_list;
  };
  fan::graphics::get_shader_list = [](uint8_t* context) -> uint8_t* {
    return (uint8_t*)&OFFSETLESS(context, loco_t, context)->shader_list;
  };
  fan::graphics::get_image_list = [](uint8_t* context) -> uint8_t* {
    return (uint8_t*)&OFFSETLESS(context, loco_t, context)->image_list;
  };
  fan::graphics::get_viewport_list = [](uint8_t* context) -> uint8_t* {
    return (uint8_t*)&OFFSETLESS(context, loco_t, context)->viewport_list;
  };  
  return 0;
}();

#if defined(loco_audio)
namespace fan {
  namespace audio {
    using piece_t = fan::audio_t::piece_t;

    fan::audio_t::piece_t open_piece(const std::string& path, fan::audio_t::PieceFlag::t flags = 0);
    /// <summary>
    /// Function checks if the stored pointer equals to nullptr. Does NOT check for actual validity.
    /// </summary>
    /// <param name="piece">Given piece to validate.</param>
    /// <returns></returns>
    bool is_piece_valid(fan::audio_t::piece_t piece);

    void play(fan::audio_t::piece_t piece, uint32_t group_id = 0, bool loop = false);
    void resume(uint32_t group_id = 0);
    void pause(uint32_t group_id = 0);
    f32_t get_volume();
    void set_volume(f32_t volume);
  }
#if defined(fan_gui)
  namespace graphics {
    namespace gui {
      void process_loop();
    }
  }
#endif
}
#endif