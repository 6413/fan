module;

#include <fan/types/types.h>
#include <fan/math/math.h>

#include <fan/graphics/opengl/init.h>

#define loco_audio

#define loco_framebuffer
#define loco_post_process
#define loco_vfi

#define loco_physics

#if defined(fan_gui)
  #include <deque>
#endif

#include <cstring>
#include <memory> // shared_ptr tp0.h
#include <array>

#define loco_opengl

#ifndef camera_list
  #define __fan_internal_camera_list (*(fan::graphics::camera_list_t*)fan::graphics::get_camera_list((uint8_t*)this))
#endif

#ifndef shader_list
  #define __fan_internal_shader_list (*(fan::graphics::shader_list_t*)fan::graphics::get_shader_list((uint8_t*)this))
#endif

#ifndef image_list
  #define __fan_internal_image_list (*(fan::graphics::image_list_t*)fan::graphics::get_image_list((uint8_t*)this))
#endif

#ifndef viewport_list
  #define __fan_internal_viewport_list (*(fan::graphics::viewport_list_t*)fan::graphics::get_viewport_list((uint8_t*)this))
#endif

// shaper

#include <fan/time/time.h>

#if defined(fan_compiler_msvc)
  #ifndef fan_std23
    #define fan_std23
  #endif
#endif
#include <fan/memory/memory.hpp>

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/imgui_impl_opengl3.h>
  #if defined(fan_vulkan)
    #include <fan/imgui/imgui_impl_vulkan.h>
  #endif
  #include <fan/imgui/imgui_impl_glfw.h>
  #include <fan/imgui/imgui_neo_sequencer.h>
  #include <fan/imgui/implot.h>
#endif

//#include <fan/graphics/algorithm/FastNoiseLite.h>

#if defined(fan_vulkan)
#include <fan/graphics/vulkan/core.h>
#endif

#ifndef __generic_malloc
#define __generic_malloc(n) malloc(n)
#endif

#ifndef __generic_realloc
#define __generic_realloc(ptr, n) realloc(ptr, n)
#endif

#ifndef __generic_free
#define __generic_free(ptr) free(ptr)
#endif

#if defined(fan_gui)
#include <fan/imgui/imgui_internal.h>
#include <fan/graphics/gui/imgui_themes.h>
#endif

#include <fan/event/types.h>
#include <uv.h>

// +cuda
#if __has_include("cuda.h")
  #include "cuda_runtime.h"
  #include <cuda.h>
  #include <nvcuvid.h>
  #define loco_cuda
#endif

export module fan.graphics.loco;

export import fan.event;
export import fan.file_dialog;

export import fan.window;
export import fan.types.color;
export import fan.random;

export import fan.io.file;
export import fan.types.fstring;
#if defined(fan_physics)
  import fan.physics.b2_integration;
#endif

#if defined(loco_audio)
  export import fan.audio;
#endif

#if defined(fan_gui)
  export import fan.console;
#endif

import fan.graphics.webp;
export import fan.graphics.opengl.core;

export import fan.physics.collision.rectangle;

#if defined(fan_json)

export import fan.types.json;

export namespace fan {
  struct json_stream_parser_t {
    std::string buf;

    struct parsed_result {
      bool success;
      fan::json value;
      std::string error;
    };

    [[nodiscard]]
    std::pair<size_t, size_t> find_next_json_bounds(std::string_view s, size_t pos = 0) const noexcept {
      pos = s.find('{', pos);
      if (pos == std::string::npos) return { pos, pos };

      int depth = 0;
      bool in_str = false;

      for (size_t i = pos; i < s.length(); i++) {
        char c = s[i];
        if (c == '"' && (i == 0 || s[i - 1] != '\\')) in_str = !in_str;
        else if (!in_str) {
          if (c == '{') depth++;
          else if (c == '}' && --depth == 0) return { pos, i + 1 };
        }
      }
      return { pos, std::string::npos };
    }

    std::vector<parsed_result> process(std::string_view chunk) {
      std::vector<parsed_result> results;
      buf += chunk;
      size_t pos = 0;

      while (pos < buf.length()) {
        auto [start, end] = find_next_json_bounds(buf, pos);
        if (start == std::string::npos) break;
        if (end == std::string::npos) {
          buf = buf.substr(start);
          break;
        }

        try {
          results.push_back({ true, fan::json::parse(buf.data() + start, buf.data() + end - start), "" });
        }
        catch (const fan::json::parse_error& e) {
          results.push_back({ false, fan::json{}, e.what() });
        }

        pos = buf.find('{', end);
        if (pos == std::string::npos) pos = end;
      }

      buf = pos < buf.length() ? buf.substr(pos) : "";
      return results;
    }

    void clear() noexcept { buf.clear(); }
  };
}

#endif

#if defined(fan_gui)
namespace fan {
  namespace graphics {
    namespace gui {
      void render_allocations_plot();
      void process_loop();
    }
  }
}
#endif

#if defined(loco_cuda)

export namespace fan {
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

export extern "C" {
  extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
}

#endif

// -cuda

//#define debug_shape_t

export struct loco_t;

#if defined(fan_physics)
namespace fan {
  namespace graphics {
    void open_bcol();
    void close_bcol();
  }
}
#endif

// to set new loco use gloco = new_loco;
struct global_loco_t {

  loco_t* loco = nullptr;

  operator loco_t* () {
    return loco;
  }

  global_loco_t& operator=(loco_t* l) {
    loco = l;
    return *this;
  }
  loco_t* operator->() {
    return loco;
  }
};

// might crash if pch or lib is built with extern/inline so if its different, 
// it will crash in random places
export inline global_loco_t gloco;

export namespace fan {
  namespace graphics {

    struct engine_init_t {
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix init_callback
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType std::function<void(loco_t*)>
#define BLL_set_CPP_CopyAtPointerChange 1
#undef BLL_set_CPP_ConstructDestruct
#include <BLL/BLL.h>

      using init_callback_nr_t = init_callback_NodeReference_t;
    };

    // cbs called every time engine opens
    inline engine_init_t::init_callback_t engine_init_cbs;

    inline uint32_t get_draw_mode(uint8_t internal_draw_mode);

#if defined(fan_gui)
    namespace gui {
      bool render_blank_window(const std::string& name) {
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        return ImGui::Begin(name.c_str(), 0,
          ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
          ImGuiWindowFlags_NoResize |
          ImGuiWindowFlags_NoTitleBar
        );
      }
    }
#endif
    struct context_shader_t {
      context_shader_t() {}
      ~context_shader_t() {}
      union {
        fan::opengl::context_t::shader_t gl;
#if defined(fan_vulkan)
        fan::vulkan::context_t::shader_t vk;
#endif
      };
    };
    struct context_image_t {
      context_image_t() {}
      ~context_image_t() {}
      union {
        fan::opengl::context_t::image_t gl;
#if defined(fan_vulkan)
        fan::vulkan::context_t::image_t vk; // note vk::image_t uses vector 
#endif
      };
    };
    struct context_t {
      context_t() {}
      ~context_t() {}
      union {
        fan::opengl::context_t gl;
#if defined(fan_vulkan)
        fan::vulkan::context_t vk;
#endif
      };
    };
  }
}

namespace fan {
  template <bool cond>
  struct type_or_uint8_t {
    template <typename T>
    using d = std::conditional_t<cond, T, uint8_t>;
  };
}

//#include <fan/graphics/vulkan/ssbo.h>
export struct loco_t {

  bool fan__init_list = [] {
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

  struct shape_t;

  fan::graphics::shader_nr_t shader_create() {
    return context_functions.shader_create(&context);
  }
  // warning does deep copy, addresses can die
  fan::graphics::context_shader_t shader_get(fan::graphics::shader_nr_t nr) {
    fan::graphics::context_shader_t context_shader;
    if (window.renderer == renderer_t::opengl) {
      context_shader.gl = *(fan::opengl::context_t::shader_t*)context_functions.shader_get(&context, nr);
    }
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      context_shader.vk = *(fan::vulkan::context_t::shader_t*)context_functions.shader_get(&context, nr);
    }
#endif
    return context_shader;
  }

  void shader_erase(fan::graphics::shader_nr_t nr) {
    context_functions.shader_erase(&context, nr);
  }

  void shader_use(fan::graphics::shader_nr_t nr) {
    context_functions.shader_use(&context, nr);
  }

  void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
    context_functions.shader_set_vertex(&context, nr, vertex_code);
  }

  void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
    context_functions.shader_set_fragment(&context, nr, fragment_code);
  }

  bool shader_compile(fan::graphics::shader_nr_t nr) {
    return context_functions.shader_compile(&context, nr);
  }

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
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      fan::throw_error("todo");
    }
#endif
  }

  fan::graphics::camera_list_t camera_list;
  fan::graphics::shader_list_t shader_list;
  fan::graphics::image_list_t image_list;
  fan::graphics::viewport_list_t viewport_list;

  std::vector<uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, GLenum format, fan::vec2 uvp = 0, fan::vec2 uvs = 1) {
    fan::throw_error("");
    return {};
  }

  fan::graphics::image_nr_t image_create() {
    return context_functions.image_create(&context);
  }

  fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr) {
    fan::graphics::context_image_t img;
    if (window.renderer == renderer_t::opengl) {
      img.gl = *(fan::opengl::context_t::image_t*)context_functions.image_get(&context, nr);
    }
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      img.vk = *(fan::vulkan::context_t::image_t*)context_functions.image_get(&context, nr);
    }
#endif
    return img;
  }

  uint64_t image_get_handle(fan::graphics::image_nr_t nr) {
    return context_functions.image_get_handle(&context, nr);
  }

  fan::graphics::image_data_t& image_get_data(fan::graphics::image_nr_t nr) {
    return image_list[nr];
  }

  void image_erase(fan::graphics::image_nr_t nr) {
    context_functions.image_erase(&context, nr);
  }

  void image_bind(fan::graphics::image_nr_t nr) {
    context_functions.image_bind(&context, nr);
  }

  void image_unbind(fan::graphics::image_nr_t nr) {
    context_functions.image_unbind(&context, nr);
  }

  fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr) {
    return context_functions.image_get_settings(&context, nr);
  }

  void image_set_settings(fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
    context_functions.image_set_settings(&context, nr, settings);
  }

  fan::graphics::image_nr_t image_load(const fan::image::image_info_t& image_info) {
    return context_functions.image_load_info(&context, image_info);
  }

  fan::graphics::image_nr_t image_load(const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    return context_functions.image_load_info_props(&context, image_info, p);
  }

  fan::graphics::image_nr_t image_load(const std::string& path) {
    return context_functions.image_load_path(&context, path);
  }

  fan::graphics::image_nr_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p) {
    return context_functions.image_load_path_props(&context, path, p);
  }

  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size) {
    return context_functions.image_load_colors(&context, colors, size);
  }

  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
    return context_functions.image_load_colors_props(&context, colors, size, p);
  }

  void image_unload(fan::graphics::image_nr_t nr) {
    context_functions.image_unload(&context, nr);
  }

  fan::graphics::image_nr_t create_missing_texture() {
    return context_functions.create_missing_texture(&context);
  }

  fan::graphics::image_nr_t create_transparent_texture() {
    return context_functions.create_transparent_texture(&context);
  }

  void image_reload(fan::graphics::image_nr_t nr, const fan::image::image_info_t& image_info) {
    context_functions.image_reload_image_info(&context, nr, image_info);
  }
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    context_functions.image_reload_image_info_props(&context, nr, image_info, p);
  }
  void image_reload(fan::graphics::image_nr_t nr, const std::string& path) {
    context_functions.image_reload_path(&context, nr, path);
  }
  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p) {
    context_functions.image_reload_path_props(&context, nr, path, p);
  }

  fan::graphics::image_nr_t image_create(const fan::color& color) {
    return context_functions.image_create_color(&context, color);
  }

  fan::graphics::image_nr_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p) {
    return context_functions.image_create_color_props(&context, color, p);
  }

  fan::graphics::camera_nr_t camera_create() {
    return context_functions.camera_create(&context);
  }

  fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr) {
    return context_functions.camera_get(&context, nr);
  }

  void camera_erase(fan::graphics::camera_nr_t nr) {
    context_functions.camera_erase(&context, nr);
  }

  fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y) {
    return context_functions.camera_create_params(&context, x, y);
  }

  fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr) {
    return context_functions.camera_get_position(&context, nr);
  }

  void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    context_functions.camera_set_position(&context, nr, cp);
  }

  fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr) {
    return context_functions.camera_get_size(&context, nr);
  }

  void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
    context_functions.camera_set_ortho(&context, nr, x, y);
  }

  void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
    context_functions.camera_set_perspective(&context, nr, fov, window_size);
  }

  void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
    context_functions.camera_rotate(&context, nr, offset);
  }

  fan::graphics::viewport_nr_t viewport_create() {
    return context_functions.viewport_create(&context);
  }
  fan::graphics::viewport_nr_t viewport_create(const fan::vec2& viewport_position, const fan::vec2& viewport_size, const fan::vec2& window_size) {
    return context_functions.viewport_create_params(&context, viewport_position, viewport_size, window_size);
  }

  fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr) {
    return context_functions.viewport_get(&context, nr);
  }

  void viewport_erase(fan::graphics::viewport_nr_t nr) {
    context_functions.viewport_erase(&context, nr);
  }

  fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr) {
    return context_functions.viewport_get_position(&context, nr);
  }

  fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr) {
    return context_functions.viewport_get_size(&context, nr);
  }

  void viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size, const fan::vec2& window_size) {
    context_functions.viewport_set(&context, viewport_position, viewport_size, window_size);
  }

  void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size, const fan::vec2& window_size) {
    context_functions.viewport_set_nr(&context, nr, viewport_position, viewport_size, window_size);
  }

  void viewport_zero(fan::graphics::viewport_nr_t nr) {
    context_functions.viewport_zero(&context, nr);
  }

  bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    return context_functions.viewport_inside(&context, nr, position);
  }

  bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    return context_functions.viewport_inside_wir(&context, nr, position);
  }

  fan::graphics::context_functions_t context_functions;
  fan::graphics::context_t context;


#include <fan/tp/tp0.h>

  static std::string read_shader(const std::string& path) {
    std::string code;
    fan::io::file::read(path, &code);
    return code;
  }

  static uint8_t* A_resize(void* ptr, uintptr_t size) {
    if (ptr) {
      if (size) {
        void* rptr = (void*)__generic_realloc(ptr, size);
        if (rptr == 0) {
          fan::throw_error_impl();
        }
        return (uint8_t*)rptr;
      }
      else {
        __generic_free(ptr);
        return 0;
      }
    }
    else {
      if (size) {
        void* rptr = (void*)__generic_malloc(size);
        if (rptr == 0) {
          fan::throw_error_impl();
        }
        return (uint8_t*)rptr;
      }
      else {
        return 0;
      }
    }
  }

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

  static void shaper_deep_copy(loco_t::shape_t* dst, const loco_t::shape_t* const src, loco_t::shaper_t::ShapeTypeIndex_t sti) {
    // alloc can be avoided inside switch
    uint8_t* KeyPack = new uint8_t[gloco->shaper.GetKeysSize(*src)];
    gloco->shaper.WriteKeys(*src, KeyPack);

    auto _vi = src->GetRenderData(gloco->shaper);
    auto vlen = gloco->shaper.GetRenderDataSize(sti);
    uint8_t* vi = new uint8_t[vlen];
    std::memcpy(vi, _vi, vlen);

    auto _ri = src->GetData(gloco->shaper);
    auto rlen = gloco->shaper.GetDataSize(sti);

    uint8_t* ri = new uint8_t[rlen];
    std::memcpy(ri, _ri, rlen);

    *dst = gloco->shaper.add(
      sti,
      KeyPack,
      gloco->shaper.GetKeysSize(*src),
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
      sprite,
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

  void use() {
    gloco = this;
  }

  void camera_move(fan::graphics::context_camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction = 12) {
    camera.velocity /= friction * dt + 1;
    static constexpr auto minimum_velocity = 0.001;
    static constexpr f32_t camera_rotate_speed = 100;
    if (camera.velocity.x < minimum_velocity && camera.velocity.x > -minimum_velocity) {
      camera.velocity.x = 0;
    }
    if (camera.velocity.y < minimum_velocity && camera.velocity.y > -minimum_velocity) {
      camera.velocity.y = 0;
    }
    if (camera.velocity.z < minimum_velocity && camera.velocity.z > -minimum_velocity) {
      camera.velocity.z = 0;
    }

    f64_t msd = (movement_speed * dt);
    if (gloco->window.key_pressed(fan::input::key_w)) {
      camera.velocity += camera.m_front * msd;
    }
    if (gloco->window.key_pressed(fan::input::key_s)) {
      camera.velocity -= camera.m_front * msd;
    }
    if (gloco->window.key_pressed(fan::input::key_a)) {
      camera.velocity -= camera.m_right * msd;
    }
    if (gloco->window.key_pressed(fan::input::key_d)) {
      camera.velocity += camera.m_right * msd;
    }

    if (gloco->window.key_pressed(fan::input::key_space)) {
       camera.velocity.y += msd;
    }
    if (gloco->window.key_pressed(fan::input::key_left_shift)) {
      camera.velocity.y -= msd;
    }

    f64_t rotate = camera.sensitivity * camera_rotate_speed * gloco->delta_time;
    if (gloco->window.key_pressed(fan::input::key_left)) {
      camera.set_yaw(camera.get_yaw() - rotate);
    }
    if (gloco->window.key_pressed(fan::input::key_right)) {
      camera.set_yaw(camera.get_yaw() + rotate);
    }
    if (gloco->window.key_pressed(fan::input::key_up)) {
      camera.set_pitch(camera.get_pitch() + rotate);
    }
    if (gloco->window.key_pressed(fan::input::key_down)) {
      camera.set_pitch(camera.get_pitch() - rotate);
    }

    camera.position += camera.velocity * gloco->delta_time;
    camera.update_view();

    camera.m_view = camera.get_view_matrix();
  }

#include <fan/graphics/opengl/texture_pack.h>

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

#if defined(fan_vulkan)
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
  static void modify_render_data_element_arr(shape_t* shape, T2 T::* attribute, std::size_t i, auto T4::* arr_member, const T3& value) {
    shaper_t::ShapeRenderData_t* data = shape->GetRenderData(gloco->shaper);

    // remove gloco
    if (gloco->window.renderer == renderer_t::opengl) {
      gloco->gl.modify_render_data_element_arr(shape, data, attribute, i, arr_member, value);
    }
#if defined(fan_vulkan)
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
#if defined(fan_vulkan)
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

public:

  std::vector<std::function<void()>> m_pre_draw;
  std::vector<std::function<void()>> m_post_draw;


  struct properties_t {
    bool render_shapes_top = false;
    bool vsync = true;
    fan::vec2 window_size = -1;
    uint64_t window_flags = 0;
    uint8_t renderer = renderer_t::opengl;
  };

  uint64_t start_time = 0;



#define shaper_get_key_safe(return_type, kps_type, variable) \
  [KeyPack] ()-> auto& { \
    auto o = gloco->shaper.GetKeyOffset( \
      offsetof(loco_t::kps_t::CONCAT(_, kps_type), variable), \
      offsetof(loco_t::kps_t::kps_type, variable) \
    );\
    static_assert(std::is_same_v<decltype(loco_t::kps_t::kps_type::variable), loco_t::return_type>, "possibly unwanted behaviour"); \
    return *(loco_t::return_type*)&KeyPack[o];\
  }()

  static void generate_commands(loco_t* loco) {
#if defined(fan_gui)
    loco->console.open();

    loco->console.commands.add("echo", [](const fan::commands_t::arg_t& args) {
      fan::commands_t::output_t out;
      out.text = fan::append_args(args) + "\n";
      out.highlight = fan::graphics::highlight_e::info;
      gloco->console.commands.output_cb(out);
      }).description = "prints something - usage echo [args]";

    loco->console.commands.add("help", [](const fan::commands_t::arg_t& args) {
      if (args.empty()) {
        fan::commands_t::output_t out;
        out.highlight = fan::graphics::highlight_e::info;
        std::string out_str;
        out_str += "{\n";
        for (const auto& i : gloco->console.commands.func_table) {
          out_str += "\t" + i.first + ",\n";
        }
        out_str += "}\n";
        out.text = out_str;
        gloco->console.commands.output_cb(out);
        return;
      }
      else if (args.size() == 1) {
        auto found = gloco->console.commands.func_table.find(args[0]);
        if (found == gloco->console.commands.func_table.end()) {
          gloco->console.commands.print_command_not_found(args[0]);
          return;
        }
        fan::commands_t::output_t out;
        out.text = found->second.description + "\n";
        out.highlight = fan::graphics::highlight_e::info;
        gloco->console.commands.output_cb(out);
      }
      else {
        gloco->console.commands.print_invalid_arg_count();
      }
      }).description = "get info about specific command - usage help command";

    loco->console.commands.add("list", [](const fan::commands_t::arg_t& args) {
      std::string out_str;
      for (const auto& i : gloco->console.commands.func_table) {
        out_str += i.first + "\n";
      }

      fan::commands_t::output_t out;
      out.text = out_str;
      out.highlight = fan::graphics::highlight_e::info;

      gloco->console.commands.output_cb(out);
      }).description = "lists all commands - usage list";

    loco->console.commands.add("alias", [](const fan::commands_t::arg_t& args) {
      if (args.size() < 2 || args[1].empty()) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      if (gloco->console.commands.insert_to_command_chain(args)) {
        return;
      }
      gloco->console.commands.func_table[args[0]] = gloco->console.commands.func_table[args[1]];
      }).description = "can create alias commands - usage alias [cmd name] [cmd]";


    loco->console.commands.add("show_fps", [](const fan::commands_t::arg_t& args) {
      if (args.size() != 1) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      gloco->show_fps = std::stoi(args[0]);
      }).description = "toggles fps - usage show_fps [value]";

    loco->console.commands.add("quit", [](const fan::commands_t::arg_t& args) {
      exit(0);
      }).description = "quits program - usage quit";

    loco->console.commands.add("clear", [](const fan::commands_t::arg_t& args) {
      gloco->console.output_buffer.clear();
      gloco->console.editor.SetText("");
      }).description = "clears output buffer - usage clear";

    loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
      if (args.size() != 1) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
      }).description = "sets gamma for postprocessing shader";

    loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
      if (args.size() != 1) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
      }).description = "sets gamma for postprocessing shader";

    loco->console.commands.add("set_exposure", [](const fan::commands_t::arg_t& args) {
      if (args.size() != 1) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "exposure", std::stof(args[0]));
      }).description = "sets exposure for postprocessing shader";

    loco->console.commands.add("set_bloom_strength", [](const fan::commands_t::arg_t& args) {
      if (args.size() != 1) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "bloom_strength", std::stof(args[0]));
      }).description = "sets bloom strength for postprocessing shader";

    loco->console.commands.add("set_vsync", [](const fan::commands_t::arg_t& args) {
      if (args.size() != 1) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      gloco->set_vsync(std::stoi(args[0]));
      }).description = "sets vsync";

    loco->console.commands.add("set_target_fps", [](const fan::commands_t::arg_t& args) {
      if (args.size() != 1) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      gloco->set_target_fps(std::stoi(args[0]));
      }).description = "sets target fps";

    loco->console.commands.add("debug_memory", [loco, nr = fan::console_t::frame_cb_t::nr_t()](const fan::commands_t::arg_t& args) mutable {
      if (args.size() != 1) {
        loco->console.commands.print_invalid_arg_count();
        return;
      }
      if (nr.iic() && std::stoi(args[0])) {
        nr = loco->console.push_frame_process([] {
          ImGui::SetNextWindowBgAlpha(0.9f);
          static int init = 0;
          ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoFocusOnAppearing;
          if (init == 0) {
            ImGui::SetNextWindowSize(fan::vec2(600, 300));
            //window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
            init = 1;
          }
          ImGui::Begin("fan_memory_dbg_wnd", 0, window_flags);
          fan::graphics::gui::render_allocations_plot();
          ImGui::End();
          });
      }
      else if (!nr.iic() && !std::stoi(args[0])) {
        loco->console.erase_frame_process(nr);
      }
      }).description = "opens memory debug window";

    /*loco->console.commands.add("console_transparency", [](const fan::commands_t::arg_t& args) {
      if (args.size() != 1) {
        gloco->console.commands.print_invalid_arg_count();
        return;
      }
      gloco->console.transparency = std::stoull(args[0]);
      for (int i = 0; i < 21; ++i) {
        (gloco->console.editor.GetPalette().data() + i = gloco->console.transparency;
      }
      }).description = "";*/

#endif
  }

  // -1 no reload, opengl = 0 etc
  uint8_t reload_renderer_to = -1;
#if defined(fan_gui)
  void load_fonts(auto& fonts, ImGuiIO& io, const std::string& name, f32_t font_size = 4) {
    for (std::size_t i = 0; i < std::size(fonts); ++i) {
      fonts[i] = io.Fonts->AddFontFromFileTTF(name.c_str(), (int)(font_size * (1 << i)) * 1.5);

      if (fonts[i] == nullptr) {
        fan::throw_error(std::string("failed to load font:") + name);
      }
    }
    io.Fonts->Build();
  }

#if defined(fan_vulkan)
  // todo move to vulkan context
  static void check_vk_result(VkResult err) {
    if (err != VK_SUCCESS) {
      fan::print("vkerr", (int)err);
    }
  }
#endif

  void init_imgui() {
    ImGui::CreateContext();
    ImPlot::CreateContext();
    auto& input_map = ImPlot::GetInputMap();
    input_map.Pan = ImGuiMouseButton_Middle;

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ///    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
      style.WindowRounding = 0.;
    }
    style.FrameRounding = 5.f;
    style.FramePadding = ImVec2(12.f, 5.f);
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;

    imgui_themes::dark();

    if (window.renderer == renderer_t::opengl) {
      glfwMakeContextCurrent(window);
      ImGui_ImplGlfw_InitForOpenGL(window, true);
      const char* glsl_version = "#version 120";
      ImGui_ImplOpenGL3_Init(glsl_version);
    }
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      ImGui_ImplGlfw_InitForVulkan(window, true);
      ImGui_ImplVulkan_InitInfo init_info = {};
      init_info.Instance = context.vk.instance;
      init_info.PhysicalDevice = context.vk.physical_device;
      init_info.Device = context.vk.device;
      init_info.QueueFamily = context.vk.queue_family;
      init_info.Queue = context.vk.graphics_queue;
      init_info.DescriptorPool = context.vk.descriptor_pool.m_descriptor_pool;
      init_info.RenderPass = context.vk.MainWindowData.RenderPass;
      init_info.Subpass = 0;
      init_info.MinImageCount = context.vk.MinImageCount;
      init_info.ImageCount = context.vk.MainWindowData.ImageCount;
      init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
      init_info.CheckVkResultFn = check_vk_result;

      ImGui_ImplVulkan_Init(&init_info);
    }
#endif

    load_fonts(fonts, io, "fonts/SourceCodePro-Regular.ttf", 4.f);
    load_fonts(fonts_bold, io, "fonts/SourceCodePro-Bold.ttf", 4.f);

    io.FontDefault = fonts[2];

    input_action.add(fan::key_escape, "open_settings");
  }
  void destroy_imgui() {
    if (window.renderer == renderer_t::opengl) {
      ImGui_ImplOpenGL3_Shutdown();
    }
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      vkDeviceWaitIdle(context.vk.device);
      ImGui_ImplVulkan_Shutdown();
    }
#endif
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    ImPlot::DestroyContext();
#if defined(fan_vulkan)
    if (window.renderer == renderer_t::vulkan) {
      context.vk.imgui_close();
    }
#endif

  }
  bool enable_overlay = true;
#endif
  void init_framebuffer() {
    if (window.renderer == renderer_t::opengl) {
      gl.init_framebuffer();
    }
  }

  loco_t() : loco_t(properties_t()) {

  }
  loco_t(const properties_t& p) {
    if (fan::init_manager_t::initialized() == false) {
      fan::init_manager_t::initialize();
    }
    fan::graphics::engine_init_cbs.Open();
    render_shapes_top = p.render_shapes_top;
    window.renderer = p.renderer;
    shape_functions.resize(shape_type_t::last);
    if (window.renderer == renderer_t::opengl) {
      new (&context.gl) fan::opengl::context_t();
      context_functions = fan::graphics::get_gl_context_functions();
      gl.open();
    }

    window.open(p.window_size, fan::window_t::default_window_name, p.window_flags);
    gloco = this;


#if fan_debug >= fan_debug_high && !defined(fan_vulkan)
    if (window.renderer == renderer_t::vulkan) {
      fan::throw_error("trying to use vulkan renderer, but fan_vulkan build flag is disabled");
    }
#endif

#if defined(fan_vulkan)
    if (window.renderer == renderer_t::vulkan) {
      context_functions = fan::graphics::get_vk_context_functions();
      new (&context.vk) fan::vulkan::context_t();
      //context.vk.enable_clear = !render_shapes_top;
      context.vk.shapes_top = render_shapes_top;
      context.vk.open(window);
    }
#endif

    start_time = fan::time::clock::now();

    set_vsync(false); // using libuv
    //fan::print("less pain", this, (void*)&lighting, (void*)((uint8_t*)&lighting - (uint8_t*)this), sizeof(*this), lighting.ambient);
    if (window.renderer == renderer_t::opengl) {
      glfwMakeContextCurrent(window);

#if fan_debug >= fan_debug_high
      get_context().gl.set_error_callback();
#endif

      gl.initialize_fb_vaos();
    }

#if defined(loco_vfi)
    window.add_buttons_callback([this](const fan::window_t::mouse_buttons_cb_data_t& d) {
      fan::vec2 window_size = window.get_size();
      vfi.feed_mouse_button(d.button, d.state);
      });

    window.add_keys_callback([&](const fan::window_t::keyboard_keys_cb_data_t& d) {
      vfi.feed_keyboard(d.key, d.state);
      });

    window.add_mouse_move_callback([&](const fan::window_t::mouse_move_cb_data_t& d) {
      vfi.feed_mouse_move(d.position);
      });

    window.add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
      vfi.feed_text(d.character);
      });
#endif

    default_texture = create_missing_texture();

    shaper.Open();

    {

      // filler
      shaper.AddKey(Key_e::light, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
      shaper.AddKey(Key_e::light_end, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
      shaper.AddKey(Key_e::depth, sizeof(loco_t::depth_t), shaper_t::KeyBitOrderLow);
      shaper.AddKey(Key_e::blending, sizeof(loco_t::blending_t), shaper_t::KeyBitOrderLow);
      shaper.AddKey(Key_e::image, sizeof(loco_t::image_t), shaper_t::KeyBitOrderLow);
      shaper.AddKey(Key_e::viewport, sizeof(loco_t::viewport_t), shaper_t::KeyBitOrderAny);
      shaper.AddKey(Key_e::camera, sizeof(loco_t::camera_t), shaper_t::KeyBitOrderAny);
      shaper.AddKey(Key_e::ShapeType, sizeof(shaper_t::ShapeTypeIndex_t), shaper_t::KeyBitOrderAny);
      shaper.AddKey(Key_e::filler, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
      shaper.AddKey(Key_e::draw_mode, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
      shaper.AddKey(Key_e::vertex_count, sizeof(uint32_t), shaper_t::KeyBitOrderAny);

      //gloco->shaper.AddKey(Key_e::image4, sizeof(loco_t::image_t) * 4, shaper_t::KeyBitOrderLow);
    }
    // order of open needs to be same with shapes enum

    {
      fan::vec2 window_size = window.get_size();
      {
        orthographic_camera.camera = open_camera(
          fan::vec2(0, window_size.x),
          fan::vec2(0, window_size.y)
        );
        orthographic_camera.viewport = open_viewport(
          fan::vec2(0, 0),
          window_size
        );
      }
      {
        perspective_camera.camera = open_camera_perspective();
        perspective_camera.viewport = open_viewport(
          fan::vec2(0, 0),
          window_size
        );
      }
    }

    if (window.renderer == renderer_t::opengl) {
      gl.shapes_open();
    }
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      vk.shapes_open();
    }
#endif


#if defined(fan_physics)
    fan::graphics::open_bcol();
#endif

#if defined(fan_gui)
    init_imgui();
    generate_commands(this);
#endif

    bool windowed = true;
    // free this xd
    gloco->window.add_keys_callback(
      [windowed](const fan::window_t::keyboard_keys_cb_data_t& data) mutable {
        if (data.key == fan::key_enter && data.state == fan::keyboard_state::press && gloco->window.key_pressed(fan::key_left_alt)) {
          windowed = !windowed;
          gloco->window.set_display_mode(windowed ? fan::window_t::mode::windowed : fan::window_t::mode::borderless);
        }
      }
    );
#if defined(fan_gui)
    settings_menu.open();
#endif

    auto it = fan::graphics::engine_init_cbs.GetNodeFirst();
    while (it != fan::graphics::engine_init_cbs.dst) {
      fan::graphics::engine_init_cbs.StartSafeNext(it);
      fan::graphics::engine_init_cbs[it](this);
      it = fan::graphics::engine_init_cbs.EndSafeNext();
    }

#if defined(loco_audio)

    if (system_audio.Open() != 0) {
      fan::throw_error("failed to open fan audio");
    }
    audio.bind(&system_audio);
    audio.Open(&piece_hover, "audio/hover.sac", 0);
    audio.Open(&piece_click, "audio/click.sac", 0);

#endif
  }
  ~loco_t() {
    destroy();
  }

  void destroy() {
    fan::graphics::engine_init_cbs.Close();
    if (window == nullptr) {
      return;
    }
#if defined(fan_gui)
    console.commands.func_table.clear();
    console.close();
#endif
#if defined(fan_physics)
    fan::graphics::close_bcol();
#endif
#if defined(fan_vulkan)
    if (window.renderer == loco_t::renderer_t::vulkan) {
      vkDeviceWaitIdle(context.vk.device);
      vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
      vk.d_attachments.close(context.vk);
      vk.post_process.close(context.vk);
    }
#endif
    shaper.Close();
#if defined(fan_gui)
    destroy_imgui();
#endif
    window.close();
#if defined(loco_audio)
    audio.unbind();
    system_audio.Close();
#endif
  }
  void close() {
    destroy();
  }

  // for renderer switch
  // input loco_t::renderer_t::
  void switch_renderer(uint8_t renderer) {
    std::vector<std::string> image_paths;
    fan::vec2 window_size = window.get_size();
    fan::vec2 window_position = window.get_position();
    uint64_t flags = window.flags;

    {// close
#if defined(fan_vulkan)
      if (window.renderer == loco_t::renderer_t::vulkan) {
        // todo wrap to vk.
        vkDeviceWaitIdle(context.vk.device);
        vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
        vk.d_attachments.close(context.vk);
        vk.post_process.close(context.vk);
        //CLOOOOSEEE POSTPROCESSS IMAGEEES
      }
      else
#endif
        if (window.renderer == loco_t::renderer_t::opengl) {
          for (auto& st : shaper.ShapeTypes) {
#if defined(fan_vulkan)
            if (std::holds_alternative<loco_t::shaper_t::ShapeType_t::vk_t>(st.renderer)) {
              auto& str = std::get<loco_t::shaper_t::ShapeType_t::vk_t>(st.renderer);
              str.shape_data.close(context.vk);
              str.pipeline.close(context.vk);
            }
#endif
            //st.BlockList.Close();
          }
          glDeleteVertexArrays(1, &gl.fb_vao);
          glDeleteBuffers(1, &gl.fb_vbo);
          context.gl.internal_close();
        }
#if defined(fan_gui)
      destroy_imgui();
#endif
      window.close();
    }
    {// reopen
      window.renderer = reload_renderer_to; // i dont like this {window.renderer = ...}
      if (window.renderer == renderer_t::opengl) {
        context_functions = fan::graphics::get_gl_context_functions();
        new (&context.gl) fan::opengl::context_t();
        gl.open();
      }

      window.open(window_size, fan::window_t::default_window_name, flags | fan::window_t::flags::no_visible);
      window.set_position(window_position);
      window.set_position(window_position);
      glfwShowWindow(window);
      window.flags = flags;
#if defined(fan_vulkan)
      if (window.renderer == renderer_t::vulkan) {
        new (&context.vk) fan::vulkan::context_t();
        context_functions = fan::graphics::get_vk_context_functions();
        context.vk.open(window);
      }
#endif
    }
    {// reload
      {
        {
          fan::graphics::camera_list_t::nrtra_t nrtra;
          fan::graphics::camera_nr_t nr;
          nrtra.Open(&__fan_internal_camera_list, &nr);
          while (nrtra.Loop(&__fan_internal_camera_list, &nr)) {
            auto& cam = __fan_internal_camera_list[nr];
            camera_set_ortho(
              nr,
              fan::vec2(cam.coordinates.left, cam.coordinates.right),
              fan::vec2(cam.coordinates.up, cam.coordinates.down)
            );
          }
          nrtra.Close(&__fan_internal_camera_list);
        }
        {
          fan::graphics::viewport_list_t::nrtra_t nrtra;
          fan::graphics::viewport_nr_t nr;
          nrtra.Open(&__fan_internal_viewport_list, &nr);
          while (nrtra.Loop(&__fan_internal_viewport_list, &nr)) {
            auto& viewport = __fan_internal_viewport_list[nr];
            viewport_set(
              nr,
              viewport.viewport_position,
              viewport.viewport_size,
              window.get_size()
            );
          }
          nrtra.Close(&__fan_internal_viewport_list);
        }
      }

      {
        {
          {
            fan::graphics::image_list_t::nrtra_t nrtra;
            fan::graphics::image_nr_t nr;
            nrtra.Open(&image_list, &nr);
            while (nrtra.Loop(&image_list, &nr)) {

              if (window.renderer == renderer_t::opengl) {
                // illegal
                image_list[nr].internal = new fan::opengl::context_t::image_t;
                fan_opengl_call(glGenTextures(1, &((fan::opengl::context_t::image_t*)context_functions.image_get(&context.gl, nr))->texture_id));
              }
#if defined(fan_vulkan)
              else if (window.renderer == renderer_t::vulkan) {
                // illegal
                image_list[nr].internal = new fan::vulkan::context_t::image_t;
              }
#endif
              // handle blur?
              auto image_path = image_list[nr].image_path;
              if (image_path.empty()) {
                fan::image::image_info_t info;
                info.data = (void*)fan::image::missing_texture_pixels;
                info.size = 2;
                info.channels = 4;
                fan::graphics::image_load_properties_t lp;
                lp.min_filter = fan::graphics::image_filter::nearest;
                lp.mag_filter = fan::graphics::image_filter::nearest;
                lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
                image_reload(nr, info, lp);
              }
              else {
                image_reload(nr, image_list[nr].image_path);
              }
            }
            nrtra.Close(&image_list);
          }
          {
            fan::graphics::shader_list_t::nrtra_t nrtra;
            fan::graphics::shader_nr_t nr;
            nrtra.Open(&__fan_internal_shader_list, &nr);
            while (nrtra.Loop(&__fan_internal_shader_list, &nr)) {
              if (window.renderer == renderer_t::opengl) {
                __fan_internal_shader_list[nr].internal = new fan::opengl::context_t::shader_t;
              }
#if defined(fan_vulkan)
              else if (window.renderer == renderer_t::vulkan) {
                __fan_internal_shader_list[nr].internal = new fan::vulkan::context_t::shader_t;
              }
#endif
            }
            nrtra.Close(&__fan_internal_shader_list);
          }
        }
        fan::image::image_info_t info;
        info.data = (void*)fan::image::missing_texture_pixels;
        info.size = 2;
        info.channels = 4;
        fan::graphics::image_load_properties_t lp;
        lp.min_filter = fan::graphics::image_filter::nearest;
        lp.mag_filter = fan::graphics::image_filter::nearest;
        lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
        image_reload(default_texture, info, lp);
      }
      shape_functions.clear();
      if (window.renderer == renderer_t::opengl) {
        gl.shapes_open();
        gl.initialize_fb_vaos();
      }
#if defined(fan_vulkan)
      else if (window.renderer == renderer_t::vulkan) {
        vk.shapes_open();
      }
#endif
#if defined(fan_gui)
      init_imgui();
      settings_menu.set_settings_theme();
#endif

      shaper._BlockListCapacityChange(shape_type_t::rectangle, 0, 1);
      shaper._BlockListCapacityChange(shape_type_t::sprite, 0, 1);

#if defined(loco_audio)
      if (system_audio.Open() != 0) {
        fan::throw_error("failed to open fan audio");
      }
      audio.bind(&system_audio);
#endif
    }
    reload_renderer_to = -1;
  }

  void draw_shapes() {
    if (window.renderer == renderer_t::opengl) {
      gl.draw_shapes();
    }
#if defined(fan_vulkan)
    else
      if (window.renderer == renderer_t::vulkan) {
        vk.draw_shapes();
      }
#endif
  }
  void process_shapes() {

#if defined(fan_vulkan)
    if (window.renderer == renderer_t::vulkan) {
      if (render_shapes_top == true) {
        vk.begin_render_pass();
      }
    }
#endif
    for (const auto& i : m_pre_draw) {
      i();
    }

    draw_shapes();

    for (const auto& i : m_post_draw) {
      i();
    }

#if defined(fan_vulkan)
    if (window.renderer == renderer_t::vulkan) {
      auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
      if (vk.image_error != (decltype(vk.image_error))-0xfff) {
        vkCmdNextSubpass(cmd_buffer, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.post_process);
        vkCmdBindDescriptorSets(
          cmd_buffer,
          VK_PIPELINE_BIND_POINT_GRAPHICS,
          vk.post_process.m_layout,
          0,
          1,
          vk.d_attachments.m_descriptor_set,
          0,
          nullptr
        );

        // render post process
        vkCmdDraw(cmd_buffer, 6, 1, 0, 0);
      }
      if (render_shapes_top == true) {
        vkCmdEndRenderPass(cmd_buffer);
      }
    }
#endif
  }
  void process_gui() {
#if defined(fan_gui)
    fan::graphics::gui::process_loop();

    if (ImGui::IsKeyPressed(ImGuiKey_F3, false)) {
      render_console = !render_console;

      // force focus xd
      console.input.InsertText("a");
      console.input.SetText("");
      console.init_focus = true;
      console.input.IsFocused() = false;
    }
    if (render_console) {
      console.render();
    }
    if (input_action.is_active("open_settings")) {
      render_settings_menu = !render_settings_menu;
    }
    if (render_settings_menu) {
      settings_menu.render();
    }

    if (show_fps) {
      ImGui::SetNextWindowBgAlpha(0.9f);
      static int init = 0;
      ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoFocusOnAppearing;
      if (init == 0) {
        window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
        init = 1;
      }
      ImGui::Begin("Performance window", 0, window_flags);

      static constexpr int buffer_size = 128;
      static std::array<float, buffer_size> samples = { 0 };
      static int insert_index = 0;
      static float running_sum = 0.0f;
      static float running_min = std::numeric_limits<float>::max();
      static float running_max = std::numeric_limits<float>::min();
      static fan::time::clock refresh_speed{ (uint64_t)0.05e9, true };

      if (refresh_speed.finished()) {
        float old_value = samples[insert_index];
        for (int i = 0; i < buffer_size - 1; ++i) {
          samples[i] = samples[i + 1];
        }

        samples[buffer_size - 1] = delta_time;

        running_sum += samples[buffer_size - 1] - samples[0];

        if (delta_time <= running_min) {
          running_min = delta_time;
        }
        else if (delta_time >= running_max) {
          running_max = delta_time;
        }

        insert_index = (insert_index + 1) % buffer_size;
        refresh_speed.restart();
      }

      float average_frame_time_ms = running_sum / buffer_size;
      float average_fps = 1.0f / average_frame_time_ms;
      float lowest_fps = 1.0f / running_max;
      float highest_fps = 1.0f / running_min;

      ImGui::Text("fps: %d", (int)(1.f / delta_time));
      ImGui::Text("Average Frame Time: %.4f ms", average_frame_time_ms);
      ImGui::Text("Lowest Frame Time: %.4f ms", running_min);
      ImGui::Text("Highest Frame Time: %.4f ms", running_max);
      ImGui::Text("Average fps: %.4f", average_fps);
      ImGui::Text("Lowest fps: %.4f", lowest_fps);
      ImGui::Text("Highest fps: %.4f", highest_fps);
      if (ImGui::Button("Reset lowest&highest")) {
        running_min = std::numeric_limits<float>::max();
        running_max = std::numeric_limits<float>::min();
      }

      if (ImPlot::BeginPlot("frame time", ImVec2(-1, 0), ImPlotFlags_NoFrame | ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("Frame Index", "FPS", ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_AutoFit); //
        ImPlot::PlotLine("FPS", samples.data(), buffer_size, 1.0, 0.0);
        ImPlot::EndPlot();
      }
      ImGui::Text("Current Frame Time: %.4f ms", delta_time);
      ImGui::End();
    }

#if defined(loco_framebuffer)

#endif

    ImGui::Render();


    if (window.renderer == renderer_t::opengl) {

      //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      //glClear(GL_COLOR_BUFFER_BIT);

      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
      // did draw
      if (vk.image_error == (decltype(vk.image_error))-0xfff) {
        vk.image_error = VK_SUCCESS;
      }
      if (render_shapes_top == false) {
        vkCmdEndRenderPass(cmd_buffer);
      }

      ImDrawData* draw_data = ImGui::GetDrawData();
      const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
      if (!is_minimized) {
        context.vk.ImGuiFrameRender(vk.image_error, clear_color);
      }
    }
#endif
#endif
  }
  void process_frame() {

    if (window.renderer == renderer_t::opengl) {
      gl.begin_process_frame();
    }

    {
      auto it = m_update_callback.GetNodeFirst();
      while (it != m_update_callback.dst) {
        m_update_callback.StartSafeNext(it);
        m_update_callback[it](this);
        it = m_update_callback.EndSafeNext();
      }
    }

#if defined(fan_physics)
    {
      auto it = shape_physics_update_cbs.GetNodeFirst();
      while (it != shape_physics_update_cbs.dst) {
        shape_physics_update_cbs.StartSafeNext(it);
        ((shape_physics_update_cb)shape_physics_update_cbs[it].cb)(shape_physics_update_cbs[it]);
        it = shape_physics_update_cbs.EndSafeNext();
      }
    }
#endif

    for (const auto& i : single_queue) {
      i();
    }

    single_queue.clear();

#if defined(fan_gui)
    ImGui::End();
#endif

    shaper.ProcessBlockEditQueue();

#if defined(fan_vulkan)
    if (window.renderer == renderer_t::vulkan) {
      vk.begin_draw();
    }
#endif

    viewport_set(0, window.get_size(), window.get_size());

    if (render_shapes_top == false) {
      process_shapes();
      process_gui();
    }
    else {
      process_gui();
      process_shapes();
    }

    if (window.renderer == renderer_t::opengl) {
      glfwSwapBuffers(window);
    }
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
#if !defined(fan_gui)
      auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
      // did draw
      vkCmdNextSubpass(cmd_buffer, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdEndRenderPass(cmd_buffer);
#endif
      VkResult err = context.vk.end_render();
      context.vk.recreate_swap_chain(&window, err);
    }
#endif
  }

  bool should_close() {
    if (window == nullptr) {
      return true;
    }
    return glfwWindowShouldClose(window);
  }
  void should_close(int flag);

  bool process_loop(const std::function<void()>& lambda = [] {}) {

#if defined(fan_gui)
    if (reload_renderer_to != (decltype(reload_renderer_to))-1) {
      switch_renderer(reload_renderer_to);
    }

    if (window.renderer == renderer_t::opengl) {
      ImGui_ImplOpenGL3_NewFrame();
    }
#if defined(fan_vulkan)
    else if (window.renderer == renderer_t::vulkan) {
      ImGui_ImplVulkan_NewFrame();
    }
#endif

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    auto& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
    ImGui::PopStyleColor(2);

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(window.get_size());

    int flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiDockNodeFlags_NoDockingSplit | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBringToFrontOnFocus;

    if (!enable_overlay) {
      flags |= ImGuiWindowFlags_NoNav;
    }

    ImGui::Begin("##global_renderer", 0, flags);
#endif

    lambda();

    // user can terminate from main loop
    if (should_close()) {
      return 1;
    }//

    process_frame();
    window.handle_events();
    
    delta_time = window.m_delta_time;

    // window can also be closed from window cb
    if (should_close()) {
      return 1;
    }//

    return 0;
  }
  void loop(const std::function<void()>& lambda = []{}) {
    main_loop = lambda;
  g_loop:
    double delay = std::round(1.0 / target_fps * 1000.0);

    if (!timer_init) {
      uv_timer_init(fan::event::event_loop, &timer_handle);
      timer_init = true;
    }
    if (!idle_init) {
      uv_idle_init(fan::event::event_loop, &idle_handle);
      idle_init = true;
    }

    timer_handle.data = this;
    idle_handle.data = this;

    if (target_fps > 0) {
      start_timer();
    }
    else {
      start_idle();
    }

    uv_run(fan::event::event_loop, UV_RUN_DEFAULT);
    if (should_close() == false) {
      goto g_loop;
    }
  }

  loco_t::camera_t open_camera(const fan::vec2& x, const fan::vec2& y) {
    loco_t::camera_t camera = camera_create();
    camera_set_ortho(camera, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
    return camera;
  }
  loco_t::camera_t open_camera_perspective(f32_t fov = 90.0f) {
    loco_t::camera_t camera = camera_create();
    camera_set_perspective(camera, fov, window.get_size());
    return camera;
  }

  loco_t::viewport_t open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    loco_t::viewport_t viewport = viewport_create();
    viewport_set(viewport, viewport_position, viewport_size, window.get_size());
    return viewport;
  }

  void set_viewport(loco_t::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    viewport_set(viewport, viewport_position, viewport_size, window.get_size());
  }

  // for checking whether you set depth or no
  //struct position3_t : public fan::vec3 {
  //  using fan::vec3::vec3;
  //  using fan::vec3::operator=;
  //  position3_t& operator=(const position3_t& p) {
  //    fan::vec3::operator=(p);
  //    return *this;
  //  }
  //};


  //
  fan::vec2 transform_matrix(const fan::vec2& position) {
    fan::vec2 window_size = window.get_size();
    // not custom ortho friendly - made for -1 1
    return position / window_size * 2 - 1;
  }

  fan::vec2 screen_to_ndc(const fan::vec2& screen_pos) {
    fan::vec2 window_size = window.get_size();
    return screen_pos / window_size * 2 - 1;
  }

  fan::vec2 ndc_to_screen(const fan::vec2& ndc_position) {
    fan::vec2 window_size = window.get_size();
    fan::vec2 normalized_position = (ndc_position + 1) / 2;
    return normalized_position * window_size;
  }
  //

  void set_vsync(bool flag) {
    vsync = flag;
    // vulkan vsync is enabled by presentation mode in swap chain
    if (window.renderer == renderer_t::opengl) {
      context.gl.set_vsync(&window, flag);
    }
  }
  void start_timer() {
    double delay;
    if (target_fps <= 0) {
      delay = 0;
    }
    else {
      delay = std::round(1.0 / target_fps * 1000.0);
    }
    if (delay > 0) {
      uv_timer_start(&timer_handle, [](uv_timer_t* handle) {
        loco_t* loco = static_cast<loco_t*>(handle->data);
        if (loco->process_loop(loco->main_loop)) {
          uv_timer_stop(handle);
          uv_stop(fan::event::event_loop);
        }
        }, 0, delay);
    }
  }
  void start_idle() {
    uv_idle_start(&idle_handle, [](uv_idle_t* handle) {
      loco_t* loco = static_cast<loco_t*>(handle->data);
      if (loco->process_loop(loco->main_loop)) {
        uv_idle_stop(handle);
        uv_stop(fan::event::event_loop);
      }
      });
  }
  void update_timer_interval() {
    double delay;
    if (target_fps <= 0) {
      delay = 0;
    }
    else {
      delay = std::round(1.0 / target_fps * 1000.0);
    }
    if (delay > 0) {
      if (timer_enabled == false) {
        start_timer();
        timer_enabled = true;
      }
      uv_idle_stop(&idle_handle);
      uv_timer_set_repeat(&timer_handle, delay);
      uv_timer_again(&timer_handle);
    }
    else {
      uv_timer_stop(&timer_handle);
      if (!idle_init) {
        uv_idle_init(fan::event::event_loop, &idle_handle);
        idle_handle.data = this;
        idle_init = true;
      }
      start_idle();
    }
  }
  void set_target_fps(int32_t new_target_fps) {
    target_fps = new_target_fps;
    update_timer_interval();
  }

  fan::graphics::context_t& get_context() {
    return context;
  }

  struct camera_impl_t {
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

    void add(const int* keys, std::size_t count, std::string_view action_name) {
      action_data_t action_data;
      action_data.count = (uint8_t)count;
      std::memcpy(action_data.keys, keys, sizeof(int) * count);
      input_actions[action_name] = action_data;
    }
    void add(int key, std::string_view action_name) {
      add(&key, 1, action_name);
    }
    void add(std::initializer_list<int> keys, std::string_view action_name) {
      add(keys.begin(), keys.size(), action_name);
    }

    void edit(int key, std::string_view action_name) {
      auto found = input_actions.find(action_name);
      if (found == input_actions.end()) {
        fan::throw_error("trying to modify non existing action");
      }
      std::memset(found->second.keys, 0, sizeof(found->second.keys));
      found->second.keys[0] = key;
      found->second.count = 1;
      found->second.combo_count = 0;
    }

    void add_keycombo(std::initializer_list<int> keys, std::string_view action_name) {
      action_data_t action_data;
      action_data.combo_count = (uint8_t)keys.size();
      std::memcpy(action_data.key_combos, keys.begin(), sizeof(int) * action_data.combo_count);
      input_actions[action_name] = action_data;
    }

    bool is_active(std::string_view action_name, int pstate = loco_t::input_action_t::press) {
      auto found = input_actions.find(action_name);
      if (found != input_actions.end()) {
        action_data_t& action_data = found->second;

        if (action_data.combo_count) {
          int state = none;
          for (int i = 0; i < action_data.combo_count; ++i) {
            int s = gloco->window.key_state(action_data.key_combos[i]);
            if (s == none) {
              return none == loco_t::input_action_t::press;
            }
            if (state == input_action_t::press && s == input_action_t::repeat) {
              state = 1;
            }
            if (state == input_action_t::press_or_repeat) {
              if (state == input_action_t::press && s == input_action_t::repeat) {
              }
            }
            else {
              state = s;
            }
          }
          if (pstate == input_action_t::press_or_repeat) {
            return state == input_action_t::press ||
              state == input_action_t::repeat;
          }
          return state == pstate;
        }
        else if (action_data.count) {
          int state = none;
          for (int i = 0; i < action_data.count; ++i) {
            int s = gloco->window.key_state(action_data.keys[i]);
            if (s != none) {
              state = s;
            }
          }
          if (pstate == input_action_t::press_or_repeat) {
            return state == input_action_t::press ||
              state == input_action_t::repeat;
          }
          //fan::print(state, pstate, state == pstate);
          return state == pstate;
        }
      }
      return none == pstate;
    }
    bool is_action_clicked(std::string_view action_name) {
      return is_active(action_name);
    }
    bool is_action_down(std::string_view action_name) {
      return is_active(action_name, press_or_repeat);
    }
    bool exists(std::string_view action_name) {
      return input_actions.find(action_name) != input_actions.end();
    }
    void insert_or_assign(int key, std::string_view action_name) {
      action_data_t action_data;
      action_data.count = (uint8_t)1;
      std::memcpy(action_data.keys, &key, sizeof(int) * 1);
      input_actions.insert_or_assign(action_name, action_data);
    }

    std::unordered_map<std::string_view, action_data_t> input_actions;
  }input_action;

  static fan::vec2 transform_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

    auto v = gloco->viewport_get(viewport);
    auto c = gloco->camera_get(camera);

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

protected:
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix update_callback
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType std::function<void(loco_t*)>
#define BLL_set_CPP_CopyAtPointerChange 1
#include <BLL/BLL.h>
public:

  using update_callback_nr_t = update_callback_NodeReference_t;

  update_callback_t m_update_callback;

  std::vector<std::function<void()>> single_queue;

  image_t default_texture;

  camera_impl_t orthographic_camera;
  camera_impl_t perspective_camera;

  fan::window_t window;
  bool idle_init = false;
  uv_idle_t idle_handle;
  bool timer_init = false;
  uv_timer_t timer_handle{};

  int32_t target_fps = 165; // must be changed from function
  bool timer_enabled = target_fps > 0;
  bool vsync = false;

  std::function<void()> main_loop; // bad, but forced

  f64_t delta_time = window.m_delta_time;

#if defined(fan_gui)
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix gui_draw_cb
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType std::function<void()>
#include <BLL/BLL.h>

  gui_draw_cb_t gui_draw_cb;
#endif

  using push_back_cb = loco_t::shape_t(*)(void*);
  using set_position2_cb = void (*)(loco_t::shape_t*, const fan::vec2&);
  // depth
  using set_position3_cb = void (*)(loco_t::shape_t*, const fan::vec3&);
  using set_size_cb = void (*)(loco_t::shape_t*, const fan::vec2&);
  using set_size3_cb = void (*)(loco_t::shape_t*, const fan::vec3&);

  using get_position_cb = fan::vec3(*)(loco_t::shape_t*);
  using get_size_cb = fan::vec2(*)(loco_t::shape_t*);
  using get_size3_cb = fan::vec3(*)(loco_t::shape_t*);

  using set_rotation_point_cb = void (*)(loco_t::shape_t*, const fan::vec2&);
  using get_rotation_point_cb = fan::vec2(*)(loco_t::shape_t*);

  using set_color_cb = void (*)(loco_t::shape_t*, const fan::color&);
  using get_color_cb = fan::color(*)(loco_t::shape_t*);

  using set_angle_cb = void (*)(loco_t::shape_t*, const fan::vec3&);
  using get_angle_cb = fan::vec3(*)(loco_t::shape_t*);

  using get_tc_position_cb = fan::vec2(*)(loco_t::shape_t*);
  using set_tc_position_cb = void (*)(loco_t::shape_t*, const fan::vec2&);

  using get_tc_size_cb = fan::vec2(*)(loco_t::shape_t*);
  using set_tc_size_cb = void (*)(loco_t::shape_t*, const fan::vec2&);

  using load_tp_cb = bool(*)(loco_t::shape_t*, loco_t::texturepack_t::ti_t*);

  using get_grid_size_cb = fan::vec2(*)(loco_t::shape_t*);
  using set_grid_size_cb = void (*)(loco_t::shape_t*, const fan::vec2&);

  using get_camera_cb = loco_t::camera_t(*)(loco_t::shape_t*);
  using set_camera_cb = void (*)(loco_t::shape_t*, loco_t::camera_t);

  using get_viewport_cb = loco_t::viewport_t(*)(loco_t::shape_t*);
  using set_viewport_cb = void (*)(loco_t::shape_t*, loco_t::viewport_t);


  using get_image_cb = loco_t::image_t(*)(loco_t::shape_t*);
  using set_image_cb = void (*)(loco_t::shape_t*, loco_t::image_t);

  using get_image_data_cb = fan::graphics::image_data_t& (*)(loco_t::shape_t*);

  using get_parallax_factor_cb = f32_t(*)(loco_t::shape_t*);
  using set_parallax_factor_cb = void (*)(loco_t::shape_t*, f32_t);
  using get_flags_cb = uint32_t(*)(loco_t::shape_t*);
  using set_flags_cb = void(*)(loco_t::shape_t*, uint32_t);
  //
  using get_radius_cb = f32_t(*)(loco_t::shape_t*);
  using get_src_cb = fan::vec3(*)(loco_t::shape_t*);
  using get_dst_cb = fan::vec3(*)(loco_t::shape_t*);
  using get_outline_size_cb = f32_t(*)(loco_t::shape_t*);
  using get_outline_color_cb = fan::color(*)(loco_t::shape_t*);
  using set_outline_color_cb = void(*)(loco_t::shape_t*, const fan::color&);

  using reload_cb = void (*)(loco_t::shape_t*, uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter);

  using draw_cb = void (*)(uint8_t draw_range);

  using set_line_cb = void (*)(loco_t::shape_t*, const fan::vec3&, const fan::vec3&);
  using set_line3_cb = void (*)(loco_t::shape_t*, const fan::vec3&, const fan::vec3&);

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


    get_flags_cb get_flags;
    set_flags_cb set_flags;

    get_radius_cb get_radius;
    get_src_cb get_src;
    get_dst_cb get_dst;
    get_outline_size_cb get_outline_size;
    get_outline_color_cb get_outline_color;
    set_outline_color_cb set_outline_color;

    reload_cb reload;

    draw_cb draw;

    set_line_cb set_line;
    set_line3_cb set_line3;
  };

  #include <fan/graphics/shape_functions_generated.h>

  std::vector<functions_t> shape_functions;

  // needs continous buffer
  std::vector<shaper_t::BlockProperties_t> BlockProperties;

  shaper_t shaper;

#if defined(fan_physics)
  fan::physics::context_t physics_context{ {} };
  struct physics_update_data_t {
    shaper_t::ShapeID_t shape_id;
    uint64_t body_id;
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
  physics_update_cbs_t::nr_t add_physics_update(const physics_update_data_t& cb_data) {
    auto it = shape_physics_update_cbs.NewNodeLast();
    shape_physics_update_cbs[it] = (physics_update_data_t)cb_data;
    return it;
  }
  void remove_physics_update(physics_update_cbs_t::nr_t nr) {
    shape_physics_update_cbs.unlrec(nr);
  }
  physics_update_cbs_t shape_physics_update_cbs;
#endif

#pragma pack(push, 1)

  struct Key_e {
    enum : shaper_t::KeyTypeIndex_t {
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

  fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport) {
    return transform_position(get_mouse_position(), viewport, camera);
  }

  fan::vec2 get_mouse_position() {
    return window.get_mouse_position();
    //return get_mouse_position(gloco->default_camera->camera, gloco->default_camera->viewport); behaving oddly
  }

  fan::vec2 translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera) {

    auto v = gloco->viewport_get(viewport);
    fan::vec2 viewport_position = v.viewport_position;
    fan::vec2 viewport_size = v.viewport_size;

    auto c = gloco->camera_get(camera);

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

  fan::vec2 translate_position(const fan::vec2& p) {
    return translate_position(p, orthographic_camera.viewport, orthographic_camera.camera);
  }

  bool is_mouse_clicked(int button = fan::mouse_left) {
    return window.key_state(button) == (int)fan::mouse_state::press;
  }
  bool is_mouse_down(int button = fan::mouse_left) {
    int state = window.key_state(button);
    return
      state == (int)fan::mouse_state::press ||
      state == (int)fan::mouse_state::repeat;
  }
  bool is_mouse_released(int button = fan::mouse_left) {
    return window.key_state(button) == (int)fan::mouse_state::release;
  }
  fan::vec2 get_mouse_drag(int button = fan::mouse_left) {
    if (is_mouse_down(button)) {
      if (window.drag_delta_start != fan::vec2(-1)) {
        return window.get_mouse_position() - window.drag_delta_start;
      }
    }
    return fan::vec2();
  }

  bool is_key_pressed(int key) {
    return window.key_state(key) == (int)fan::mouse_state::press;
  }
  bool is_key_down(int key) {
    int state = window.key_state(key);
    return
      state == (int)fan::mouse_state::press ||
      state == (int)fan::mouse_state::repeat;
  }
  bool is_key_released(int key) {
    return window.key_state(key) == (int)fan::mouse_state::release;
  }

  struct shape_t : shaper_t::ShapeID_t {
    using shaper_t::ShapeID_t::ShapeID_t;
    shape_t() {
      sic();
    }
    template <typename T>
    requires requires(T t) { typename T::type_t; }
    shape_t(const T& properties) : shape_t() {
      auto shape_type = T::type_t::shape_type;
      *this = gloco->shape_functions[shape_type].push_back((void*)&properties);
    }

    shape_t(shaper_t::ShapeID_t&& s) {
      //if (s.iic() == false) {
      //  if (((shape_t*)&s)->get_shape_type() == shape_type_t::polygon) {
      //    loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
      //    loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
      //    *dst_data = *src_data;
      //  }
      //}
      NRI = s.NRI;
      s.sic();
    }

    shape_t(const shaper_t::ShapeID_t& s) : shape_t() {

      if (s.iic()) {
        return;
      }

      {
        auto sti = gloco->shaper.ShapeList[s].sti;
        shaper_deep_copy(this, (const loco_t::shape_t*)&s, sti);
      }
      if (((shape_t*)&s)->get_shape_type() == shape_type_t::polygon) {
        loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
        loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
        if (gloco->get_renderer() == renderer_t::opengl) {
          dst_data->vao.open(gloco->context.gl);
          dst_data->vbo.open(gloco->context.gl, src_data->vbo.m_target);

          auto& shape_data = gloco->shaper.GetShapeTypes(shape_type_t::polygon).renderer.gl;
          fan::graphics::context_shader_t shader;
          if (!shape_data.shader.iic()) {
            shader = gloco->shader_get(shape_data.shader);
          }
          dst_data->vao.bind(gloco->context.gl);
          dst_data->vbo.bind(gloco->context.gl);
          uint64_t ptr_offset = 0;
          for (shape_gl_init_t& location : gloco->polygon.locations) {
            if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
              location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
            }
            fan_opengl_call(glEnableVertexAttribArray(location.index.first));
            switch (location.type) {
            case GL_UNSIGNED_INT:
            case GL_INT: {
              fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
              break;
            }
            default: {
              fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
            }
            }
            // instancing
            if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
              if (shape_data.instanced) {
                fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
              }
            }
            switch (location.type) {
            case GL_FLOAT: {
              ptr_offset += location.size * sizeof(GLfloat);
              break;
            }
            case GL_UNSIGNED_INT: {
              ptr_offset += location.size * sizeof(GLuint);
              break;
            }
            default: {
              fan::throw_error_impl();
            }
            }
          }
          fan::opengl::core::write_glbuffer(gloco->context.gl, dst_data->vbo.m_buffer, 0, dst_data->buffer_size, dst_data->vbo.m_usage, dst_data->vbo.m_target);
          glBindBuffer(GL_COPY_READ_BUFFER, src_data->vbo.m_buffer);
          glBindBuffer(GL_COPY_WRITE_BUFFER, dst_data->vbo.m_buffer);
          glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, dst_data->buffer_size);
          loco_t::polygon_vertex_t* ri = new loco_t::polygon_vertex_t[dst_data->buffer_size / sizeof(loco_t::polygon_vertex_t)];
          loco_t::polygon_vertex_t* ri2 = new loco_t::polygon_vertex_t[dst_data->buffer_size / sizeof(loco_t::polygon_vertex_t)];
          fan::opengl::core::get_glbuffer(gloco->context.gl, ri, dst_data->vbo.m_buffer, dst_data->buffer_size, 0, dst_data->vbo.m_target);
          fan::opengl::core::get_glbuffer(gloco->context.gl, ri2, src_data->vbo.m_buffer, src_data->buffer_size, 0, src_data->vbo.m_target);
          delete[] ri;
        }
        else {
          fan::throw_error_impl();
        }
      }
    }

    shape_t(shape_t&& s) : shape_t(std::move(*dynamic_cast<shaper_t::ShapeID_t*>(&s))) {

    }

    shape_t(const loco_t::shape_t& s) : shape_t(*dynamic_cast<const shaper_t::ShapeID_t*>(&s)) {
      //NRI = s.NRI;
    }

    loco_t::shape_t& operator=(const loco_t::shape_t& s) {
      if (iic() == false) {
        remove();
      }
      if (s.iic()) {
        return *this;
      }
      if (this != &s) {
        {
          auto sti = gloco->shaper.ShapeList[s].sti;

          shaper_deep_copy(this, (const loco_t::shape_t*)&s, sti);
        }
        if (((shape_t*)&s)->get_shape_type() == shape_type_t::polygon) {
          loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
          loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
          if (gloco->get_renderer() == renderer_t::opengl) {
            dst_data->vao.open(gloco->context.gl);
            dst_data->vbo.open(gloco->context.gl, src_data->vbo.m_target);

            auto& shape_data = gloco->shaper.GetShapeTypes(shape_type_t::polygon).renderer.gl;
            fan::graphics::context_shader_t shader;
            if (!shape_data.shader.iic()) {
              shader = gloco->shader_get(shape_data.shader);
            }
            dst_data->vao.bind(gloco->context.gl);
            dst_data->vbo.bind(gloco->context.gl);
            uint64_t ptr_offset = 0;
            for (shape_gl_init_t& location : gloco->polygon.locations) {
              if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
                location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
              }
              fan_opengl_call(glEnableVertexAttribArray(location.index.first));
              switch (location.type) {
              case GL_UNSIGNED_INT:
              case GL_INT: {
                fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
                break;
              }
              default: {
                fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
              }
              }
              // instancing
              if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
                if (shape_data.instanced) {
                  fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
                }
              }
              switch (location.type) {
              case GL_FLOAT: {
                ptr_offset += location.size * sizeof(GLfloat);
                break;
              }
              case GL_UNSIGNED_INT: {
                ptr_offset += location.size * sizeof(GLuint);
                break;
              }
              default: {
                fan::throw_error_impl();
              }
              }
              fan::opengl::core::write_glbuffer(gloco->context.gl, dst_data->vbo.m_buffer, 0, dst_data->buffer_size, dst_data->vbo.m_usage, dst_data->vbo.m_target);
              glBindBuffer(GL_COPY_READ_BUFFER, src_data->vbo.m_buffer);
              glBindBuffer(GL_COPY_WRITE_BUFFER, dst_data->vbo.m_buffer);
              glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, dst_data->buffer_size);
            }
          }
          else {
            fan::throw_error_impl();
          }
        }
        //fan::print("i dont know what to do");
        //NRI = s.NRI;
      }
      return *this;
    }

    loco_t::shape_t& operator=(loco_t::shape_t&& s) {
      if (iic() == false) {
        remove();
      }
      if (s.iic()) {
        return *this;
      }

      if (this != &s) {
        if (s.iic() == false) {

        }
        NRI = s.NRI;
        s.sic();
      }
      return *this;
    }
#if defined(fan_json)
    operator fan::json();
    operator std::string();
    shape_t& operator=(const fan::json& json);
    shape_t& operator=(const std::string&); // assume json string
#endif

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
      if (gloco->shaper.ShapeList.Usage() == 0) {
        return;
      }
      auto shape_type = get_shape_type();
#if defined(loco_vfi)
      if (shape_type == loco_t::shape_type_t::vfi) {
        gloco->vfi.erase(*this);
        sic();
        return;
      }
#endif
      if (shape_type == loco_t::shape_type_t::polygon) {
        auto ri = (polygon_t::ri_t*)GetData(gloco->shaper);
        ri->vbo.close(gloco->context.gl);
        ri->vao.close(gloco->context.gl);
      }
      gloco->shaper.remove(*this);
      sic();
    }
    void erase() {
      remove();
    }


    // many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t

    uint16_t get_shape_type() const {
      return gloco->shaper.ShapeList[*this].sti;
    }

    void set_position(const fan::vec2& position) {
      gloco->shape_functions[get_shape_type()].set_position2(this, position);
    }
    void set_position(const fan::vec3& position) {
      gloco->shape_functions[get_shape_type()].set_position3(this, position);
    }

    fan::vec3 get_position() {
      auto shape_type = get_shape_type();
      return gloco->shape_functions[shape_type].get_position(this);
    }

    void set_size(const fan::vec2& size) {
      gloco->shape_functions[get_shape_type()].set_size(this, size);
    }

    void set_size3(const fan::vec3& size) {
      gloco->shape_functions[get_shape_type()].set_size3(this, size);
    }

    fan::vec2 get_size() {
      return gloco->shape_functions[get_shape_type()].get_size(this);
    }

    fan::vec3 get_size3() {
      return gloco->shape_functions[get_shape_type()].get_size3(this);
    }

    void set_rotation_point(const fan::vec2& rotation_point) {
      gloco->shape_functions[get_shape_type()].set_rotation_point(this, rotation_point);
    }

    fan::vec2 get_rotation_point() {
      return gloco->shape_functions[get_shape_type()].get_rotation_point(this);
    }

    void set_color(const fan::color& color) {
      gloco->shape_functions[get_shape_type()].set_color(this, color);
    }

    fan::color get_color() {
      return gloco->shape_functions[get_shape_type()].get_color(this);
    }

    void set_angle(const fan::vec3& angle) {
      gloco->shape_functions[get_shape_type()].set_angle(this, angle);
    }

    fan::vec3 get_angle() {
      return gloco->shape_functions[get_shape_type()].get_angle(this);
    }

    fan::vec2 get_tc_position() {
      return gloco->shape_functions[get_shape_type()].get_tc_position(this);
    }

    void set_tc_position(const fan::vec2& tc_position) {
      gloco->shape_functions[get_shape_type()].set_tc_position(this, tc_position);
    }

    fan::vec2 get_tc_size() {
      return gloco->shape_functions[get_shape_type()].get_tc_size(this);
    }

    void set_tc_size(const fan::vec2& tc_size) {
      gloco->shape_functions[get_shape_type()].set_tc_size(this, tc_size);
    }

    bool load_tp(loco_t::texturepack_t::ti_t* ti) {
      return gloco->shape_functions[get_shape_type()].load_tp(this, ti);
    }

    loco_t::texturepack_t::ti_t get_tp() {
      loco_t::texturepack_t::ti_t ti;
      ti.image = &gloco->default_texture;
      auto& image_data = gloco->image_get_data(*ti.image);
      ti.position = get_tc_position() * image_data.size;
      ti.size = get_tc_size() * image_data.size;
      return ti;
      //return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tp(this);
    }

    bool set_tp(loco_t::texturepack_t::ti_t* ti) {
      return load_tp(ti);
    }

    loco_t::camera_t get_camera() {
      return gloco->shape_functions[get_shape_type()].get_camera(this);
    }

    void set_camera(loco_t::camera_t camera) {
      gloco->shape_functions[get_shape_type()].set_camera(this, camera);
    }

    loco_t::viewport_t get_viewport() {
      return gloco->shape_functions[get_shape_type()].get_viewport(this);
    }

    void set_viewport(loco_t::viewport_t viewport) {
      gloco->shape_functions[get_shape_type()].set_viewport(this, viewport);
    }

    fan::vec2 get_grid_size() {
      return gloco->shape_functions[get_shape_type()].get_grid_size(this);
    }

    void set_grid_size(const fan::vec2& grid_size) {
      gloco->shape_functions[get_shape_type()].set_grid_size(this, grid_size);
    }

    loco_t::image_t get_image() {
      return gloco->shape_functions[get_shape_type()].get_image(this);
    }

    void set_image(loco_t::image_t image) {
      gloco->shape_functions[get_shape_type()].set_image(this, image);
    }

    fan::graphics::image_data_t& get_image_data() {
      return gloco->shape_functions[get_shape_type()].get_image_data(this);
    }

    std::array<loco_t::image_t, 30> get_images() {
      auto shape_type = get_shape_type();
      if (shape_type == shape_type_t::sprite) {
        return ((sprite_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images;
      }
      else if (shape_type == shape_type_t::unlit_sprite) {
        return ((unlit_sprite_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images;
      }
#if fan_debug >= fan_debug_medium
      fan::throw_error("only for sprite and unlit_sprite");
#endif
      return {};
    }

    void set_images(const std::array<loco_t::image_t, 30>& images) {
      auto shape_type = get_shape_type();
      if (shape_type == shape_type_t::sprite) {
        ((sprite_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images = images;
      }
      else if (shape_type == shape_type_t::unlit_sprite) {
        ((unlit_sprite_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images = images;
      }
#if fan_debug >= fan_debug_medium
      else {
        fan::throw_error("only for sprite and unlit_sprite");
      }
#endif
    }

    f32_t get_parallax_factor() {
      return gloco->shape_functions[get_shape_type()].get_parallax_factor(this);
    }

    void set_parallax_factor(f32_t parallax_factor) {
      gloco->shape_functions[get_shape_type()].set_parallax_factor(this, parallax_factor);
    }

    uint32_t get_flags() {
      auto f = gloco->shape_functions[get_shape_type()].get_flags;
      if (f) {
        return f(this);
      }
      return 0;
    }

    void set_flags(uint32_t flag) {
      return gloco->shape_functions[get_shape_type()].set_flags(this, flag);
    }

    f32_t get_radius() {
      return gloco->shape_functions[get_shape_type()].get_radius(this);
    }

    fan::vec3 get_src() {
      return gloco->shape_functions[get_shape_type()].get_src(this);
    }

    fan::vec3 get_dst() {
      return gloco->shape_functions[get_shape_type()].get_dst(this);
    }

    f32_t get_outline_size() {
      return gloco->shape_functions[get_shape_type()].get_outline_size(this);
    }

    fan::color get_outline_color() {
      return gloco->shape_functions[get_shape_type()].get_outline_color(this);
    }

    void set_outline_color(const fan::color& color) {
      return gloco->shape_functions[get_shape_type()].set_outline_color(this, color);
    }

    void reload(uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter = fan::graphics::image_filter::linear) {
      gloco->shape_functions[get_shape_type()].reload(this, format, image_data, image_size, filter);
    }

    void reload(uint8_t format, const fan::vec2& image_size, uint32_t filter = fan::graphics::image_filter::linear) {
      void* data[4]{};
      gloco->shape_functions[get_shape_type()].reload(this, format, data, image_size, filter);
    }

    // universal image specific
    void reload(uint8_t format, loco_t::image_t images[4], uint32_t filter = fan::graphics::image_filter::linear) {
      loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)GetData(gloco->shaper);
      uint8_t image_count_new = fan::graphics::get_channel_amount(format);
      if (format != ri.format) {
        auto sti = gloco->shaper.ShapeList[*this].sti;
        uint8_t* KeyPack = gloco->shaper.GetKeys(*this);
        loco_t::image_t vi_image = shaper_get_key_safe(loco_t::image_t, texture_t, image);


        auto shader = gloco->shaper.GetShader(sti);
        gloco->shader_set_vertex(
          shader,
          loco_t::read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
        );
        {
          std::string fs;
          switch (format) {
          case fan::graphics::image_format::yuv420p: {
            fs = loco_t::read_shader("shaders/opengl/2D/objects/yuv420p.fs");
            break;
          }
          case fan::graphics::image_format::nv12: {
            fs = loco_t::read_shader("shaders/opengl/2D/objects/nv12.fs");
            break;
          }
          default: {
            fan::throw_error("unimplemented format");
          }
          }
          gloco->shader_set_fragment(shader, fs);
          gloco->shader_compile(shader);
        }
        set_image(images[0]);
        std::copy(&images[1], &images[0] + ri.images_rest.size(), ri.images_rest.data());
        ri.format = format;
      }
    }

    void set_line(const fan::vec2& src, const fan::vec2& dst) {
      gloco->shape_functions[get_shape_type()].set_line(this, src, dst);
    }

    bool is_mouse_inside() {
      switch (get_shape_type()) {
      case shape_type_t::rectangle: {
        return fan_2d::collision::rectangle::point_inside_no_rotation(
          gloco->get_mouse_position(get_camera(), get_viewport()),
          get_position(),
          get_size()
        );
      }
      default: {
        break;
      }
      }
    }

  private:
  };

  struct light_flags_e {
    enum {
      circle = 0,
      square = 1 << 0,
      lava = 1 << 1, // does this belong here
      additive = 1 << 2,
      multiplicative = 1 << 3,
    };
  };

  struct light_t {

    static inline shaper_t::KeyTypeIndex_t shape_type = shape_type_t::light;
    static constexpr int kpi = kp::light;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t parallax_factor;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      uint32_t flags = 0;
      fan::vec3 angle;
    };;
    struct ri_t {

    };

#pragma pack(pop)

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{{5, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{6, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

    struct properties_t {
      using type_t = light_t;


      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      uint32_t flags = 0;
      fan::vec3 angle = 0;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
    };

    shape_t push_back(const properties_t& properties) {
      vi_t vi;
      vi.position = properties.position;
      vi.parallax_factor = properties.parallax_factor;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.flags = properties.flags;
      vi.angle = properties.angle;
      ri_t ri;

      return shape_add(shape_type, vi, ri,
        Key_e::light, (uint8_t)0,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }
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

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_color"}, 4, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
      shape_gl_init_t{{1, "in_src"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
      shape_gl_init_t{{2, "in_dst"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
    };

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


    shape_t push_back(const properties_t& properties) {
      vi_t vi;
      vi.src = properties.src;
      vi.dst = properties.dst;
      vi.color = properties.color;
      ri_t ri;

      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.src.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }

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
    std::vector<shape_gl_init_t> locations = {
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


    shape_t push_back(const properties_t& properties) {
      vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.color = properties.color;
      vi.outline_color = properties.outline_color;
      vi.angle = properties.angle;
      vi.rotation_point = properties.rotation_point;
      ri_t ri;

      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }

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

    std::vector<shape_gl_init_t> locations = {
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
    };

    struct properties_t {
      using type_t = sprite_t;

      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      uint32_t flags = light_flags_e::circle | light_flags_e::multiplicative;
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

    shape_t push_back(const properties_t& properties) {

      vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;
      vi.tc_position = properties.tc_position;
      vi.tc_size = properties.tc_size;
      vi.parallax_factor = properties.parallax_factor;
      vi.seed = properties.seed;

      ri_t ri;
      ri.images = properties.images;

      loco_t& loco = *OFFSETLESS(this, loco_t, sprite);
      if (loco.window.renderer == loco_t::renderer_t::opengl) {

        if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
          return shape_add(
            shape_type, vi, ri,
            Key_e::depth,
            static_cast<uint16_t>(properties.position.z),
            Key_e::blending, static_cast<uint8_t>(properties.blending),
            Key_e::image, properties.image,
            Key_e::viewport, properties.viewport,
            Key_e::camera, properties.camera,
            Key_e::ShapeType, shape_type,
            Key_e::draw_mode, properties.draw_mode,
            Key_e::vertex_count, properties.vertex_count
          );
        }
        else {
          // Legacy version requires array of 6 identical vertices
          vi_t vertices[6];
          for (int i = 0; i < 6; i++) {
            vertices[i] = vi;
          }

          return shape_add(
            shape_type, vertices[0], ri, Key_e::depth,
            static_cast<uint16_t>(properties.position.z),
            Key_e::blending, static_cast<uint8_t>(properties.blending),
            Key_e::image, properties.image, Key_e::viewport,
            properties.viewport, Key_e::camera, properties.camera,
            Key_e::ShapeType, shape_type,
            Key_e::draw_mode, properties.draw_mode,
            Key_e::vertex_count, properties.vertex_count
          );
        }
      }
      else if (loco.window.renderer == renderer_t::vulkan) {
        return shape_add(
          shape_type, vi, ri, Key_e::depth,
          static_cast<uint16_t>(properties.position.z),
          Key_e::blending, static_cast<uint8_t>(properties.blending),
          Key_e::image, properties.image, Key_e::viewport,
          properties.viewport, Key_e::camera, properties.camera,
          Key_e::ShapeType, shape_type,
          Key_e::draw_mode, properties.draw_mode,
          Key_e::vertex_count, properties.vertex_count
        );
      }

      return {};
    }

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

    std::vector<shape_gl_init_t> locations = {
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

    shape_t push_back(const properties_t& properties) {
      //KeyPack.ShapeType = shape_type;
      vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;
      vi.tc_position = properties.tc_position;
      vi.tc_size = properties.tc_size;
      vi.parallax_factor = properties.parallax_factor;
      vi.seed = properties.seed;
      ri_t ri;
      ri.images = properties.images;
      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.image,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }

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

      std::string text;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
    };

    shape_t push_back(const properties_t& properties) {
      return gloco->shaper.add(shape_type_t::text, nullptr, 0, nullptr, nullptr);
    }
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
      fan::vec3 angle;
      uint32_t flags;
    };
    struct ri_t {

    };

#pragma pack(pop)

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
      shape_gl_init_t{{1, "in_radius"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
      shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
      shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
      shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
      shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))}
    };

    struct properties_t {
      using type_t = circle_t;

      fan::vec3 position = 0;
      f32_t radius = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = 0;
      uint32_t flags = 0;

      bool blending = false;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
    };


    loco_t::shape_t push_back(const circle_t::properties_t& properties) {
      circle_t::vi_t vi;
      vi.position = properties.position;
      vi.radius = properties.radius;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;
      circle_t::ri_t ri;
      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }

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
      fan::vec3 angle;
      uint32_t flags;
      fan::color outline_color;
    };
    struct ri_t {

    };

#pragma pack(pop)

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
      shape_gl_init_t{{1, "in_center0"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, center0)) },
      shape_gl_init_t{{2, "in_center1"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, center1)) },
      shape_gl_init_t{{3, "in_radius"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
      shape_gl_init_t{{4, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
      shape_gl_init_t{{5, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
      shape_gl_init_t{{6, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
      shape_gl_init_t{{7, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{{8, "in_outline_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color)) },
    };

    struct properties_t {
      using type_t = capsule_t;

      fan::vec3 position = 0;
      fan::vec2 center0 = 0;
      fan::vec2 center1 = { 0, 1.f };
      f32_t radius = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::color outline_color = color;
      fan::vec3 angle = 0;
      uint32_t flags = 0;

      bool blending = true;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
    };
    loco_t::shape_t push_back(const capsule_t::properties_t& properties) {
      capsule_t::vi_t vi;
      vi.position = properties.position;
      vi.center0 = properties.center0;
      vi.center1 = properties.center1;
      vi.radius = properties.radius;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.outline_color = properties.outline_color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;
      capsule_t::ri_t ri;
      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }
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

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, position)) },
      shape_gl_init_t{{1, "in_color"}, 4, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, color)) },
      shape_gl_init_t{{2, "in_offset"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, offset)) },
      shape_gl_init_t{{3, "in_angle"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, angle)) },
      shape_gl_init_t{{4, "in_rotation_point"}, 2, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, rotation_point)) },
    };

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
    loco_t::shape_t push_back(const properties_t& properties) {
      if (properties.vertices.empty()) {
        fan::throw_error("invalid vertices");
      }

      std::vector<loco_t::polygon_vertex_t> polygon_vertices(properties.vertices.size());
      for (std::size_t i = 0; i < properties.vertices.size(); ++i) {
        polygon_vertices[i].position = properties.vertices[i].position;
        polygon_vertices[i].color = properties.vertices[i].color;
        polygon_vertices[i].offset = properties.position;
        polygon_vertices[i].angle = properties.angle;
        polygon_vertices[i].rotation_point = properties.rotation_point;
      }

      vi_t vis;
      ri_t ri;
      ri.buffer_size = sizeof(decltype(polygon_vertices)::value_type) * polygon_vertices.size();
      ri.vao.open(gloco->context.gl);
      ri.vao.bind(gloco->context.gl);
      ri.vbo.open(gloco->context.gl, GL_ARRAY_BUFFER);
      fan::opengl::core::write_glbuffer(
        gloco->context.gl,
        ri.vbo.m_buffer,
        polygon_vertices.data(),
        ri.buffer_size,
        GL_STATIC_DRAW,
        ri.vbo.m_target
      );

      auto& shape_data = gloco->shaper.GetShapeTypes(shape_type).renderer.gl;

      fan::graphics::context_shader_t shader;
      if (!shape_data.shader.iic()) {
        shader = gloco->shader_get(shape_data.shader);
      }
      uint64_t ptr_offset = 0;
      for (shape_gl_init_t& location : locations) {
        if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
          location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
        }
        fan_opengl_call(glEnableVertexAttribArray(location.index.first));
        switch (location.type) {
        case GL_UNSIGNED_INT:
        case GL_INT: {
          fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
          break;
        }
        default: {
          fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
        }
        }
        // instancing
        if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
          if (shape_data.instanced) {
            fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
          }
        }
        switch (location.type) {
        case GL_FLOAT: {
          ptr_offset += location.size * sizeof(GLfloat);
          break;
        }
        case GL_UNSIGNED_INT: {
          ptr_offset += location.size * sizeof(GLuint);
          break;
        }
        default: {
          fan::throw_error_impl();
        }
        }
      }

      return shape_add(shape_type, vis, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, (uint32_t)properties.vertices.size()
      );
    }
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

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, size)},
      shape_gl_init_t{{2, "in_grid_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, grid_size)},
      shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, rotation_point)},
      shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, color)},
      shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, angle)},
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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
    };

    shape_t push_back(const properties_t& properties) {
      vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.grid_size = properties.grid_size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      ri_t ri;
      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }
  }grid;


  struct particles_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::particles;
    static constexpr int kpi = kp::texture;

    std::vector<shape_gl_init_t> locations = {};

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

    shape_t push_back(const properties_t& properties) {
      //KeyPack.ShapeType = shape_type;
      vi_t vi;
      ri_t ri;
      ri.position = properties.position;
      ri.size = properties.size;
      ri.color = properties.color;

      ri.begin_time = fan::time::clock::now();
      ri.alive_time = properties.alive_time;
      ri.respawn_time = properties.respawn_time;
      ri.count = properties.count;
      ri.position_velocity = properties.position_velocity;
      ri.angle_velocity = properties.angle_velocity;
      ri.begin_angle = properties.begin_angle;
      ri.end_angle = properties.end_angle;
      ri.angle = properties.angle;
      ri.gap_size = properties.gap_size;
      ri.max_spread_size = properties.max_spread_size;
      ri.size_velocity = properties.size_velocity;
      ri.shape = properties.shape;

      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.image,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }

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
      uint8_t format = fan::graphics::image_format::undefined;
    };

#pragma pack(pop)

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shape_gl_init_t{{3, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))}
      };

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

    shape_t push_back(const properties_t& properties) {
      vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.tc_position = properties.tc_position;
      vi.tc_size = properties.tc_size;
      ri_t ri;
      // + 1
      std::copy(&properties.images[1], &properties.images[0] + properties.images.size(), ri.images_rest.data());
      shape_t shape = shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.images[0],
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      ((ri_t*)shape.GetData(gloco->shaper))->format = shape.get_image_data().image_settings.format;

      return shape;
    }

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

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 0)},
      shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 1)},
      shape_gl_init_t{{5, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 2)},
      shape_gl_init_t{{6, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 3)},
      shape_gl_init_t{{7, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

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


    shape_t push_back(const properties_t& properties) {
      kps_t::common_t KeyPack;
      KeyPack.ShapeType = shape_type;
      KeyPack.depth = properties.position.z;
      KeyPack.blending = properties.blending;
      KeyPack.camera = properties.camera;
      KeyPack.viewport = properties.viewport;
      vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.rotation_point = properties.rotation_point;
      ri_t ri;

      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }


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

    std::vector<shape_gl_init_t> locations = {
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
      bool blending = true;

      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 6;
    };

    shape_t push_back(const properties_t& properties) {
      //KeyPack.ShapeType = shape_type;
      vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;
      vi.tc_position = properties.tc_position;
      vi.tc_size = properties.tc_size;
      vi.parallax_factor = properties.parallax_factor;
      vi.seed = properties.seed;
      ri_t ri;
      ri.images = properties.images;
      loco_t::shape_t ret = shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.image,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      gloco->shaper.GetShader(shape_type) = properties.shader;
      return ret;
    }

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

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t,  position)},
      shape_gl_init_t{{1, "in_size"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{{2, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))}
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

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
      uint32_t vertex_count = 36;
    };


    shape_t push_back(const properties_t& properties) {
      vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.color = properties.color;
      //vi.angle = properties.angle;
      ri_t ri;

      loco_t& loco = *OFFSETLESS(this, loco_t, rectangle3d);

      if (loco.window.renderer == loco_t::renderer_t::opengl) {
        if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
          // might not need depth
          return shape_add(shape_type, vi, ri,
            Key_e::depth, (uint16_t)properties.position.z,
            Key_e::blending, (uint8_t)properties.blending,
            Key_e::viewport, properties.viewport,
            Key_e::camera, properties.camera,
            Key_e::ShapeType, shape_type,
            Key_e::draw_mode, properties.draw_mode,
            Key_e::vertex_count, properties.vertex_count
          );
        }
        else {
          vi_t vertices[36];
          for (int i = 0; i < 36; i++) {
            vertices[i] = vi;
          }

          return shape_add(shape_type, vertices[0], ri,
            Key_e::depth, (uint16_t)properties.position.z,
            Key_e::blending, (uint8_t)properties.blending,
            Key_e::viewport, properties.viewport,
            Key_e::camera, properties.camera,
            Key_e::ShapeType, shape_type,
            Key_e::draw_mode, properties.draw_mode,
            Key_e::vertex_count, properties.vertex_count
          );
        }
      }
      else if (loco.window.renderer == loco_t::renderer_t::vulkan) {

      }
      fan::throw_error();
      return{};
    }

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

    std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{{0, "in_color"}, 4, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
      shape_gl_init_t{{1, "in_src"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
      shape_gl_init_t{{2, "in_dst"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
    };

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

    shape_t push_back(const properties_t& properties) {
      vi_t vi;
      vi.src = properties.src;
      vi.dst = properties.dst;
      vi.color = properties.color;
      ri_t ri;

      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.src.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }

  }line3d;

  //-------------------------------------shapes-------------------------------------

  using shape_shader_locations_t = decltype(loco_t::shaper_t::BlockProperties_t::gl_t::locations);

  inline void shape_open(
    uint16_t shape_type,
    std::size_t sizeof_vi,
    std::size_t sizeof_ri,
    const shape_shader_locations_t& shape_shader_locations,
    const std::string& vertex,
    const std::string& fragment,
    loco_t::shaper_t::ShapeRenderDataSize_t instance_count = 1,
    bool instanced = true
  ) {
    loco_t::shader_t shader = shader_create();

    shader_set_vertex(shader,
      read_shader(vertex)
    );

    shader_set_fragment(shader,
      read_shader(fragment)
    );

    shader_compile(shader);

    shaper_t::BlockProperties_t bp;
    bp.MaxElementPerBlock = (loco_t::shaper_t::MaxElementPerBlock_t)MaxElementPerBlock;
    bp.RenderDataSize = (decltype(loco_t::shaper_t::BlockProperties_t::RenderDataSize))(sizeof_vi * instance_count);
    bp.DataSize = sizeof_ri;

    if (window.renderer == renderer_t::opengl) {
      loco_t::shaper_t::BlockProperties_t::gl_t d;
      d.locations = shape_shader_locations;
      d.shader = shader;
      d.instanced = instanced;
      bp.renderer.gl = d;
    }
#if defined(fan_vulkan)
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

      vk.shape_data.open_descriptors(gloco->context.vk, { ds_properties.begin(), ds_properties.end() });
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
      bp.renderer.vk = d;
    }
#endif

    gloco->shaper.SetShapeType(shape_type, bp);

    gloco->shape_functions[shape_type] = get_shape_functions(shape_type);
  }


#if defined(loco_sprite)
  loco_t::shader_t get_sprite_vertex_shader(const std::string& fragment) {
    loco_t::shader_t shader = shader_create();
    shader_set_vertex(
      shader,
      loco_t::read_shader("shaders/opengl/2D/objects/sprite.vs")
    );
    shader_set_fragment(shader, fragment);
    if (!shader_compile(shader)) {
      shader_erase(shader);
      shader.sic();
    }
    return shader;
  }
#endif


#if defined(loco_vfi)
#include <fan/graphics/gui/vfi.h>
  vfi_t vfi;
#endif

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
  bool show_fps = false;
  bool render_settings_menu = 0;

  ImFont* fonts[6];
  ImFont* fonts_bold[6];

#include <fan/graphics/gui/settings_menu.h>
  settings_menu_t settings_menu;
#endif
  bool render_shapes_top = false;
  //gui

  std::vector<uint8_t> create_noise_image_data(const fan::vec2& image_size, int seed = fan::random::value_i64(0, ((uint32_t)-1) / 2)) {
    fan::print("TODOO");
    //FastNoiseLite noise;
    //noise.SetFractalType(FastNoiseLite::FractalType_FBm);
    //noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    //noise.SetFrequency(0.010);
    //noise.SetFractalGain(0.5);
    //noise.SetFractalLacunarity(2.0);
    //noise.SetFractalOctaves(5);
    //noise.SetSeed(seed);
    //noise.SetFractalPingPongStrength(2.0);
    f32_t noise_tex_min = -1;
    f32_t noise_tex_max = 0.1;

    std::vector<uint8_t> noise_data_rgb(image_size.multiply() * 3);

    //int index = 0;

    //float scale = 255.f / (noise_tex_max - noise_tex_min);

    //for (int y = 0; y < image_size.y; y++)
    //{
    //  for (int x = 0; x < image_size.x; x++)
    //  {
    //    float noiseValue = noise.GetNoise((float)x, (float)y);
    //    unsigned char cNoise = (unsigned char)std::max(0.0f, std::min(255.0f, (noiseValue - noise_tex_min) * scale));
    //    noise_data_rgb[index * 3 + 0] = cNoise;
    //    noise_data_rgb[index * 3 + 1] = cNoise;
    //    noise_data_rgb[index * 3 + 2] = cNoise;
    //    index++;
    //  }
    //}

    return noise_data_rgb;
  }

  loco_t::image_t create_noise_image(const fan::vec2& image_size) {

    loco_t::image_load_properties_t lp;
    lp.format = fan::graphics::image_format::rgb_unorm;
    lp.internal_format = fan::graphics::image_format::rgb_unorm;
    lp.min_filter = fan::graphics::image_filter::linear;
    lp.mag_filter = fan::graphics::image_filter::linear;
    lp.visual_output = fan::graphics::image_sampler_address_mode::mirrored_repeat;

    loco_t::image_t image;

    auto noise_data = create_noise_image_data(image_size);

    fan::image::image_info_t ii;
    ii.data = noise_data.data();
    ii.size = image_size;
    ii.channels = 3;

    image = image_load(ii, lp);
    return image;
  }
  loco_t::image_t create_noise_image(const fan::vec2& image_size, const std::vector<uint8_t>& noise_data) {

    loco_t::image_load_properties_t lp;
    lp.format = fan::graphics::image_format::rgb_unorm;
    lp.internal_format = fan::graphics::image_format::rgb_unorm;
    lp.min_filter = fan::graphics::image_filter::linear;
    lp.mag_filter = fan::graphics::image_filter::linear;
    lp.visual_output = fan::graphics::image_sampler_address_mode::mirrored_repeat;

    loco_t::image_t image;

    fan::image::image_info_t ii;
    ii.data = (void*)noise_data.data();
    ii.size = image_size;
    ii.channels = 3;

    image = image_load(ii, lp);
    return image;
  }
  static fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position, const fan::vec2i& window_size) {
    return fan::vec2((2.0f * mouse_position.x) / window_size.x - 1.0f, 1.0f - (2.0f * mouse_position.y) / window_size.y);
  }
  fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const {
    return convert_mouse_to_ndc(mouse_position, gloco->window.get_size());
  }
  fan::vec2 convert_mouse_to_ndc() const {
    return convert_mouse_to_ndc(gloco->get_mouse_position(), gloco->window.get_size());
  }
  static fan::ray3_t convert_mouse_to_ray(const fan::vec2i& mouse_position, const fan::vec2& screen_size, const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {

    fan::vec4 ray_ndc((2.0f * mouse_position.x) / screen_size.x - 1.0f, 1.0f - (2.0f * mouse_position.y) / screen_size.y, 1.0f, 1.0f);

    fan::mat4 inverted_projection = projection.inverse();

    fan::vec4 ray_clip = inverted_projection * ray_ndc;

    ray_clip.z = -1.0f;
    ray_clip.w = 0.0f;

    fan::mat4 inverted_view = view.inverse();

    fan::vec4 ray_world = inverted_view * ray_clip;

    fan::vec3 ray_dir = fan::vec3(ray_world.x, ray_world.y, ray_world.z).normalize();

    fan::vec3 ray_origin = camera_position;
    return fan::ray3_t(ray_origin, ray_dir);
  }
  fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
    return convert_mouse_to_ray(get_mouse_position(), window.get_size(), camera_position, projection, view);
  }
  fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view) {
    return convert_mouse_to_ray(get_mouse_position(), window.get_size(), camera_get_position(perspective_camera.camera), projection, view);
  }
  static bool is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size) {
    fan::vec3 min_bounds = position - size;
    fan::vec3 max_bounds = position + size;

    fan::vec3 t_min = (min_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));
    fan::vec3 t_max = (max_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));

    fan::vec3 t1 = t_min.min(t_max);
    fan::vec3 t2 = t_min.max(t_max);

    float t_near = fan::max(t1.x, fan::max(t1.y, t1.z));
    float t_far = fan::min(t2.x, fan::min(t2.y, t2.z));

    return t_near <= t_far && t_far >= 0.0f;
  }


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
      loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)cid.GetData(gloco->shaper);
      uint8_t image_amount = fan::graphics::get_channel_amount(ri.format);
      for (uint32_t i = 0; i < image_amount; ++i) {
        wresources[i].close();
        gloco->image_unload(ri.images_rest[i]);
      }
    }

    void resize(loco_t* loco, loco_t::shape_t& id, uint8_t format, fan::vec2ui size, uint32_t filter = fan::graphics::image_filter::linear) {
      auto vi_image = id.get_image();
      if (vi_image.iic() || vi_image == loco->default_texture) {
        id.reload(format, size, filter);
      }

      auto& ri = *(universal_image_renderer_t::ri_t*)id.GetData(loco->shaper);

      if (inited == false) {
        id.reload(format, size, filter);
        vi_image = id.get_image();

        uint8_t image_amount = fan::graphics::get_channel_amount(format);

        for (uint32_t i = 0; i < image_amount; ++i) {
          if (i == 0) {
            wresources[i].open(gloco->image_get_handle(vi_image));
          }
          else {
            wresources[i].open(gloco->image_get_handle(ri.images_rest[i - 1]));
          }
        }
        inited = true;
      }
      else {
        if (gloco->image_get_data(vi_image).size == size) {
          return;
        }

        for (uint32_t i = 0; i < fan::graphics::get_channel_amount(ri.format); ++i) {
          wresources[i].close();
        }

        id.reload(format, size, filter);
        vi_image = id.get_image();

        ri = *(universal_image_renderer_t::ri_t*)id.GetData(loco->shaper);

        uint8_t image_amount = fan::graphics::get_channel_amount(format);

        // Re-register with CUDA after successful reload
        for (uint32_t i = 0; i < image_amount; ++i) {
          if (i == 0) {
            wresources[i].open(gloco->image_get_handle(vi_image));
          }
          else {
            wresources[i].open(gloco->image_get_handle(ri.images_rest[i - 1]));
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
        if (resource == nullptr) {
          return;
        }
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

#if defined(fan_json)
export namespace fan {
  namespace graphics {

    fan::json image_to_json(const auto& image) {
      fan::json image_json;
      if (image.iic()) {
        return image_json;
      }

      auto shape_data = gloco->image_list[image];
      image_json["image_path"] = shape_data.image_path;

      auto lp = gloco->image_get_settings(image);
      image_json["image_visual_output"] = lp.visual_output;
      image_json["image_format"] = lp.format;
      image_json["image_type"] = lp.type;
      image_json["image_min_filter"] = lp.min_filter;
      image_json["image_mag_filter"] = lp.mag_filter;

      return image_json;
    }

    bool shape_to_json(loco_t::shape_t& shape, fan::json* json) {
      fan::json& out = *json;
      switch (shape.get_shape_type()) {
      case loco_t::shape_type_t::light: {
        out["shape"] = "light";
        out["position"] = shape.get_position();
        out["parallax_factor"] = shape.get_parallax_factor();
        out["size"] = shape.get_size();
        out["rotation_point"] = shape.get_rotation_point();
        out["color"] = shape.get_color();
        out["flags"] = shape.get_flags();
        out["angle"] = shape.get_angle();
        break;
      }
      case loco_t::shape_type_t::line: {
        out["shape"] = "line";
        out["color"] = shape.get_color();
        out["src"] = shape.get_src();
        out["dst"] = shape.get_dst();
        break;
      }
      case loco_t::shape_type_t::rectangle: {
        out["shape"] = "rectangle";
        out["position"] = shape.get_position();
        out["size"] = shape.get_size();
        out["rotation_point"] = shape.get_rotation_point();
        out["color"] = shape.get_color();
        out["outline_color"] = shape.get_outline_color();
        out["angle"] = shape.get_angle();
        break;
      }
      case loco_t::shape_type_t::sprite: {
        out["shape"] = "sprite";
        out["position"] = shape.get_position();
        out["parallax_factor"] = shape.get_parallax_factor();
        out["size"] = shape.get_size();
        out["rotation_point"] = shape.get_rotation_point();
        out["color"] = shape.get_color();
        out["angle"] = shape.get_angle();
        out["flags"] = shape.get_flags();
        out["tc_position"] = shape.get_tc_position();
        out["tc_size"] = shape.get_tc_size();
        fan::json images_array = fan::json::array();

        auto main_image = shape.get_image();
        images_array.push_back(image_to_json(main_image));

        auto images = shape.get_images();
        for (auto& image : images) {
          images_array.push_back(image_to_json(image));
        }

        out["images"] = images_array;
        break;
      }
      case loco_t::shape_type_t::unlit_sprite: {
        out["shape"] = "unlit_sprite";
        out["position"] = shape.get_position();
        out["parallax_factor"] = shape.get_parallax_factor();
        out["size"] = shape.get_size();
        out["rotation_point"] = shape.get_rotation_point();
        out["color"] = shape.get_color();
        out["angle"] = shape.get_angle();
        out["flags"] = shape.get_flags();
        out["tc_position"] = shape.get_tc_position();
        out["tc_size"] = shape.get_tc_size();

        fan::json images_array = fan::json::array();

        auto main_image = shape.get_image();
        images_array.push_back(image_to_json(main_image));

        auto images = shape.get_images();
        for (auto& image : images) {
          images_array.push_back(image_to_json(image));
        }

        out["images"] = images_array;
        
        break;
      }
      case loco_t::shape_type_t::text: {
        out["shape"] = "text";
        break;
      }
      case loco_t::shape_type_t::circle: {
        out["shape"] = "circle";
        out["position"] = shape.get_position();
        out["radius"] = shape.get_radius();
        out["rotation_point"] = shape.get_rotation_point();
        out["color"] = shape.get_color();
        out["angle"] = shape.get_angle();
        break;
      }
      case loco_t::shape_type_t::grid: {
        out["shape"] = "grid";
        out["position"] = shape.get_position();
        out["size"] = shape.get_size();
        out["grid_size"] = shape.get_grid_size();
        out["rotation_point"] = shape.get_rotation_point();
        out["color"] = shape.get_color();
        out["angle"] = shape.get_angle();
        break;
      }
      case loco_t::shape_type_t::particles: {
        auto& ri = *(loco_t::particles_t::ri_t*)shape.GetData(gloco->shaper);
        out["shape"] = "particles";
        out["position"] = ri.position;
        out["size"] = ri.size;
        out["color"] = ri.color;
        out["begin_time"] = ri.begin_time;
        out["alive_time"] = ri.alive_time;
        out["respawn_time"] = ri.respawn_time;
        out["count"] = ri.count;
        out["position_velocity"] = ri.position_velocity;
        out["angle_velocity"] = ri.angle_velocity;
        out["begin_angle"] = ri.begin_angle;
        out["end_angle"] = ri.end_angle;
        out["angle"] = ri.angle;
        out["gap_size"] = ri.gap_size;
        out["max_spread_size"] = ri.max_spread_size;
        out["size_velocity"] = ri.size_velocity;
        out["particle_shape"] = ri.shape;
        out["blending"] = ri.blending;
        break;
      }
      default: {
        fan::throw_error("unimplemented shape");
      }
      }
      return false;
    }

    bool json_to_shape(const fan::json& in, loco_t::shape_t* shape) {
      std::string shape_type = in["shape"];
      switch (fan::get_hash(shape_type.c_str())) {
      case fan::get_hash("rectangle"): {
        loco_t::rectangle_t::properties_t p;
        p.position = in["position"];
        p.size = in["size"];
        p.rotation_point = in["rotation_point"];
        p.color = in["color"];
        p.outline_color = in.contains("outline_color") ? in["outline_color"].template get<fan::color>() : p.color;
        p.angle = in["angle"];
        *shape = p;
        break;
      }
      case fan::get_hash("light"): {
        loco_t::light_t::properties_t p;
        p.position = in["position"];
        p.parallax_factor = in["parallax_factor"];
        p.size = in["size"];
        p.rotation_point = in["rotation_point"];
        p.color = in["color"];
        p.flags = in["flags"];
        p.angle = in["angle"];
        *shape = p;
        break;
      }
      case fan::get_hash("line"): {
        loco_t::line_t::properties_t p;
        p.color = in["color"];
        p.src = in["src"];
        p.dst = in["dst"];
        *shape = p;
        break;
      }
      case fan::get_hash("sprite"): {
        loco_t::sprite_t::properties_t p;
        p.blending = true;
        p.position = in["position"];
        p.parallax_factor = in["parallax_factor"];
        p.size = in["size"];
        p.rotation_point = in["rotation_point"];
        p.color = in["color"];
        p.angle = in["angle"];
        p.flags = in["flags"];
        p.tc_position = in["tc_position"];
        p.tc_size = in["tc_size"];
        *shape = p;
        fan::graphics::image_load_properties_t lp;
        if (in.contains("image_visual_output")) {
          lp.visual_output = in["image_visual_output"];
        }
        if (in.contains("image_format")) {
          lp.format = in["image_format"];
        }
        if (in.contains("image_type")) {
          lp.type = in["image_type"];
        }
        if (in.contains("image_min_filter")) {
          lp.min_filter = in["image_min_filter"];
        }
        if (in.contains("image_mag_filter")) {
          lp.mag_filter = in["image_mag_filter"];
        }
        if (in.contains("images") && in["images"].is_array()) {
          for (const auto [i, image_json] : fan::enumerate(in["images"])) {
            if (!image_json.contains("image_path")) continue;

            auto path = image_json["image_path"];
            if (fan::io::file::exists(path)) {
              fan::graphics::image_load_properties_t lp;

              if (image_json.contains("image_visual_output")) lp.visual_output = image_json["image_visual_output"];
              if (image_json.contains("image_format")) lp.format = image_json["image_format"];
              if (image_json.contains("image_type")) lp.type = image_json["image_type"];
              if (image_json.contains("image_min_filter")) lp.min_filter = image_json["image_min_filter"];
              if (image_json.contains("image_mag_filter")) lp.mag_filter = image_json["image_mag_filter"];

              auto image = gloco->image_load(path, lp);

              if (i == 0) {
                shape->set_image(image);
              }
              else {
                auto images = shape->get_images();
                images[i - 1] = image;
                shape->set_images(images);
              }
              gloco->image_list[image].image_path = path;
            }
          }
        }

        break;
      }
      case fan::get_hash("unlit_sprite"): {
        loco_t::unlit_sprite_t::properties_t p;
        p.blending = true;
        p.position = in["position"];
        p.parallax_factor = in["parallax_factor"];
        p.size = in["size"];
        p.rotation_point = in["rotation_point"];
        p.color = in["color"];
        p.angle = in["angle"];
        p.flags = in["flags"];
        p.tc_position = in["tc_position"];
        p.tc_size = in["tc_size"];
        *shape = p;
        fan::graphics::image_load_properties_t lp;
        if (in.contains("image_visual_output")) {
          lp.visual_output = in["image_visual_output"];
        }
        if (in.contains("image_format")) {
          lp.format = in["image_format"];
        }
        if (in.contains("image_type")) {
          lp.type = in["image_type"];
        }
        if (in.contains("image_min_filter")) {
          lp.min_filter = in["image_min_filter"];
        }
        if (in.contains("image_mag_filter")) {
          lp.mag_filter = in["image_mag_filter"];
        }
       
        if (in.contains("images") && in["images"].is_array()) {
          for (const auto [i, image_json] : fan::enumerate(in["images"])) {
            if (!image_json.contains("image_path")) continue;

            auto path = image_json["image_path"];
            if (fan::io::file::exists(path)) {
              fan::graphics::image_load_properties_t lp;

              if (image_json.contains("image_visual_output")) lp.visual_output = image_json["image_visual_output"];
              if (image_json.contains("image_format")) lp.format = image_json["image_format"];
              if (image_json.contains("image_type")) lp.type = image_json["image_type"];
              if (image_json.contains("image_min_filter")) lp.min_filter = image_json["image_min_filter"];
              if (image_json.contains("image_mag_filter")) lp.mag_filter = image_json["image_mag_filter"];

              auto image = gloco->image_load(path, lp);

              if (i == 0) {
                shape->set_image(image);
              }
              else {
                auto images = shape->get_images();
                images[i - 1] = image;
                shape->set_images(images);
              }
              gloco->image_list[image].image_path = path;
            }
          }
        }

        break;
      }
      case fan::get_hash("circle"): {
        loco_t::circle_t::properties_t p;
        p.position = in["position"];
        p.radius = in["radius"];
        p.rotation_point = in["rotation_point"];
        p.color = in["color"];
        p.angle = in["angle"];
        *shape = p;
        break;
      }
      case fan::get_hash("grid"): {
        loco_t::grid_t::properties_t p;
        p.position = in["position"];
        p.size = in["size"];
        p.grid_size = in["grid_size"];
        p.rotation_point = in["rotation_point"];
        p.color = in["color"];
        p.angle = in["angle"];
        *shape = p;
        break;
      }
      case fan::get_hash("particles"): {
        loco_t::particles_t::properties_t p;
        p.position = in["position"];
        p.size = in["size"];
        p.color = in["color"];
        p.begin_time = in["begin_time"];
        p.alive_time = in["alive_time"];
        p.respawn_time = in["respawn_time"];
        p.count = in["count"];
        p.position_velocity = in["position_velocity"];
        p.angle_velocity = in["angle_velocity"];
        p.begin_angle = in["begin_angle"];
        p.end_angle = in["end_angle"];
        p.angle = in["angle"];
        p.gap_size = in["gap_size"];
        p.max_spread_size = in["max_spread_size"];
        p.size_velocity = in["size_velocity"];
        p.shape = in["particle_shape"];
        p.blending = in["blending"];
        *shape = p;
        break;
      }
      default: {
        fan::throw_error("unimplemented shape");
      }
      }
      return false;
    }

    bool shape_serialize(loco_t::shape_t& shape, fan::json* out) {
      return shape_to_json(shape, out);
    }
  }
}

export namespace fan {

  namespace graphics {
    bool shape_to_bin(loco_t::shape_t& shape, std::vector<uint8_t>* data) {
      std::vector<uint8_t>& out = *data;
      fan::write_to_vector(out, shape.get_shape_type());
      fan::write_to_vector(out, shape.gint());
      switch (shape.get_shape_type()) {
      case loco_t::shape_type_t::light: {
        fan::write_to_vector(out, shape.get_position());
        fan::write_to_vector(out, shape.get_parallax_factor());
        fan::write_to_vector(out, shape.get_size());
        fan::write_to_vector(out, shape.get_rotation_point());
        fan::write_to_vector(out, shape.get_color());
        fan::write_to_vector(out, shape.get_flags());
        fan::write_to_vector(out, shape.get_angle());
        break;
      }
      case loco_t::shape_type_t::line: {
        fan::write_to_vector(out, shape.get_color());
        fan::write_to_vector(out, shape.get_src());
        fan::write_to_vector(out, shape.get_dst());
        break;
      case loco_t::shape_type_t::rectangle: {
        fan::write_to_vector(out, shape.get_position());
        fan::write_to_vector(out, shape.get_size());
        fan::write_to_vector(out, shape.get_rotation_point());
        fan::write_to_vector(out, shape.get_color());
        fan::write_to_vector(out, shape.get_angle());
        break;
      }
      case loco_t::shape_type_t::sprite: {
        fan::write_to_vector(out, shape.get_position());
        fan::write_to_vector(out, shape.get_parallax_factor());
        fan::write_to_vector(out, shape.get_size());
        fan::write_to_vector(out, shape.get_rotation_point());
        fan::write_to_vector(out, shape.get_color());
        fan::write_to_vector(out, shape.get_angle());
        fan::write_to_vector(out, shape.get_flags());
        fan::write_to_vector(out, shape.get_image_data().image_path);
        fan::graphics::image_load_properties_t ilp = gloco->image_get_settings(shape.get_image());
        fan::write_to_vector(out, ilp.visual_output);
        fan::write_to_vector(out, ilp.format);
        fan::write_to_vector(out, ilp.type);
        fan::write_to_vector(out, ilp.min_filter);
        fan::write_to_vector(out, ilp.mag_filter);
        fan::write_to_vector(out, shape.get_tc_position());
        fan::write_to_vector(out, shape.get_tc_size());
        break;
      }
      case loco_t::shape_type_t::unlit_sprite: {
        fan::write_to_vector(out, shape.get_position());
        fan::write_to_vector(out, shape.get_parallax_factor());
        fan::write_to_vector(out, shape.get_size());
        fan::write_to_vector(out, shape.get_rotation_point());
        fan::write_to_vector(out, shape.get_color());
        fan::write_to_vector(out, shape.get_angle());
        fan::write_to_vector(out, shape.get_flags());
        fan::write_to_vector(out, shape.get_image_data().image_path);
        fan::graphics::image_load_properties_t ilp = gloco->image_get_settings(shape.get_image());
        fan::write_to_vector(out, ilp.visual_output);
        fan::write_to_vector(out, ilp.format);
        fan::write_to_vector(out, ilp.type);
        fan::write_to_vector(out, ilp.min_filter);
        fan::write_to_vector(out, ilp.mag_filter);
        fan::write_to_vector(out, shape.get_tc_position());
        fan::write_to_vector(out, shape.get_tc_size());
        break;
      }
      case loco_t::shape_type_t::circle: {
        fan::write_to_vector(out, shape.get_position());
        fan::write_to_vector(out, shape.get_radius());
        fan::write_to_vector(out, shape.get_rotation_point());
        fan::write_to_vector(out, shape.get_color());
        fan::write_to_vector(out, shape.get_angle());
        break;
      }
      case loco_t::shape_type_t::grid: {
        fan::write_to_vector(out, shape.get_position());
        fan::write_to_vector(out, shape.get_size());
        fan::write_to_vector(out, shape.get_grid_size());
        fan::write_to_vector(out, shape.get_rotation_point());
        fan::write_to_vector(out, shape.get_color());
        fan::write_to_vector(out, shape.get_angle());
        break;
      }
      case loco_t::shape_type_t::particles: {
        auto& ri = *(loco_t::particles_t::ri_t*)shape.GetData(gloco->shaper);
        fan::write_to_vector(out, ri.position);
        fan::write_to_vector(out, ri.size);
        fan::write_to_vector(out, ri.color);
        fan::write_to_vector(out, ri.begin_time);
        fan::write_to_vector(out, ri.alive_time);
        fan::write_to_vector(out, ri.respawn_time);
        fan::write_to_vector(out, ri.count);
        fan::write_to_vector(out, ri.position_velocity);
        fan::write_to_vector(out, ri.angle_velocity);
        fan::write_to_vector(out, ri.begin_angle);
        fan::write_to_vector(out, ri.end_angle);
        fan::write_to_vector(out, ri.angle);
        fan::write_to_vector(out, ri.gap_size);
        fan::write_to_vector(out, ri.max_spread_size);
        fan::write_to_vector(out, ri.size_velocity);
        fan::write_to_vector(out, ri.shape);
        fan::write_to_vector(out, ri.blending);
        break;
      }
      }
      case loco_t::shape_type_t::light_end: {
        break;
      }
      default: {
        fan::throw_error("unimplemented shape");
      }
      }
      return false;
    }

    bool bin_to_shape(const std::vector<uint8_t>& in, loco_t::shape_t* shape, uint64_t& offset) {
      using sti_t = std::remove_reference_t<decltype(loco_t::shape_t().get_shape_type())>;
      using nr_t = std::remove_reference_t<decltype(loco_t::shape_t().gint())>;
      sti_t shape_type = fan::vector_read_data<sti_t>(in, offset);
      nr_t nri = fan::vector_read_data<nr_t>(in, offset);
      switch (shape_type) {
      case loco_t::shape_type_t::rectangle: {
        loco_t::rectangle_t::properties_t p;
        p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
        p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
        p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
        p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
        p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
        p.outline_color = p.color;
        *shape = p;
        return false;
      }
      case loco_t::shape_type_t::light: {
        loco_t::light_t::properties_t p;
        p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
        p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
        p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
        p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
        p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
        p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
        p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
        *shape = p;
        break;
      }
      case loco_t::shape_type_t::line: {
        loco_t::line_t::properties_t p;
        p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
        p.src = fan::vector_read_data<decltype(p.src)>(in, offset);
        p.dst = fan::vector_read_data<decltype(p.dst)>(in, offset);
        *shape = p;
        break;
      }
      case loco_t::shape_type_t::sprite: {
        loco_t::sprite_t::properties_t p;
        p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
        p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
        p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
        p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
        p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
        p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
        p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);

        std::string image_path = fan::vector_read_data<std::string>(in, offset);
        fan::graphics::image_load_properties_t ilp;
        ilp.visual_output = fan::vector_read_data<decltype(ilp.visual_output)>(in, offset);
        ilp.format = fan::vector_read_data<decltype(ilp.format)>(in, offset);
        ilp.type = fan::vector_read_data<decltype(ilp.type)>(in, offset);
        ilp.min_filter = fan::vector_read_data<decltype(ilp.min_filter)>(in, offset);
        ilp.mag_filter = fan::vector_read_data<decltype(ilp.mag_filter)>(in, offset);
        p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
        p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
        *shape = p;
        if (image_path.size()) {
          shape->get_image_data().image_path = image_path;
          shape->set_image(gloco->image_load(image_path, ilp));
        }
        break;
      }
      case loco_t::shape_type_t::unlit_sprite: {
        loco_t::unlit_sprite_t::properties_t p;
        p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
        p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
        p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
        p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
        p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
        p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
        p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
        std::string image_path = fan::vector_read_data<std::string>(in, offset);
        fan::graphics::image_load_properties_t ilp;
        ilp.visual_output = fan::vector_read_data<decltype(ilp.visual_output)>(in, offset);
        ilp.format = fan::vector_read_data<decltype(ilp.format)>(in, offset);
        ilp.type = fan::vector_read_data<decltype(ilp.type)>(in, offset);
        ilp.min_filter = fan::vector_read_data<decltype(ilp.min_filter)>(in, offset);
        ilp.mag_filter = fan::vector_read_data<decltype(ilp.mag_filter)>(in, offset);
        p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
        p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
        *shape = p;
        if (image_path.size()) {
          shape->get_image_data().image_path = image_path;
          shape->set_image(gloco->image_load(image_path, ilp));
        }
        break;
      }
      case loco_t::shape_type_t::circle: {
        loco_t::circle_t::properties_t p;
        p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
        p.radius = fan::vector_read_data<decltype(p.radius)>(in, offset);
        p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
        p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
        p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
        *shape = p;
        break;
      }
      case loco_t::shape_type_t::grid: {
        loco_t::grid_t::properties_t p;
        p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
        p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
        p.grid_size = fan::vector_read_data<decltype(p.grid_size)>(in, offset);
        p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
        p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
        p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
        *shape = p;
        break;
      }
      case loco_t::shape_type_t::particles: {
        loco_t::particles_t::properties_t p;
        p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
        p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
        p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
        p.begin_time = fan::vector_read_data<decltype(p.begin_time)>(in, offset);
        p.alive_time = fan::vector_read_data<decltype(p.alive_time)>(in, offset);
        p.respawn_time = fan::vector_read_data<decltype(p.respawn_time)>(in, offset);
        p.count = fan::vector_read_data<decltype(p.count)>(in, offset);
        p.position_velocity = fan::vector_read_data<decltype(p.position_velocity)>(in, offset);
        p.angle_velocity = fan::vector_read_data<decltype(p.angle_velocity)>(in, offset);
        p.begin_angle = fan::vector_read_data<decltype(p.begin_angle)>(in, offset);
        p.end_angle = fan::vector_read_data<decltype(p.end_angle)>(in, offset);
        p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
        p.gap_size = fan::vector_read_data<decltype(p.gap_size)>(in, offset);
        p.max_spread_size = fan::vector_read_data<decltype(p.max_spread_size)>(in, offset);
        p.size_velocity = fan::vector_read_data<decltype(p.size_velocity)>(in, offset);
        p.shape = fan::vector_read_data<decltype(p.shape)>(in, offset);
        p.blending = fan::vector_read_data<decltype(p.blending)>(in, offset);
        *shape = p;
        break;
      }
      case loco_t::shape_type_t::light_end: {
        return false;
      }
      default: {
        fan::throw_error("unimplemented");
      }
      }
      if (shape->gint() != nri) {
        fan::throw_error("");
      }
      return false;
    }

    bool shape_serialize(loco_t::shape_t& shape, std::vector<uint8_t>* out) {
      return shape_to_bin(shape, out);
    }

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
      bool iterate(const std::vector<uint8_t>& bin_data, loco_t::shape_t* shape) {
        if (bin_data.empty()) {
          return 0;
        }
        else if (data.offset >= bin_data.size()) {
          return 0;
        }
        bin_to_shape(bin_data, shape, data.offset);
        return 1;
      }
    };
  }
}

#endif

#if defined(fan_json)
loco_t::shape_t::operator fan::json() {
  fan::json out;
  fan::graphics::shape_to_json(*this, &out);
  return out;
}
loco_t::shape_t::operator std::string() {
  fan::json out;
  fan::graphics::shape_to_json(*this, &out);
  return out.dump(2);
}
loco_t::shape_t& loco_t::shape_t::operator=(const fan::json& json) {
  fan::graphics::json_to_shape(json, this);
  return *this;
}
loco_t::shape_t& loco_t::shape_t::operator=(const std::string& json_string) {
  return loco_t::shape_t::operator=(fan::json::parse(json_string));
}
#endif

#include <fan/graphics/collider.h>

//vk

#if defined(fan_vulkan)
#include <fan/graphics/vulkan/uniform_block.h>
#include <fan/graphics/vulkan/memory.h>
#endif

#if defined(loco_audio)
export namespace fan {
  namespace audio {
    using piece_t = fan::audio_t::piece_t;

    fan::audio_t::piece_t open_piece(const std::string& path, fan::audio_t::PieceFlag::t flags = 0) {
      fan::audio_t::piece_t piece;
      sint32_t err = gloco->audio.Open(&piece, path, flags);
      if (err != 0) {
        fan::throw_error("failed to open piece:", err);
      }
      return piece;
    }
    /// <summary>
    /// Function checks if the stored pointer equals to nullptr. Does NOT check for actual validity.
    /// </summary>
    /// <param name="piece">Given piece to validate.</param>
    /// <returns></returns>
    bool is_piece_valid(fan::audio_t::piece_t piece) {
      char test_block[sizeof(piece)];
      memset(test_block, 0, sizeof(piece));
      return memcmp(&piece, test_block, sizeof(piece));
    }

    fan::audio_t::SoundPlayID_t play(fan::audio_t::piece_t piece, uint32_t group_id = 0, bool loop = false) {
      fan::audio_t::PropertiesSoundPlay_t p{};
      p.Flags.Loop = loop;
      p.GroupID = 0;
      return gloco->audio.SoundPlay(&piece, &p);
    }
    void stop(fan::audio_t::SoundPlayID_t id) {
      fan::audio_t::PropertiesSoundStop_t p{};
      p.FadeOutTo = 0;
      gloco->audio.SoundStop(id, &p);
    }
    void resume(uint32_t group_id = 0) {
      gloco->audio.Resume();
    }
    void pause(uint32_t group_id = 0) {
      gloco->audio.Pause();
    }
    f32_t get_volume() {
      return gloco->audio.GetVolume();
    }

    void set_volume(f32_t volume) {
      gloco->audio.SetVolume(volume);
    }
  }
#if defined(fan_gui)
  namespace graphics {
    using texture_packe0 = loco_t::texture_packe0;
    using ti_t = loco_t::ti_t;
  }
#endif
}
#endif

#if defined(fan_gui)
namespace fan {
  namespace graphics {
    namespace gui {
      void process_loop() {
        auto it = gloco->gui_draw_cb.GetNodeFirst();
        while (it != gloco->gui_draw_cb.dst) {
          gloco->gui_draw_cb.StartSafeNext(it);
          gloco->gui_draw_cb[it]();
          it = gloco->gui_draw_cb.EndSafeNext();
        }
      }
      // fan_track_allocations() must be called in global scope before calling this function
      void render_allocations_plot() {
#if defined(fan_std23)
        static std::vector<f32_t> allocation_sizes;
        static std::vector<fan::heap_profiler_t::memory_data_t> allocations;

        allocation_sizes.clear();
        allocations.clear();


        f32_t max_y = 0;
        for (const auto& entry : fan::heap_profiler_t::instance().memory_map) {
          f32_t v = (f32_t)entry.second.n / (1024 * 1024);
          if (v < 0.001) {
            continue;
          }
          allocation_sizes.push_back(v);
          max_y = std::max(max_y, v);
          allocations.push_back(entry.second);
        }
        static std::stacktrace stack;
        if (ImPlot::BeginPlot("Memory Allocations", ImGui::GetWindowSize(), ImPlotFlags_NoFrame | ImPlotFlags_NoLegend)) {
          float max_allocation = *std::max_element(allocation_sizes.begin(), allocation_sizes.end());
          ImPlot::SetupAxis(ImAxis_Y1, "Memory (MB)");
          ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_y);
          ImPlot::SetupAxis(ImAxis_X1, "Allocations");
          ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(allocation_sizes.size()));

          ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
          ImPlot::PlotBars("Allocations", allocation_sizes.data(), allocation_sizes.size());
          //if (ImPlot::IsPlotHovered()) {
          //  fan::print("A");
          //}
          ImPlot::PopStyleVar();

          bool hovered = false;
          if (ImPlot::IsPlotHovered()) {
            ImPlotPoint mouse = ImPlot::GetPlotMousePos();
            f32_t half_width = 0.25;
            //mouse.x             = ImPlot::RoundTime(ImPlotTime::FromDouble(mouse.x), ImPlotTimeUnit_Day).ToDouble();
            mouse.x = (int)mouse.x;
            float  tool_l = ImPlot::PlotToPixels(mouse.x - half_width * 1.5, mouse.y).x;
            float  tool_r = ImPlot::PlotToPixels(mouse.x + half_width * 1.5, mouse.y).x;
            float  tool_t = ImPlot::GetPlotPos().y;
            float  tool_b = tool_t + ImPlot::GetPlotSize().y;
            ImPlot::PushPlotClipRect();
            auto draw_list = ImGui::GetWindowDrawList();
            draw_list->AddRectFilled(ImVec2(tool_l, tool_t), ImVec2(tool_r, tool_b), IM_COL32(128, 128, 128, 64));
            ImPlot::PopPlotClipRect();

            if (mouse.x >= 0 && mouse.x < allocation_sizes.size()) {
              if (ImGui::IsMouseClicked(0)) {
                ImGui::OpenPopup("view stack");
              }
              stack = allocations[(int)mouse.x].line_data;
              hovered = true;
            }
          }
          if (hovered) {
            ImGui::BeginTooltip();
            std::ostringstream oss;
            oss << stack;
            std::string stack_str = oss.str();
            std::string final_str;
            std::size_t pos = 0;
            while (true) {
              auto end = stack_str.find(')', pos);
              if (end != std::string::npos) {
                end += 1;
                auto begin = stack_str.rfind('\\', end);
                if (begin != std::string::npos) {
                  begin += 1;
                  final_str += stack_str.substr(begin, end - begin);
                  final_str += "\n";
                  pos = end + 1;
                }
                else {
                  break;
                }
              }
              else {
                break;
              }
            }
            ImGui::TextUnformatted(final_str.c_str());
            ImGui::EndTooltip();
          }
          if (ImGui::BeginPopup("view stack", ImGuiWindowFlags_AlwaysHorizontalScrollbar)) {
            std::ostringstream oss;
            oss << stack;
            ImGui::TextUnformatted(oss.str().c_str());
            ImGui::EndPopup();
          }
          ImPlot::EndPlot();
        }

#else
        ImGui::Text("std::stacktrace not supported");
#endif
      }
    }
  }
}
#endif

inline uint32_t fan::graphics::get_draw_mode(uint8_t internal_draw_mode) {
  if (gloco->get_renderer() == loco_t::renderer_t::opengl) {
#if defined(loco_opengl)
    return fan::opengl::core::get_draw_mode(internal_draw_mode);
#endif
  }
  else if (gloco->get_renderer() == loco_t::renderer_t::vulkan) {
#if defined(fan_vulkan)
    return fan::vulkan::core::get_draw_mode(internal_draw_mode);
#endif
  }
#if fan_debug >= fan_debug_medium
  fan::throw_error("invalid get");
#endif
  return -1;
}