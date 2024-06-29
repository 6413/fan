#pragma once

#include <fan/graphics/loco_settings.h>

#define loco_opengl
#define loco_framebuffer
#define loco_post_process
#define loco_vfi
#define loco_physics

#include <fan/window/window.h>
#include <fan/graphics/opengl/gl_core.h>
#include <fan/io/file.h>

#if defined(loco_imgui)
#include <fan/imgui/imgui.h>
#include <fan/imgui/imgui_impl_opengl3.h>
#include <fan/imgui/imgui_impl_glfw.h>
#include <fan/imgui/imgui_neo_sequencer.h>
#include <fan/imgui/imgui-combo-filter.h>
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

  //template <> struct adl_serializer<fan::vec2> { static void to_json(json& j, const fan::vec2& v) { j = json{v.x, v.y}; } };
  //template <> struct adl_serializer<fan::vec3> { static void to_json(json& j, const fan::vec3& v) { j = json{v.x, v.y, v.z}; } };
  //template <> struct adl_serializer<fan::vec4> { static void to_json(json& j, const fan::vec4& v) { j = json{v.x, v.y, v.z, v.w}; } };
  //template <> struct adl_serializer<fan::color> { static void to_json(json& j, const fan::color& c) { j = json{c.r, c.g, c.b, c.a}; } };

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
#endif

#include <fan/tp/tp0.h>

#define loco_line
#define loco_rectangle
#define loco_sprite
#define loco_light
#define loco_circle
#define loco_letter
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
inline thread_local global_loco_t gloco;

namespace fan {
  void printcl(auto&&... values);
  void printclh(int highlight = 0, auto&&... values);
}

#if defined(loco_letter)
#include <fan/graphics/font.h>
#endif


struct loco_t : fan::opengl::context_t {

#define WITCH_LIBC 1
static uint8_t* A_resize(void* ptr, uintptr_t size) {
#if WITCH_LIBC
  if (ptr) {
    if (size) {
      void* rptr = (void*)realloc(ptr, size);
      if (rptr == 0) {
        fan::throw_error_impl();
      }
      return (uint8_t*)rptr;
    }
    else {
      free(ptr);
      return 0;
    }
  }
  else {
    if (size) {
      void* rptr = (void*)malloc(size);
      if (rptr == 0) {
        fan::throw_error_impl();
      }
      return (uint8_t*)rptr;
    }
    else {
      return 0;
    }
  }
#endif
}

  static constexpr uint32_t MaxElementPerBlock = 0x1000;

  struct shape_gl_init_t {
    uint32_t index;
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
    uintptr_t LastKeyOffset = s - (sizeof(Ts), ...) - 1;
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
      letter,
      circle,
      grid,
      vfi,
      particles,
      universal_image_renderer,
      gradient,
      light_end,
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

  using get_position_cb = fan::vec3 (*)(shape_t*);
  using get_size_cb = fan::vec2 (*)(shape_t*);

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

  struct functions_t {
    push_back_cb push_back;

    get_position_cb get_position;
    set_position2_cb set_position2;
    set_position3_cb set_position3;

    get_size_cb get_size;
    set_size_cb set_size;

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
  };

  template <typename T, typename T2>
  static T2& get_render_data(shape_t* shape, T2 T::* attribute) {
    shaper_t::ShapeRenderData_t* data = gloco->shaper.GetRenderData(*shape);
    return ((T*)data)->*attribute;
  }

  template <typename T, typename T2, typename T3>
  static void modify_render_data_element(shape_t* shape, T2 T::* attribute, const T3& value) {
    shaper_t::ShapeRenderData_t* data = gloco->shaper.GetRenderData(*shape);
    ((T*)data)->*attribute = value;
    gloco->shaper.ElementIsPartiallyEdited(
      gloco->shaper.GetSTI(*shape),
      gloco->shaper.GetBLID(*shape),
      gloco->shaper.GetElementIndex(*shape),
      fan::member_offset(attribute),
      sizeof(T3)
    );
  };

  template <typename T>
  static functions_t get_functions() {
    functions_t funcs{
      //.push_back = [](void* data) {
      //  // Implement push_back function
      //},
      .get_position = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, position)) {
          return get_render_data(shape, &T::position);
        }
        else {
          fan::throw_error("unimplemented get - for line use get_src()");
          return fan::vec3();
        }
      },
      .set_position2 = [](shape_t* shape, const fan::vec2& position) {
        if constexpr (fan_has_variable(T, position)) {
          modify_render_data_element(shape, &T::position, position);
        }
        else {
          fan::throw_error("unimplemented set - for line use set_src()");
        }
      },
      .set_position3 = [](shape_t* shape, const fan::vec3& position) {
        if constexpr (fan_has_variable(T, position)) {
            auto sti = gloco->shaper.GetSTI(*shape);

            // alloc can be avoided inside switch
            auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
            uint8_t* KeyPack = new uint8_t[KeyPackSize];
            gloco->shaper.WriteKeys(*shape, KeyPack);
            

            switch (sti) {       
            case loco_t::shape_type_t::light: {
              break;
            }
            // common
            case loco_t::shape_type_t::gradient:
            case loco_t::shape_type_t::grid:
            case loco_t::shape_type_t::circle:
            case loco_t::shape_type_t::letter:
            case loco_t::shape_type_t::rectangle:
            case loco_t::shape_type_t::line: {
              auto o = gloco->shaper.GetKeyOffset(
                offsetof(kps_t::_common_t, depth),
                offsetof(kps_t::common_t, depth)
              );
              *(depth_t*)&KeyPack[o] = position.z;
              break;
            }
                                           // texture
            case loco_t::shape_type_t::particles:
            case loco_t::shape_type_t::universal_image_renderer:
            case loco_t::shape_type_t::unlit_sprite:
            case loco_t::shape_type_t::sprite: {
              auto o = gloco->shaper.GetKeyOffset(
                offsetof(kps_t::_texture_t, depth),
                offsetof(kps_t::texture_t, depth)
              );
              *(depth_t*)&KeyPack[o] = position.z;
              break;
            }
            default: {
              fan::throw_error("unimplemented");
            }
            }

    
            auto _vi = gloco->shaper.GetRenderData(*shape);
            auto vlen = gloco->shaper.GetRenderDataSize(sti);
            uint8_t* vi = new uint8_t[vlen];
            std::memcpy(vi, _vi, vlen);
            ((T*)vi)->position = position;

            auto _ri = gloco->shaper.GetData(*shape);
            auto rlen = gloco->shaper.GetDataSize(sti);
            uint8_t* ri = new uint8_t[rlen];
            std::memcpy(ri, _ri, rlen);

            shape->remove();
            *shape = gloco->shaper.add(
              sti,
              KeyPack,
              KeyPackSize,
              vi,
              ri
            );
#if defined(debug_shape_t)
            fan::print("+", shape->NRI);
#endif
            delete[] KeyPack;
            delete[] vi;
            delete[] ri;
        }
        else {
          fan::throw_error("unimplemented set - for line use set_src()");
        }
        },
        .get_size = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, size)) {
            return get_render_data(shape, &T::size);
          }
          else if constexpr (fan_has_variable(T, radius)) {
            return fan::vec2(get_render_data(shape, &T::radius));
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec2();
          }
        },
        .set_size = [](shape_t* shape, const fan::vec2& size) {
          if constexpr (fan_has_variable(T, size)) {
            modify_render_data_element(shape, &T::size, size);
          }
          else if constexpr (fan_has_variable(T, radius)) {
            modify_render_data_element(shape, &T::radius, size.x);
          }
          else {
            fan::throw_error("unimplemented get");
          }
        },
        .get_rotation_point = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, rotation_point)) {
            return get_render_data(shape, &T::rotation_point);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec2();
          }
        },
        .set_rotation_point = [](shape_t* shape, const fan::vec2& rotation_point) {
          if constexpr (fan_has_variable(T, rotation_point)) {
            modify_render_data_element(shape, &T::rotation_point, rotation_point);
          }
          else {
            fan::throw_error("unimplemented set");
          }
              },
        .get_color = [](shape_t* shape) -> fan::color{
          if constexpr (fan_has_variable(T, color)) {
            return *(fan::color*)&get_render_data(shape, &T::color);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::color();
          }
        },
        .set_color = [](shape_t* shape, const fan::color& color) {
          if constexpr (fan_has_variable(T, color)) {
            if constexpr (!std::is_same_v<T, loco_t::gradient_t::vi_t>) {
              modify_render_data_element(shape, &T::color, color);
            }
          }
          else {
            fan::throw_error("unimplemented set");
          }
        },
        .get_angle = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, angle)) {
            return get_render_data(shape, &T::angle);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec3();
          }
        },
        .set_angle = [](shape_t* shape, const fan::vec3& angle) {
          if constexpr (fan_has_variable(T, angle)) {
            modify_render_data_element(shape, &T::angle, angle);
          }
          else {
            fan::throw_error("unimplemented set");
          }
        },
        .get_tc_position = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, tc_position)) {
            return get_render_data(shape, &T::tc_position);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec2();
          }
        },
        .set_tc_position = [](shape_t* shape, const fan::vec2& tc_position) {
          if constexpr (fan_has_variable(T, tc_position)) {
            modify_render_data_element(shape, &T::tc_position, tc_position);
          }
          else {
            fan::throw_error("unimplemented set");
          }
        },
        .get_tc_size = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, tc_size)) {
            return get_render_data(shape, &T::tc_size);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec2();
          }
        },
        .set_tc_size = [](shape_t* shape, const fan::vec2& tc_size) {
          if constexpr (fan_has_variable(T, tc_size)) {
            modify_render_data_element(shape, &T::tc_size, tc_size);
          }
          else {
            fan::throw_error("unimplemented set");
          }
        },
        .load_tp = [](shape_t* shape, loco_t::texturepack_t::ti_t* ti) -> bool {
          if constexpr(std::is_same_v<T, loco_t::sprite_t::vi_t> ||
          std::is_same_v<T, loco_t::unlit_sprite_t::vi_t>) {
            auto sti = gloco->shaper.GetSTI(*shape);
            
            auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
            uint8_t* KeyPack = new uint8_t[KeyPackSize];
            gloco->shaper.WriteKeys(*shape, KeyPack);
            switch (sti) {
              // texture
              case loco_t::shape_type_t::particles:
              case loco_t::shape_type_t::universal_image_renderer:
              case loco_t::shape_type_t::unlit_sprite:
              case loco_t::shape_type_t::sprite: {
                auto o = gloco->shaper.GetKeyOffset(
                  offsetof(kps_t::_texture_t, image),
                  offsetof(kps_t::texture_t, image)
                );
                *(loco_t::image_t*)&KeyPack[o] = *ti->image;
                break;
              }
              default: {
                fan::throw_error("unimplemented");
              }
            }

            auto& im = *ti->image;
            auto& img = gloco->image_get_data(im);

            auto _vi = gloco->shaper.GetRenderData(*shape);
            auto vlen = gloco->shaper.GetRenderDataSize(sti);
            uint8_t* vi = new uint8_t[vlen];
            std::memcpy(vi, _vi, vlen);
            ((T*)vi)->tc_position = ti->position / img.size;
            ((T*)vi)->tc_size = ti->size / img.size;

            auto _ri = gloco->shaper.GetData(*shape);
            auto rlen = gloco->shaper.GetDataSize(sti);
            uint8_t* ri = new uint8_t[rlen];
            std::memcpy(ri, _ri, rlen);

            shape->remove();
            *shape = gloco->shaper.add(
              sti,
              KeyPack,
              KeyPackSize,
              vi,
              ri
            );
#if defined(debug_shape_t)
            fan::print("+", shape->NRI);
#endif
            delete[] KeyPack;
            delete[] vi;
            delete[] ri;
            }
          return 0;
        },
        .get_grid_size = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, grid_size)) {
            return get_render_data(shape, &T::grid_size);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec2();
          }
        },
        .set_grid_size = [](shape_t* shape, const fan::vec2& grid_size) {
          if constexpr (fan_has_variable(T, grid_size)) {
            modify_render_data_element(shape, &T::grid_size, grid_size);
          }
          else {
            fan::throw_error("unimplemented set");
          }
        },
        .get_camera = [](shape_t* shape) {
          auto sti = gloco->shaper.GetSTI(*shape);

          // alloc can be avoided inside switch
          uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);

          switch (sti) {
            // light
          case loco_t::shape_type_t::light: {
            auto o = gloco->shaper.GetKeyOffset(
              offsetof(kps_t::_light_t, camera),
              offsetof(kps_t::light_t, camera)
            );
            return *(camera_t*)&KeyPack[o];
          }
                                          // common
          case loco_t::shape_type_t::gradient:
          case loco_t::shape_type_t::grid:
          case loco_t::shape_type_t::circle:
          case loco_t::shape_type_t::letter:
          case loco_t::shape_type_t::rectangle:
          case loco_t::shape_type_t::line: {
            auto o = gloco->shaper.GetKeyOffset(
              offsetof(kps_t::_common_t, camera),
              offsetof(kps_t::common_t, camera)
            );
            return *(camera_t*)&KeyPack[o];
          }
                                         // texture
          case loco_t::shape_type_t::particles:
          case loco_t::shape_type_t::universal_image_renderer:
          case loco_t::shape_type_t::unlit_sprite:
          case loco_t::shape_type_t::sprite: {
            auto o = gloco->shaper.GetKeyOffset(
              offsetof(kps_t::_texture_t, camera),
              offsetof(kps_t::texture_t, camera)
            );
            return *(camera_t*)&KeyPack[o];
          }
          default: {
            fan::throw_error("unimplemented");
          }
          }
          return loco_t::camera_t();
        },
        .set_camera = [](shape_t* shape, loco_t::camera_t camera) {
          {
             auto sti = gloco->shaper.GetSTI(*shape);

            // alloc can be avoided inside switch
            auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
            uint8_t* KeyPack = new uint8_t[KeyPackSize];
            gloco->shaper.WriteKeys(*shape, KeyPack);

            switch(sti) {
              // light
              case loco_t::shape_type_t::light: {
                auto o = gloco->shaper.GetKeyOffset(
                  offsetof(kps_t::_light_t, camera),
                  offsetof(kps_t::light_t, camera)
                );
                *(camera_t*)&KeyPack[o] = camera;
                break;
              }
              // common
              case loco_t::shape_type_t::gradient:
              case loco_t::shape_type_t::grid:
              case loco_t::shape_type_t::circle:
              case loco_t::shape_type_t::letter:
              case loco_t::shape_type_t::rectangle:
              case loco_t::shape_type_t::line: {
                auto o = gloco->shaper.GetKeyOffset(
                  offsetof(kps_t::_common_t, camera),
                  offsetof(kps_t::common_t, camera)
                );
                *(camera_t*)&KeyPack[o] = camera;
                break;
              }
              // texture
              case loco_t::shape_type_t::particles:
              case loco_t::shape_type_t::universal_image_renderer:
              case loco_t::shape_type_t::unlit_sprite:
              case loco_t::shape_type_t::sprite: {
                auto o = gloco->shaper.GetKeyOffset(
                  offsetof(kps_t::_texture_t, camera),
                  offsetof(kps_t::texture_t, camera)
                );
                *(camera_t*)&KeyPack[o] = camera;
                break;
              }
              default: {
                fan::throw_error("unimplemented");
              }
            }

            auto _vi = gloco->shaper.GetRenderData(*shape);
            auto vlen = gloco->shaper.GetRenderDataSize(sti);
            uint8_t* vi = new uint8_t[vlen];
            std::memcpy(vi, _vi, vlen);

            auto _ri = gloco->shaper.GetData(*shape);
            auto rlen = gloco->shaper.GetDataSize(sti);
            uint8_t* ri = new uint8_t[rlen];
            std::memcpy(ri, _ri, rlen);

            shape->remove();
            *shape = gloco->shaper.add(
              sti,
              KeyPack,
              KeyPackSize,
              vi,
              ri
            );
#if defined(debug_shape_t)
            fan::print("+", shape->NRI);
#endif
            delete[] KeyPack;
            delete[] vi;
            delete[] ri;
          }
        },
        .get_viewport = [](shape_t* shape) {
          uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);

          auto sti = gloco->shaper.GetSTI(*shape);

          switch(sti) {
            // light
            case loco_t::shape_type_t::light: {
              auto o = gloco->shaper.GetKeyOffset(
                offsetof(kps_t::_light_t, viewport),
                offsetof(kps_t::light_t, viewport)
              );
              return *(viewport_t*)&KeyPack[o];
            }
            // common
            case loco_t::shape_type_t::gradient:
            case loco_t::shape_type_t::grid:
            case loco_t::shape_type_t::circle:
            case loco_t::shape_type_t::letter:
            case loco_t::shape_type_t::rectangle:
            case loco_t::shape_type_t::line: {
              auto o = gloco->shaper.GetKeyOffset(
                offsetof(kps_t::_common_t, viewport),
                offsetof(kps_t::common_t, viewport)
              );
              return *(viewport_t*)&KeyPack[o];
            }
            // texture
            case loco_t::shape_type_t::particles:
            case loco_t::shape_type_t::universal_image_renderer:
            case loco_t::shape_type_t::unlit_sprite:
            case loco_t::shape_type_t::sprite: {
              auto o = gloco->shaper.GetKeyOffset(
                offsetof(kps_t::_texture_t, viewport),
                offsetof(kps_t::texture_t, viewport)
              );
              return *(viewport_t*)&KeyPack[o];
            }
            default: {
              fan::throw_error("unimplemented");
            }
          }
          return loco_t::viewport_t();
        },
        .set_viewport = [](shape_t* shape, loco_t::viewport_t viewport) {
          {
            auto sti = gloco->shaper.GetSTI(*shape);

            // alloc can be avoided inside switch
            auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
            uint8_t* KeyPack = new uint8_t[KeyPackSize];
            gloco->shaper.WriteKeys(*shape, KeyPack);
            
            switch(sti) {
              // light
              case loco_t::shape_type_t::light: {
                auto o = gloco->shaper.GetKeyOffset(
                  offsetof(kps_t::_light_t, viewport),
                  offsetof(kps_t::light_t, viewport)
                );
                *(viewport_t*)&KeyPack[o] = viewport;
                break;
              }
              // common
              case loco_t::shape_type_t::gradient:
              case loco_t::shape_type_t::grid:
              case loco_t::shape_type_t::circle:
              case loco_t::shape_type_t::letter:
              case loco_t::shape_type_t::rectangle:
              case loco_t::shape_type_t::line: {
                auto o = gloco->shaper.GetKeyOffset(
                  offsetof(kps_t::_common_t, viewport),
                  offsetof(kps_t::common_t, viewport)
                );
                *(viewport_t*)&KeyPack[o] = viewport;
                break;
              }
              // texture
              case loco_t::shape_type_t::particles:
              case loco_t::shape_type_t::universal_image_renderer:
              case loco_t::shape_type_t::unlit_sprite:
              case loco_t::shape_type_t::sprite: {
                auto o = gloco->shaper.GetKeyOffset(
                  offsetof(kps_t::_texture_t, viewport),
                  offsetof(kps_t::texture_t, viewport)
                );
                *(viewport_t*)&KeyPack[o] = viewport;
                break;
              }
              default: {
                fan::throw_error("unimplemented");
              }
            }

            auto _vi = gloco->shaper.GetRenderData(*shape);
            auto vlen = gloco->shaper.GetRenderDataSize(sti);
            uint8_t* vi = new uint8_t[vlen];
            std::memcpy(vi, _vi, vlen);

            auto _ri = gloco->shaper.GetData(*shape);
            auto rlen = gloco->shaper.GetDataSize(sti);
            uint8_t* ri = new uint8_t[rlen];
            std::memcpy(ri, _ri, rlen);

            shape->remove();
            *shape = gloco->shaper.add(
              sti,
              KeyPack,
              KeyPackSize,
              vi,
              ri
            );
#if defined(debug_shape_t)
            fan::print("+", shape->NRI);
#endif
            delete[] KeyPack;
            delete[] vi;
            delete[] ri;
          }
        },

        .get_image = [](shape_t* shape) -> loco_t::image_t {
          auto sti = gloco->shaper.GetSTI(*shape);
          auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
          uint8_t* KeyPack = new uint8_t[KeyPackSize];
          gloco->shaper.WriteKeys(*shape, KeyPack);
          switch (sti) {
          // texture
          case loco_t::shape_type_t::particles:
          case loco_t::shape_type_t::universal_image_renderer:
          case loco_t::shape_type_t::unlit_sprite:
          case loco_t::shape_type_t::sprite: {
            auto o = gloco->shaper.GetKeyOffset(
              offsetof(kps_t::_texture_t, image),
              offsetof(kps_t::texture_t, image)
            );
            loco_t::image_t image = *(loco_t::image_t*)&KeyPack[o];
            delete[] KeyPack;
            return image;
          }
          default: {
            fan::throw_error("unimplemented");
          }
          }
          return loco_t::image_t();
        },
        .set_image = [](shape_t* shape, loco_t::image_t image) {
         
          auto sti = gloco->shaper.GetSTI(*shape);

          // alloc can be avoided inside switch
          auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
          uint8_t* KeyPack = new uint8_t[KeyPackSize];
          gloco->shaper.WriteKeys(*shape, KeyPack);

          switch (sti) {
          // texture
          case loco_t::shape_type_t::particles:
          case loco_t::shape_type_t::universal_image_renderer:
          case loco_t::shape_type_t::unlit_sprite:
          case loco_t::shape_type_t::sprite: {
            auto o = gloco->shaper.GetKeyOffset(
              offsetof(kps_t::_texture_t, image),
              offsetof(kps_t::texture_t, image)
            );
            *(loco_t::image_t*)&KeyPack[o] = image;
            break;
          }
          default: {
            fan::throw_error("unimplemented");
          }
          }
            
          auto _vi = gloco->shaper.GetRenderData(*shape);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);

          auto _ri = gloco->shaper.GetData(*shape);
          auto rlen = gloco->shaper.GetDataSize(sti);
          uint8_t* ri = new uint8_t[rlen];
          std::memcpy(ri, _ri, rlen);

          shape->remove();
          *shape = gloco->shaper.add(
            sti,
            KeyPack,
            KeyPackSize,
            vi,
            ri
          );
#if defined(debug_shape_t)
          fan::print("+", shape->NRI);
#endif
          delete[] KeyPack;
          delete[] vi;
          delete[] ri;
        },
        .get_parallax_factor = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, parallax_factor)) {
            return get_render_data(shape, &T::parallax_factor);
          }
          else {
            fan::throw_error("unimplemented get");
            return 0.0f;
          }
        },
        .set_parallax_factor = [](shape_t* shape, f32_t parallax_factor) {
          if constexpr (fan_has_variable(T, parallax_factor)) {
            modify_render_data_element(shape, &T::parallax_factor, parallax_factor);
          }
          else {
            fan::throw_error("unimplemented set");
          }
        },
        .get_rotation_vector = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, rotation_vector)) {
            return get_render_data(shape, &T::rotation_vector);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec3();
          }
        },
        .get_flags = [](shape_t* shape) -> uint32_t {
          if constexpr (fan_has_variable(T, flags)) {
            return get_render_data(shape, &T::flags);
          }
          else {
            fan::throw_error("unimplemented get");
            return 0;
          }
        },
        .set_flags = [](shape_t* shape, uint32_t flags) {
          if constexpr (fan_has_variable(T, flags)) {
            modify_render_data_element(shape, &T::flags, flags);
          }
          else {
            fan::throw_error("unimplemented set");
          }
        },
        .get_radius = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, radius)) {
            return get_render_data(shape, &T::radius);
          }
          else {
            fan::throw_error("unimplemented get");
            return 0.0f;
          }
        },
        .get_src = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, src)) {
            return get_render_data(shape, &T::src);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec3();
          }
        },
        .get_dst = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, dst)) {
            return get_render_data(shape, &T::dst);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec3();
          }
        },
        .get_outline_size = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, outline_size)) {
            return get_render_data(shape, &T::outline_size);
          }
          else {
            fan::throw_error("unimplemented get");
            return 0.0f;
          }
        },
        .get_outline_color = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, outline_color)) {
            return get_render_data(shape, &T::outline_color);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::color();
          }
        },
        .reload = [](shape_t* shape, uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter) {
          if (shape->get_shape_type() != loco_t::shape_type_t::universal_image_renderer) {
            fan::throw_error("only meant to be used with universal_image_renderer");
          }
          loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)gloco->shaper.GetData(*shape);
          if (format != ri.format) {
            auto sti = gloco->shaper.GetSTI(*shape);
            auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
            uint8_t* KeyPack = new uint8_t[KeyPackSize];
            gloco->shaper.WriteKeys(*shape, KeyPack);
            auto o = gloco->shaper.GetKeyOffset(
              offsetof(kps_t::_texture_t, camera),
              offsetof(kps_t::texture_t, camera)
            );

            loco_t::image_t vi_image = *(loco_t::image_t*)&KeyPack[o];

            delete[] KeyPack;


            auto shader = gloco->shaper.GetShader(sti);
            gloco->shader_set_vertex(
              shader,
              read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
            );
            {
              fan::string fs;
              switch(format) {
                case fan::pixel_format::yuv420p: {
                  fs = read_shader("shaders/opengl/2D/objects/yuv420p.fs");
                  break;
                }
                case fan::pixel_format::nv12: {
                  fs = read_shader("shaders/opengl/2D/objects/nv12.fs");
                  break;
                }
                default: {
                  fan::throw_error("unimplemented format");
                }
              }
              gloco->shader_set_fragment(shader, fs);
              gloco->shader_compile(shader);
            }

            uint8_t image_count_old = fan::pixel_format::get_texture_amount(ri.format);
            uint8_t image_count_new = fan::pixel_format::get_texture_amount(format);
            if (image_count_new < image_count_old) {
              for (uint32_t i = image_count_old; i > image_count_new; --i) {
                if (i == 0) {
                  gloco->image_erase(vi_image);
                }
                else {
                  gloco->image_erase(ri.images_rest[i - 1]);
                }
              }
            }
            else if (image_count_new > image_count_old) {
              loco_t::image_t images[4];
              for (uint32_t i = image_count_old; i < image_count_new; ++i) {
                images[i] = gloco->image_create();
              }
              shape->set_image(images[0]);
              std::memcpy(ri.images_rest, &images[1], sizeof(ri.images_rest));
            }
          }

          auto vi_image = shape->get_image();

          uint8_t image_count_new = fan::pixel_format::get_texture_amount(format);
          for (uint32_t i = 0; i < image_count_new; i++) {
            fan::webp::image_info_t image_info;
            image_info.data = image_data[i];
            image_info.size = fan::pixel_format::get_image_sizes(format, image_size)[i];
            auto lp = fan::pixel_format::get_image_properties<loco_t::image_load_properties_t>(format)[i];
            lp.min_filter = filter;
            lp.mag_filter = filter;
            if (i == 0) {
              gloco->image_reload_pixels(
                vi_image,
                image_info,
                lp
              );
            }
            else {
              gloco->image_reload_pixels(
                ri.images_rest[i - 1],
                image_info,
                lp
              );
            }
          }
          ri.format = format;
        },
        .draw = [](uint8_t draw_range) {
          // Implement draw function
        },
        .set_line = [](shape_t* shape, const fan::vec2& src, const fan::vec2& dst) {
          if constexpr (fan_has_variable(T, src) && fan_has_variable(T, dst)) {
            modify_render_data_element(shape, &T::src, src);
            modify_render_data_element(shape, &T::dst, dst);
          }
          else {
            fan::throw_error("unimplemented set");
          }
        }
      };
    return funcs;
  }

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

  struct properties_t {
    bool vsync = true;
    fan::vec2 window_size = -1;
    uint64_t window_flags = 0;
  };

  uint64_t start_time = fan::time::clock::now();

  void init_framebuffer();

  loco_t();
  loco_t(const properties_t& p);

  void process_frame();

  bool process_loop(const fan::function_t<void()>& lambda = [] {});
  void loop(const fan::function_t<void()>& lambda);

  loco_t::camera_t open_camera(const fan::vec2 & x, const fan::vec2 & y);
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

  uint32_t get_fps();
  void set_vsync(bool flag);

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

#define fan_imgui_dragfloat_named(name, variable, speed, m_min, m_max) \
  [&] <typename T5>(T5& var) -> bool{ \
    if constexpr(std::is_same_v<f32_t, T5>)  { \
      return ImGui::DragFloat(fan::string(std::move(name)).c_str(), &var, (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec2, T5>)  { \
      return ImGui::DragFloat2(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec3, T5>)  { \
      return ImGui::DragFloat3(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec4, T5>)  { \
      return ImGui::DragFloat4(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::color, T5>)  { \
      return ImGui::DragFloat4(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else {\
      fan::throw_error_impl(); \
      return 0;\
    } \
  }(variable)

#define fan_imgui_dragfloat(variable, speed, m_min, m_max) \
    fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, m_min, m_max)


#define fan_imgui_dragfloat1(variable, speed) \
    fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, -1, -1)

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
    ) {
      //fan::vec_wrap_t < sizeof(T) / fan::conditional_value_t < std::is_class_v<T>, sizeof(T{} [0] ), sizeof(T) > , f32_t > initial = initial_;
      fan::vec_wrap_t<fan::conditional_value_t<std::is_arithmetic_v<T>, 1, sizeof(T) / sizeof(f32_t)>::value, f32_t> 
        initial;
      if constexpr (std::is_arithmetic_v<T>) {
        initial = (f32_t)initial_;
      }
      else {
        initial = initial_;
      }
        fan::opengl::context_t::shader_t& shader = gloco->shader_get(shader_nr);
        auto found = shader.uniform_type_table.find(var_name);
        if (found == shader.uniform_type_table.end()) {
          //fan::print("failed to set uniform value");
          return;
          //fan::throw_error("failed to set uniform value");
        }
      ie = [str = found->second, shader_nr, var_name, speed, min, max, data = initial]() mutable {
        bool modify = false;
        switch(fan::get_hash(str)) {
          case fan::get_hash(std::string_view("float")): {
            modify = ImGui::DragFloat(fan::string(std::move(var_name)).c_str(), &data[0], (f32_t)speed, (f32_t)min, (f32_t)max);
            break;
          }
          case fan::get_hash(std::string_view("vec2")): {
            modify = ImGui::DragFloat2(fan::string(std::move(var_name)).c_str(), ((fan::vec2*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
            break;
          }
          case fan::get_hash(std::string_view("vec3")): {
            modify = ImGui::DragFloat3(fan::string(std::move(var_name)).c_str(), ((fan::vec3*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
            break;
          }
          case fan::get_hash(std::string_view("vec4")): {
            modify = ImGui::DragFloat4(fan::string(std::move(var_name)).c_str(), ((fan::vec4*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
            break;
          }
        }
        if (modify) {
          gloco->get_context().shader_set_value(shader_nr, var_name, data);
        }
      };
      gloco->get_context().shader_set_value(shader_nr, var_name, initial);
    }
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
      repeat = (int)fan::keyboard_state::repeat
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

  f64_t& delta_time = window.m_delta_time;

  std::vector<functions_t> shape_functions;

  // needs continous buffer
  std::vector<shaper_t::BlockProperties_t> BlockProperties;

  shaper_t shaper;
  
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
      else if constexpr (std::is_same_v<T, letter_t::properties_t>) {
        if constexpr (fan_has_variable(loco_t, letter)) {
          *this = gloco->letter.push_back(properties);
        }
      }
      else if constexpr (std::is_same_v<T, circle_t::properties_t>) {
        if constexpr (fan_has_variable(loco_t, circle)) {
          *this = gloco->circle.push_back(properties);
        }
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

    fan::vec2 get_size();

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
      shape_gl_init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector))},
      shape_gl_init_t{6, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{7, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
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
    shape_gl_init_t{0, 4, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
    shape_gl_init_t{1, 3, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
    shape_gl_init_t{2, 3, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
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
      fan::vec3 angle;
    };
    struct ri_t {
      
    };

#pragma pack(pop)

    inline static  std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{1, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{3, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{4, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

    struct properties_t {
      using type_t = rectangle_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::color color = fan::colors::white;
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
    shape_gl_init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
    shape_gl_init_t{1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
    shape_gl_init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
    shape_gl_init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
    shape_gl_init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
    shape_gl_init_t{5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
    shape_gl_init_t{6, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
    shape_gl_init_t{7, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
    shape_gl_init_t{8, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
    shape_gl_init_t{9, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
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
      shape_gl_init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shape_gl_init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
      shape_gl_init_t{6, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shape_gl_init_t{7, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shape_gl_init_t{8, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
      shape_gl_init_t{9, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
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


  struct letter_t {

    static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::letter;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t outline_size;
      fan::vec2 size;
      fan::vec2 tc_position;
      fan::color color;
      fan::color outline_color;
      fan::vec2 tc_size;
      fan::vec3 angle;
    };
    
    struct ri_t {
      uint32_t letter_id;
      f32_t font_size;
    };

#pragma pack(pop)

    inline static std::vector<shape_gl_init_t> locations = {
      shape_gl_init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_size))},
      shape_gl_init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shape_gl_init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shape_gl_init_t{5, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color))},
      shape_gl_init_t{6, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
      shape_gl_init_t{7, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

    struct properties_t {
      using type_t = letter_t;

      fan::vec3 position;
      f32_t outline_size = 1;
      fan::vec2 size;
      fan::vec2 tc_position;
      fan::color color = fan::colors::white;
      fan::color outline_color;
      fan::vec2 tc_size;
      fan::vec3 angle = 0;

      bool blending = true;

      uint32_t letter_id;
      f32_t font_size;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }letter;

  struct text_t {

    struct vi_t {

    };

    struct ri_t {
      //letter_t::properties_t p;
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
      shape_gl_init_t{ 0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
      shape_gl_init_t{ 1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
      shape_gl_init_t{ 2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
      shape_gl_init_t{ 3, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
      shape_gl_init_t{ 4, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector)) },
      shape_gl_init_t{ 5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
      shape_gl_init_t{6, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))}
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
      shape_gl_init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{1, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, size)},
      shape_gl_init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, grid_size)},
      shape_gl_init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, rotation_point)},
      shape_gl_init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, color)},
      shape_gl_init_t{5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, angle)},
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
    shape_gl_init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
    shape_gl_init_t{1, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
    shape_gl_init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
    shape_gl_init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))}
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
      shape_gl_init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shape_gl_init_t{1, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shape_gl_init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shape_gl_init_t{3, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 0)},
      shape_gl_init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 1)},
      shape_gl_init_t{5, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 2)},
      shape_gl_init_t{6, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 3)},
      shape_gl_init_t{7, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
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


  template <typename T>
  inline void shape_open(T* shape, const fan::string& vertex, const fan::string& fragment) {
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
        .MaxElementPerBlock = (shaper_t::MaxElementPerBlock_t)MaxElementPerBlock,
        .RenderDataSize = sizeof(typename T::vi_t),
        .DataSize = sizeof(typename T::ri_t),
        .locations = T::locations,
        .shader = shader
      }
    );

    loco_t::functions_t functions = loco_t::get_functions<typename T::vi_t>();
    gloco->shape_functions.push_back(functions);
  }


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

#if defined(loco_letter)
  fan::graphics::gl_font_impl::font_t font;
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
  bool toggle_console = false;
  bool toggle_fps = false;

  ImFont* fonts[6];
#endif
  //gui


  loco_t::image_t create_noise_image(const fan::vec2& image_size) {
    loco_t::image_load_properties_t lp;
    lp.format = fan::opengl::GL_RGBA; // Change this to GL_RGB
    lp.internal_format = fan::opengl::GL_RGBA; // Change this to GL_RGB
    lp.min_filter = loco_t::image_filter::linear;
    lp.mag_filter = loco_t::image_filter::linear;
    lp.visual_output = fan::opengl::GL_MIRRORED_REPEAT;

    loco_t::image_t image;

    FastNoiseLite noise;
    noise.SetFractalType(FastNoiseLite::FractalType_FBm);
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noise.SetFrequency(0.010);
    noise.SetFractalGain(0.5);
    noise.SetFractalLacunarity(2.0);
    noise.SetFractalOctaves(5);
    noise.SetSeed(1337);
    noise.SetFractalPingPongStrength(2.0);
    f32_t noise_tex_min = -1;
    f32_t noise_tex_max = 0.1;
    //noise
    // Gather noise data
    std::vector<uint8_t> noiseDataRGB(image_size.multiply() * 4);

    int index = 0;

    float scale = 255 / (noise_tex_max - noise_tex_min);

    for (int y = 0; y < image_size.y; y++)
    {
      for (int x = 0; x < image_size.x; x++)
      {
        float noiseValue = noise.GetNoise((float)x, (float)y);
        unsigned char cNoise = (unsigned char)std::max(0.0f, std::min(255.0f, (noiseValue - noise_tex_min) * scale));
        noiseDataRGB[index * 4 + 0] = cNoise;
        noiseDataRGB[index * 4 + 1] = cNoise;
        noiseDataRGB[index * 4 + 2] = cNoise;
        noiseDataRGB[index * 4 + 3] = 255;
        index++;
      }
    }

    fan::webp::image_info_t ii;
    ii.data = noiseDataRGB.data();
    ii.size = image_size;

    image = image_load(ii, lp);
    return image;
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
      auto& ri = *(universal_image_renderer_t::ri_t*)gloco->shaper.GetData(id);
      auto vi_image = id.get_image();
      uint8_t image_amount = fan::pixel_format::get_texture_amount(format);
      if (inited == false) {
        // purge cid's images here
        // update cids images
        id.reload(format, size, filter);
        for (uint32_t i = 0; i < image_amount; ++i) {
          wresources[i].open(ri.images_rest[i].NRI);
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
          wresources[i].open(ri.images_rest[i].NRI);
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

#if defined(loco_imgui)
    using imgui_element_t = loco_t::imgui_element_t;
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

    //struct text_properties_t {
    //  camera_impl_t* camera = &gloco->orthographic_camera;
    //  std::string text = "";
    //  fan::color color = fan::colors::white;
    //  fan::color outline_color = fan::colors::black;
    //  fan::vec3 position = fan::vec3(fan::math::inf, -0.9, 0);
    //  fan::vec2 size = -1;
    //};

    //struct text_t : loco_t::shape_t {
    //  text_t(text_properties_t p = text_properties_t()) {
    //    *(loco_t::shape_t*)this = loco_t::shape_t(
    //      fan_init_struct(
    //        typename loco_t::shapes_t::responsive_text_t::properties_t,
    //        .camera = p.camera->camera,
    //        .viewport = p.camera->viewport,
    //        .position = p.position.x == fan::math::inf ? fan::vec3(-1 + 0.025 * p.text.size(), -0.9, 0) : p.position,
    //        .text = p.text,
    //        .line_limit = 1,
    //        .outline_color = p.outline_color,
    //        .letter_size_y_multipler = 1,
    //        .size = p.size == -1 ? fan::vec2(0.025 * p.text.size(), 0.1) : p.size,
    //        .color = p.color
    //      ));
    //  }
    //};

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

    
  struct letter_properties_t {
    camera_impl_t* camera = &gloco->orthographic_camera;
    fan::vec3 position = fan::vec3(0, 0, 0);
    f32_t outline_size = 1;
    fan::vec2 size = fan::vec2(0.1, 0.1);
    fan::color color = fan::color(1, 1, 1, 1);
    fan::color outline_color = fan::color(0, 0, 0, 1);
    fan::vec3 angle = fan::vec3(0, 0, 0);
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;

    uint32_t letter_id;
    f32_t font_size;

    bool blending = true;
  };

  struct letter_t : loco_t::shape_t {
    letter_t(letter_properties_t p = letter_properties_t()) {
      *(loco_t::shape_t*)this = loco_t::shape_t(
        fan_init_struct(
          typename loco_t::letter_t::properties_t,
          .camera = p.camera->camera,
          .viewport = p.camera->viewport,
          .position = p.position,
          .outline_size = p.outline_size,
          .size = p.size,
          .color = p.color,
          .outline_color = p.outline_color,
          .angle = p.angle,
          .tc_position = p.tc_position,
          .tc_size = p.tc_size,
          .blending = p.blending,
          .letter_id = p.letter_id,
          .font_size = p.font_size
        ));
    }
  };

#if defined(loco_circle)
    struct circle_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t radius = 0.1f;
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


#if defined(loco_vfi)

    // for line
    static fan::line3 get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index) {
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
  }
}

// Imgui extensions
#if defined(loco_imgui)
namespace ImGui {
  IMGUI_API void Image(loco_t::image_t img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), const ImVec4& tint_col = ImVec4(1, 1, 1, 1), const ImVec4& border_col = ImVec4(0, 0, 0, 0));
  IMGUI_API bool ImageButton(loco_t::image_t img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0, 0, 0, 0), const ImVec4& tint_col = ImVec4(1, 1, 1, 1));

  bool ToggleButton(const char* str_id, bool* v);

  bool ToggleImageButton(loco_t::image_t image, const ImVec2& size, bool* toggle);
  

  template <std::size_t N>
  bool ToggleImageButton(const std::array<loco_t::image_t, N>& images, const ImVec2& size, int* selectedIndex)
  {
    f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y -  ImGui::GetStyle().FramePadding.y / 2;
    
    bool clicked = false;

    for (std::size_t i = 0; i < images.size(); ++i) {
      ImVec4 tintColor = ImVec4(0.2, 0.2, 0.2, 1.0);
      if (*selectedIndex == i) {
        tintColor = ImVec4(0.8, 0.8, 0.8, 1.0f);
      }
      /*if (ImGui::IsItemHovered()) {
        tintColor = ImVec4(1, 1, 1, 1.0f);
      }*/
      ImGui::SetCursorPosY(y_pos);
      if (ImGui::ImageButton(images[i], size, ImVec2(0, 0), ImVec2(1, 1), 0, ImVec4(0, 0, 0, 0), tintColor)) {
        *selectedIndex = i;
        clicked = true;
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

    protected:

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
    public:

      imgui_content_browser_t() {
        search_buffer.resize(32);
        asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
        current_directory = std::filesystem::path(asset_path) / "images";
        update_directory_cache();
      }

      void update_directory_cache() {
        for (auto& img : directory_cache) {
          if (img.preview_image.iic() == false) {
            gloco->image_unload(img.preview_image);
          }
        }
        directory_cache.clear();
        fan::io::iterate_directory_sorted_by_name(current_directory, [this](const std::filesystem::directory_entry& path) {
          file_info_t file_info;
          // SLOW
          auto relative_path = std::filesystem::relative(path, asset_path);
          file_info.filename = relative_path.filename().string();
          file_info.item_path = relative_path.wstring();
          file_info.is_directory = path.is_directory();
          file_info.some_path = path.path().filename();//?
          //fan::print(get_file_extension(path.path().string()));
          if (fan::io::file::extension(path.path().string()) == ".webp") {
            file_info.preview_image = gloco->image_load(std::filesystem::absolute(path.path()).string());
          }
          directory_cache.push_back(file_info);
        });
      }

      enum viewmode_e {
        view_mode_list,
        view_mode_large_thumbnails,
        // Add more view modes as needed
      };

      viewmode_e current_view_mode = view_mode_list;
      float thumbnail_size = 128.0f;
      f32_t padding = 16.0f;

      void render();

      std::string search_buffer;

      void render_large_thumbnails_view() {
        float thumbnail_size = 128.0f;
        float panel_width = ImGui::GetContentRegionAvail().x;
        int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);

        ImGui::Columns(column_count, 0, false);

        int pressed_key = -1;
        for (int i = ImGuiKey_A; i != ImGuiKey_Z + 1; ++i) {
          if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
            pressed_key = (i - ImGuiKey_A) + 'A';
            break;
          }
        }

        // Render thumbnails or icons
        for (std::size_t i = 0; i < directory_cache.size(); ++i) {

          // reference somehow corrupts
          auto file_info = directory_cache[i];
          if (std::string(search_buffer.c_str()).size() && file_info.filename.find(search_buffer) == std::string::npos) {
            continue;
          }

          if (pressed_key != -1 && ImGui::IsWindowFocused()) {
            if (!file_info.filename.empty() && std::tolower(file_info.filename[0]) == std::tolower(pressed_key)) {
              ImGui::SetScrollHereY();
            }
          }

          ImGui::PushID(file_info.filename.c_str());
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
          ImGui::ImageButton(file_info.preview_image.iic() == false ? file_info.preview_image : file_info.is_directory ? icon_directory : icon_file, ImVec2(thumbnail_size, thumbnail_size));

          // Handle drag and drop, double click, etc.
          handle_item_interaction(file_info);

          ImGui::PopStyleColor();
          ImGui::TextWrapped(file_info.filename.c_str());
          ImGui::NextColumn();
          ImGui::PopID();
        }

        ImGui::Columns(1);
      }

      void render_list_view() {
        if (ImGui::BeginTable("##FileTable", 1, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY
          | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV
          | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable)) {
            ImGui::TableSetupColumn("##Filename", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            int pressed_key = -1;
            for (int i = ImGuiKey_A; i != ImGuiKey_Z + 1; ++i) {
              if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
                pressed_key = (i - ImGuiKey_A) + 'A';
                break;
              }
            }

            // Render table view
            for (std::size_t i = 0; i < directory_cache.size(); ++i) {

              // reference somehow corrupts
              auto file_info = directory_cache[i];

              if (pressed_key != -1 && ImGui::IsWindowFocused()) {
                if (!file_info.filename.empty() && std::tolower(file_info.filename[0]) == std::tolower(pressed_key)) {
                  ImGui::SetScrollHereY();
                }
              }

              if (search_buffer.size() && strstr(file_info.filename.c_str(), search_buffer.c_str()) == nullptr) {
                continue;
              }
              ImGui::TableNextRow();
              ImGui::TableSetColumnIndex(0); // Icon column
              fan::vec2 cursor_pos = ImGui::GetWindowPos() + ImGui::GetCursorPos() + fan::vec2(ImGui::GetScrollX(), -ImGui::GetScrollY());
              fan::vec2 image_size = ImVec2(thumbnail_size / 4, thumbnail_size / 4);
              ImGuiStyle& style = ImGui::GetStyle();
              std::string space = "";
              while (ImGui::CalcTextSize(space.c_str()).x < image_size.x) {
                space += " ";
              }
              auto str = space + file_info.filename;
            
              ImGui::Selectable(str.c_str());
              if (file_info.preview_image.iic() == false) {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get(file_info.preview_image), cursor_pos, cursor_pos + image_size);
              }
              else if (file_info.is_directory) {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get(icon_directory), cursor_pos, cursor_pos + image_size);
              }
              else {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get(icon_file), cursor_pos, cursor_pos + image_size);
              }
            
              handle_item_interaction(file_info);
            }

            ImGui::EndTable();
        }
      }

      void handle_item_interaction(auto file_info) {
        if (file_info.is_directory == false) {

          if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload("CONTENT_BROWSER_ITEM", file_info.item_path.data(), (file_info.item_path.size() + 1) * sizeof(wchar_t));
            ImGui::Text(file_info.filename.c_str());
            ImGui::EndDragDropSource();
          }
        }

        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          if (file_info.is_directory) {
            current_directory /= file_info.some_path;
            update_directory_cache();
          }
        }
      }
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

#endif

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