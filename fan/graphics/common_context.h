#pragma once

#include <variant>

#include <fan/types/types.h>
#include <fan/graphics/common_context_functions_declare.h>

#include <fan/graphics/camera.h>
#include <fan/graphics/image_load.h>

namespace fan {
  namespace graphics {
    enum image_format {
      r8b8g8a8_unorm,
      b8g8r8a8_unorm,
      r8_unorm,
      rg8_unorm,
      rgb_unorm,
      rgba_unorm,
      r8_uint,
      r8g8b8a8_srgb,
      r11f_g11f_b10f
    };
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
    };
    enum {
      fan_unsigned_byte,
      fan_byte,
      fan_unsigned_int,
      fan_float,
    };
    struct image_load_properties_defaults {
      static constexpr uint32_t visual_output = repeat;
      static constexpr uint32_t internal_format = rgba_unorm;
      static constexpr uint32_t format = rgba_unorm;
      static constexpr uint32_t type = fan_unsigned_byte; // internal
      static constexpr uint32_t min_filter = nearest;
      static constexpr uint32_t mag_filter = nearest;
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

    struct context_viewport_t{
      fan::vec2 viewport_position;
      fan::vec2 viewport_size;
    };

    #include "camera_list_builder_settings.h"
    #include <BLL/BLL.h>
    using camera_nr_t = camera_list_NodeReference_t;

    struct shader_data_t {
      std::string svertex, sfragment;
      std::unordered_map<std::string, std::string> uniform_type_table;
      void* internal;
    };
    // stores list here and inside renderer for resetting renderer without closing nrs
    #include "shader_list_builder_settings.h"
    #include <BLL/BLL.h>
    using shader_nr_t = shader_list_NodeReference_t;

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
    #include "image_list_builder_settings.h"
    #include <BLL/BLL.h>
    using image_nr_t = image_list_NodeReference_t;

    #include "viewport_list_builder_settings.h"
    #include <BLL/BLL.h>
    using viewport_nr_t = viewport_list_NodeReference_t;

    typedef uint8_t*(*this_offset_camera_list_t)(uint8_t* context);
    typedef uint8_t*(*this_offset_shader_list_t)(uint8_t* context);
    typedef uint8_t*(*this_offset_image_list_t)(uint8_t* context);
    typedef uint8_t*(*this_offset_viewport_list_t)(uint8_t* context);

    inline this_offset_camera_list_t get_camera_list;
    inline this_offset_shader_list_t get_shader_list;
    inline this_offset_image_list_t get_image_list;
    inline this_offset_viewport_list_t get_viewport_list;

    static constexpr f32_t znearfar = 0xffff;

    struct primitive_topology_t {
      static constexpr uint32_t points = 0;
      static constexpr uint32_t lines = 1;
      static constexpr uint32_t line_strip = 2;
      static constexpr uint32_t triangles = 3;
      static constexpr uint32_t triangle_strip = 4;
      static constexpr uint32_t triangle_fan = 5;
      static constexpr uint32_t lines_with_adjacency = 6;
      static constexpr uint32_t line_strip_with_adjacency = 7;
      static constexpr uint32_t triangles_with_adjacency = 8;
      static constexpr uint32_t triangle_strip_with_adjacency = 9;
    };

    struct context_functions_t {
      context_build_shader_functions(context_typedef_func_ptr);
      context_build_image_functions(context_typedef_func_ptr);
      context_build_camera_functions(context_typedef_func_ptr);
      context_build_viewport_functions(context_typedef_func_ptr);
    };
    context_functions_t get_gl_context_functions();
    context_functions_t get_vk_context_functions();
  }

  struct pixel_format {
    enum {
      undefined,
      yuv420p,
      nv12,
    };

    constexpr static uint8_t get_texture_amount(uint8_t format) {
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
    constexpr static std::array<fan::vec2ui, 4> get_image_sizes(uint8_t format, const fan::vec2ui& image_size) {
      switch (format) {
      case yuv420p: {
        return std::array<fan::vec2ui, 4>{image_size, image_size / 2, image_size / 2};
      }
      case nv12: {
        return std::array<fan::vec2ui, 4>{image_size, fan::vec2ui{ image_size.x / 2, image_size.y / 2 }};
      }
      default: {
        fan::throw_error("invalid format");
        return std::array<fan::vec2ui, 4>{};
      }
      }
    }
    template <typename T>
    static constexpr std::array<T, 4> get_image_properties(uint8_t format) {
      std::array<T, 4> result{};

      switch (format) {
      case yuv420p:
        for (int i = 0; i < 3; ++i) {
          result[i].internal_format = fan::graphics::image_format::r8_unorm;
          result[i].format = fan::graphics::image_format::r8_unorm;
        }
        break;

      case nv12:
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

#undef context_typedef_func_ptr
#undef context_typedef_func_ptr2
#undef context_declare_func
#undef context_declare_func2

#ifndef camera_list
  #define camera_list (*(fan::graphics::camera_list_t*)fan::graphics::get_camera_list((uint8_t*)&context))
#endif

#ifndef shader_list
  #define shader_list (*(fan::graphics::shader_list_t*)fan::graphics::get_shader_list((uint8_t*)&context))
#endif

#ifndef image_list
  #define image_list (*(fan::graphics::image_list_t*)fan::graphics::get_image_list((uint8_t*)&context))
#endif

#ifndef viewport_list
  #define viewport_list (*(fan::graphics::viewport_list_t*)fan::graphics::get_viewport_list((uint8_t*)&context))
#endif