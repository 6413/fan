module;

#include <fan/types/types.h>
#include <fan/math/math.h>

#include <fan/graphics/common_context_functions_declare.h>

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

export module fan.graphics.common_context;

export import fan.types.color;
export import fan.graphics.image_load;
export import fan.camera;

import fan.print;

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
    enum {
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

  };
}

namespace bll_builds {
#include "camera_list_builder_settings.h"
#include <BLL/BLL.h>
  using camera_nr_t = camera_list_NodeReference_t;

#include "shader_list_builder_settings.h"
#include <BLL/BLL.h>
  using shader_nr_t = shader_list_NodeReference_t;

#include "image_list_builder_settings.h"
#include <BLL/BLL.h>
  using image_nr_t = image_list_NodeReference_t;

#include "viewport_list_builder_settings.h"
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
      context_build_shader_functions(context_typedef_func_ptr);
      context_build_image_functions(context_typedef_func_ptr);
      context_build_camera_functions(context_typedef_func_ptr);
      context_build_viewport_functions(context_typedef_func_ptr);
    };
    context_functions_t get_vk_context_functions();

    constexpr uint8_t get_channel_amount(uint8_t format) {
      switch (static_cast<image_format>(format)) {
      case image_format::undefined: return 0;

      case image_format::r8_unorm:
      case image_format::r8_uint: return 1;

      case image_format::rg8_unorm: return 2;

      case image_format::rgb_unorm: return 3;

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

#undef context_typedef_func_ptr
#undef context_typedef_func_ptr2
#undef context_declare_func
#undef context_declare_func2