#pragma once

#include <fan/graphics/opengl/core.h>
#include <fan/graphics/vulkan/core.h>

namespace fan {
  namespace graphics {
    struct context_t {
      context_t(){}
      ~context_t(){}
      context_t(const context_t&) {}
      union {
        fan::opengl::context_t gl;
        fan::vulkan::context_t vk;
      };
    };

    struct context_shader_t {
      context_shader_t() {}
      ~context_shader_t() {}
      union {
        fan::opengl::context_t::shader_t* gl;
        fan::vulkan::context_t::shader_t* vk;
      };
    };
    struct context_shader_nr_t {
      context_shader_nr_t() {}
      ~context_shader_nr_t() {}
      bool iic() const {
        return ((fan::opengl::context_t::shader_nr_t*)&nr)->iic();
      }
      union {
        struct {
          uint8_t nr[sizeof(fan::opengl::context_t::shader_nr_t)];
        };
        fan::opengl::context_t::shader_nr_t gl;
        fan::vulkan::context_t::shader_nr_t vk;
      };
    };
    typedef context_shader_nr_t(*shader_create_t)(context_t&);
    typedef context_shader_t (*shader_get_t)(context_t&, context_shader_nr_t);
    typedef void (*shader_erase_t)(context_t&, context_shader_nr_t);
    typedef void (*shader_use_t)(context_t&, context_shader_nr_t);
    typedef void (*shader_set_vertex_t)(context_t&, context_shader_nr_t, const std::string& code);
    typedef void (*shader_set_fragment_t)(context_t&, context_shader_nr_t, const std::string& code);
    typedef bool (*shader_compile_t)(context_t&, context_shader_nr_t);

    enum image_format {
      b8g8r8a8_unorm,
      r8_unorm,
      rg8_unorm,
      rgb_unorm,
      rgba_unorm,
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
    struct image_load_properties_t {
      uint32_t visual_output = image_load_properties_defaults::visual_output;
      uintptr_t internal_format = image_load_properties_defaults::internal_format;
      uintptr_t format = image_load_properties_defaults::format;
      uintptr_t type = image_load_properties_defaults::type;
      uintptr_t min_filter = image_load_properties_defaults::min_filter;
      uintptr_t mag_filter = image_load_properties_defaults::mag_filter;

      image_load_properties_t() = default;
      image_load_properties_t(const fan::opengl::context_t::image_load_properties_t& lp) {
        visual_output = lp.visual_output;
        internal_format = lp.internal_format;
        format = lp.format;
        type = lp.type;
        min_filter = lp.min_filter;
        mag_filter = lp.mag_filter;
      }
      image_load_properties_t(const fan::vulkan::context_t::image_load_properties_t& lp) {
        /* .visual_output = visual_output,
          .internal_format = internal_format,
          .format = format,
          .type = type,
          .min_filter = min_filter,
          .mag_filter = mag_filter,*/
      }

      operator fan::opengl::context_t::image_load_properties_t() const {
        return {
          .visual_output = visual_output,
          .internal_format = internal_format,
          .format = format,
          .type = type,
          .min_filter = min_filter,
          .mag_filter = mag_filter,
        };
      }
      operator fan::vulkan::context_t::image_load_properties_t() const {
        return {
         /* .visual_output = visual_output,
          .internal_format = internal_format,
          .format = format,
          .type = type,
          .min_filter = min_filter,
          .mag_filter = mag_filter,*/
        };
      }
    };
    
    struct context_image_common_t{
      fan::vec2 size;
    };
    struct context_image_t : context_image_common_t {
      context_image_t() {}
      ~context_image_t() {}
      union {
        fan::opengl::context_t::image_t gl;
        fan::vulkan::context_t::image_t vk;
      };
    };
    struct context_image_nr_common_t {
      
    };
    struct context_image_nr_t {
      context_image_nr_t() {}
      ~context_image_nr_t() {}
      union {
        uint8_t nr[sizeof(fan::opengl::context_t::image_nr_t)];
        fan::opengl::context_t::image_nr_t gl;
        static_assert(sizeof(nr) == sizeof(gl), "invalid context image common nr");
        fan::vulkan::context_t::image_nr_t vk;
        static_assert(sizeof(nr) == sizeof(vk), "invalid context image common nr");
      };
      bool iic() const {
        return ((fan::opengl::context_t::image_nr_t*)&nr)->iic();
      }
    };
    typedef context_image_nr_t (*image_create_t)(context_t&);
    typedef context_image_t  (*image_get_t)(context_t&, context_image_nr_t nr); // 
    typedef uint64_t (*image_get_handle_t)(context_t&, context_image_nr_t nr);
    typedef void (*image_erase_t)(context_t&, context_image_nr_t nr);
    typedef void (*image_bind_t)(context_t&, context_image_nr_t nr);
    typedef void (*image_unbind_t)(context_t&, context_image_nr_t nr);
    typedef void (*image_set_settings_t)(context_t&, const image_load_properties_t& p);
    typedef context_image_nr_t (*image_load_info_t)(context_t&, const fan::image::image_info_t& image_info);
    typedef context_image_nr_t (*image_load_info_props_t)(context_t&, const fan::image::image_info_t& image_info, const image_load_properties_t& p);
    typedef context_image_nr_t (*image_load_path_t)(context_t&, const fan::string& path);
    typedef context_image_nr_t (*image_load_path_props_t)(context_t&, const fan::string& path, const image_load_properties_t& p);
    typedef context_image_nr_t (*image_load_colors_t)(context_t&, fan::color* colors, const fan::vec2ui& size_);
    typedef context_image_nr_t (*image_load_colors_props_t)(context_t&, fan::color* colors, const fan::vec2ui& size_, const image_load_properties_t& p);
    typedef void (*image_unload_t)(context_t&, context_image_nr_t nr);
    typedef context_image_nr_t (*create_missing_texture_t)(context_t&);
    typedef context_image_nr_t (*create_transparent_texture_t)(context_t&);
    typedef void (*image_reload_pixels_t)(context_t&, context_image_nr_t nr, const fan::image::image_info_t& image_info);
    typedef void (*image_reload_pixels_props_t)(context_t&, context_image_nr_t nr, const fan::image::image_info_t& image_info, const image_load_properties_t& p);
    typedef context_image_nr_t (*image_create_color_t)(context_t&, const fan::color& color);
    typedef context_image_nr_t (*image_create_color_props_t)(context_t&, const fan::color& color, const image_load_properties_t& p);

    struct context_camera_common_t : fan::camera {
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

    struct context_camera_t {
      context_camera_t() {}
      union {
        fan::opengl::context_t::camera_t* gl;
        fan::vulkan::context_t::camera_t* vk;
      };
    };
    struct context_camera_nr_t {
      context_camera_nr_t() {}
      ~context_camera_nr_t() {}
      bool iic() const {
        return ((fan::opengl::context_t::camera_nr_t*)&nr)->iic();
      }
      void sic() {
        ((fan::opengl::context_t::camera_nr_t*)&nr)->sic();
      }
      union {
        struct {
          uint8_t nr[sizeof(fan::opengl::context_t::camera_nr_t)];
        };
        fan::opengl::context_t::camera_nr_t gl;
        fan::vulkan::context_t::camera_nr_t vk;
      };
    };
    typedef context_camera_nr_t (*camera_create_t)(context_t&);
    typedef context_camera_t (*camera_get_t)(context_t&, context_camera_nr_t nr);
    typedef void (*camera_erase_t)(context_t&, context_camera_nr_t nr);
    typedef context_camera_nr_t (*camera_open_t)(context_t&, const fan::vec2& x, const fan::vec2& y);
    typedef fan::vec3 (*camera_get_position_t)(context_t&, context_camera_nr_t nr);
    typedef void (*camera_set_position_t)(context_t&, context_camera_nr_t nr, const fan::vec3& cp);
    typedef fan::vec2 (*camera_get_size_t)(context_t&, context_camera_nr_t nr);
    typedef void (*camera_set_ortho_t)(context_t&, context_camera_nr_t nr, fan::vec2 x, fan::vec2 y);
    typedef void (*camera_set_perspective_t)(context_t&, context_camera_nr_t nr, f32_t fov, const fan::vec2& window_size);
    typedef void (*camera_rotate_t)(context_t&, context_camera_nr_t nr, const fan::vec2& offset);


    struct context_viewport_t {
      context_viewport_t() {}
      ~context_viewport_t() {}
      union {
        struct {
          fan::vec2 viewport_position;
          fan::vec2 viewport_size;
        };
        fan::opengl::context_t::viewport_t gl;
        fan::vulkan::context_t::viewport_t vk;
      };
    };
    struct context_viewport_nr_t {
      context_viewport_nr_t() {}
      ~context_viewport_nr_t() {}
      bool iic() const {
        return ((fan::opengl::context_t::viewport_nr_t*)&nr)->iic();
      }
      void sic() {
        ((fan::opengl::context_t::viewport_nr_t*)&nr)->sic();
      }
      union {
        struct {
          uint8_t nr[sizeof(fan::opengl::context_t::viewport_nr_t)];
        };
        fan::opengl::context_t::viewport_nr_t gl;
        fan::vulkan::context_t::viewport_nr_t vk;
      };
    };
    typedef context_viewport_nr_t (*viewport_create_t)(context_t&);
    typedef context_viewport_t (*viewport_get_t)(context_t&, context_viewport_nr_t nr);
    typedef void (*viewport_erase_t)(context_t&, context_viewport_nr_t nr);
    typedef fan::vec2 (*viewport_get_position_t)(context_t&, context_viewport_nr_t nr);
    typedef fan::vec2 (*viewport_get_size_t)(context_t&, context_viewport_nr_t nr);
    typedef void (*viewport_set_t)(context_t&, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
    typedef void (*viewport_set_nr_t)(context_t&, context_viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
    typedef void (*viewport_zero_t)(context_t&, context_viewport_nr_t nr);
    typedef bool (*viewport_inside_t)(context_t&, context_viewport_nr_t nr, const fan::vec2& position);
    typedef bool (*viewport_inside_wir_t)(context_t&, context_viewport_nr_t nr, const fan::vec2& position);


    struct context_functions_t {
      shader_create_t shader_create;
      shader_get_t shader_get;
      shader_erase_t shader_erase;
      shader_use_t shader_use;
      shader_set_vertex_t shader_set_vertex;
      shader_set_fragment_t shader_set_fragment;
      shader_compile_t shader_compile;

      image_create_t image_create;
      image_get_t image_get;
      image_get_handle_t image_get_handle;
      image_erase_t image_erase;
      image_bind_t image_bind;
      image_unbind_t image_unbind;
      image_set_settings_t image_set_settings;
      image_load_info_t image_load_info;
      image_load_info_props_t image_load_info_props;
      image_load_path_t image_load_path;
      image_load_path_props_t image_load_path_props;
      image_load_colors_t image_load_colors;
      image_load_colors_props_t image_load_colors_props;
      image_unload_t image_unload;
      create_missing_texture_t create_missing_texture;
      create_transparent_texture_t create_transparent_texture;
      image_reload_pixels_t image_reload_pixels;
      image_reload_pixels_props_t image_reload_pixels_props;
      image_create_color_t image_create_color;
      image_create_color_props_t image_create_color_props;

      camera_create_t camera_create;
      camera_get_t camera_get;
      camera_erase_t camera_erase;
      camera_open_t camera_open;
      camera_get_position_t camera_get_position;
      camera_set_position_t camera_set_position;
      camera_get_size_t camera_get_size;
      camera_set_ortho_t camera_set_ortho;
      camera_set_perspective_t camera_set_perspective;
      camera_rotate_t camera_rotate;

      viewport_create_t viewport_create;
      viewport_get_t viewport_get;
      viewport_erase_t viewport_erase;
      viewport_get_position_t viewport_get_position;
      viewport_get_size_t viewport_get_size;
      viewport_set_t viewport_set;
      viewport_set_nr_t viewport_set_nr;
      viewport_zero_t viewport_zero;
      viewport_inside_t viewport_inside;
      viewport_inside_wir_t viewport_inside_wir;
    };

    context_functions_t get_gl_context_functions();
    context_functions_t get_vk_context_functions();
  }
}

namespace fan {
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