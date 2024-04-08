#pragma once

#include <fan/types/types.h>
#include <fan/graphics/opengl/gl_core.h>
#include <fan/graphics/webp.h>

namespace fan {
  namespace graphics {

    struct image_t;

    struct gl_image_impl {
      #include <fan/graphics/opengl/image_list_builder_settings.h>
      #if defined(loco_opengl)
      #elif defined(loco_vulkan)
      #include <fan/graphics/vulkan/image_list_builder_settings.h>
      #endif
      #include <fan/BLL/BLL.h>
    };

    static constexpr fan::vec4_wrap_t<fan::vec2> default_texture_coordinates = fan::vec4_wrap_t<fan::vec2>(
      fan::vec2(0, 0),
      fan::vec2(1, 0),
      fan::vec2(1, 1),
      fan::vec2(0, 1)
    );

    struct image_t {

      struct format {
        static constexpr auto b8g8r8a8_unorm = fan::opengl::GL_BGRA;
        static constexpr auto r8_unorm = fan::opengl::GL_RED;
        static constexpr auto rg8_unorm = fan::opengl::GL_RG;
      };

      struct sampler_address_mode {
        static constexpr auto repeat = fan::opengl::GL_REPEAT;
        static constexpr auto mirrored_repeat = fan::opengl::GL_MIRRORED_REPEAT;
        static constexpr auto clamp_to_edge = fan::opengl::GL_CLAMP_TO_EDGE;
        static constexpr auto clamp_to_border = fan::opengl::GL_CLAMP_TO_BORDER;
        static constexpr auto mirrored_clamp_to_edge = fan::opengl::GL_MIRROR_CLAMP_TO_EDGE;
      };

      struct filter {
        static constexpr auto nearest = fan::opengl::GL_NEAREST;
        static constexpr auto linear = fan::opengl::GL_LINEAR;
      };

      struct load_properties_defaults {
        static constexpr uint32_t visual_output = fan::opengl::GL_REPEAT;
        static constexpr uint32_t internal_format = fan::opengl::GL_RGBA;
        static constexpr uint32_t format = fan::opengl::GL_RGBA;
        static constexpr uint32_t type = fan::opengl::GL_UNSIGNED_BYTE;
        static constexpr uint32_t min_filter = fan::opengl::GL_NEAREST;
        static constexpr uint32_t mag_filter = fan::opengl::GL_NEAREST;
      };

      struct load_properties_t {
        uint32_t            visual_output = load_properties_defaults::visual_output;
        uintptr_t           internal_format = load_properties_defaults::internal_format;
        uintptr_t           format = load_properties_defaults::format;
        uintptr_t           type = load_properties_defaults::type;
        uintptr_t           min_filter = load_properties_defaults::min_filter;
        uintptr_t           mag_filter = load_properties_defaults::mag_filter;
      };

      /*
            void open(fan::opengl::context_t* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_) {
        viewport_reference = viewport_list_NewNode(&context.viewport_list);
        auto node = viewport_list_GetNodeByReference(&context.viewport_list, viewport_reference);
        node->data.viewport_id = this;
      }
      void close(fan::opengl::context_t* context) {
        viewport_list_Recycle(&context.viewport_list, viewport_reference);
      }
      */

      image_t();

      image_t(const fan::webp::image_info_t image_info);

      image_t(const fan::webp::image_info_t image_info, load_properties_t p);

      image_t(const char* path);

      image_t(const char* path, load_properties_t p);

      bool is_invalid() const;

      void create_texture();
      void erase_texture();

      void bind_texture();

      void unbind_texture();

      fan::opengl::GLuint& get_texture();

      bool load(fan::webp::image_info_t image_info);

      bool load(fan::webp::image_info_t image_info, load_properties_t p);

      // returns 0 on success
      bool load(const fan::string& path);

      bool load(const fan::string& path, const load_properties_t& p);

      bool load(fan::color* colors, const fan::vec2ui& size_);

      bool load(fan::color* colors, const fan::vec2ui& size_, load_properties_t p);

      void reload_pixels(const fan::webp::image_info_t& image_info);

      void reload_pixels(const fan::webp::image_info_t& image_info, const load_properties_t& p);

      void unload();

      void create(const fan::color& color, const fan::vec2& size_);

      // creates single colored text size.x*size.y sized
      void create(const fan::color& color, const fan::vec2& size_, load_properties_t p);

      void create_missing_texture();

      void create_missing_texture(load_properties_t p);

      void create_transparent_texture();

      fan::vec4_wrap_t<fan::vec2> calculate_aspect_ratio(const fan::vec2& size, f32_t scale);

      void get_pixel_data(void* data, fan::opengl::GLenum format);

      // slow
      std::unique_ptr<uint8_t[]> get_pixel_data(fan::opengl::GLenum format, fan::vec2 uvp = 0, fan::vec2 uvs = 1);


      gl_image_impl::image_list_NodeReference_t texture_reference;
      //public:
      fan::vec2 size;
    };
  }
}