#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)

#ifndef fan_platform_android

namespace fan {
  namespace opengl {
   
    static constexpr fan::_vec4<fan::vec2> default_texture_coordinates = {
      vec2(0, 0),
      vec2(1, 0),
      vec2(1, 1),
      vec2(0, 1)
    };

    struct image_t {

      struct load_properties_defaults {
        static constexpr uint32_t visual_output = GL_CLAMP_TO_BORDER;
        static constexpr uint32_t internal_format = GL_RGBA;
        static constexpr uint32_t format = GL_RGBA;
        static constexpr uint32_t type = GL_UNSIGNED_BYTE;
        static constexpr uint32_t filter = GL_NEAREST;
      };

      struct load_properties_t {
        constexpr load_properties_t() noexcept {}
        constexpr load_properties_t(auto a, auto b, auto c, auto d, auto e)
          : visual_output(a), internal_format(b), format(c), type(d), filter(e) {}
        uint32_t visual_output = load_properties_defaults::visual_output;
        uintptr_t           internal_format = load_properties_defaults::internal_format;
        uintptr_t           format = load_properties_defaults::format;
        uintptr_t           type = load_properties_defaults::type;
        uintptr_t           filter = load_properties_defaults::filter;
      };

      /*
            void open(fan::opengl::context_t* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_) {
        viewport_reference = viewport_list_NewNode(&context->viewport_list);
        auto node = viewport_list_GetNodeByReference(&context->viewport_list, viewport_reference);
        node->data.viewport_id = this;
      }
      void close(fan::opengl::context_t* context) {
        viewport_list_Recycle(&context->viewport_list, viewport_reference);
      }
      */

      void create_texture(fan::opengl::context_t* context) {
        texture_reference = context->image_list.NewNode();
        context->opengl.call(context->opengl.glGenTextures, 1, get_texture(context));
      }
      void erase_texture(fan::opengl::context_t* context) {
        context->opengl.glDeleteTextures(1, get_texture(context));
        context->image_list.Recycle(texture_reference);
      }

      void bind_texture(fan::opengl::context_t* context) {
        context->opengl.call(context->opengl.glBindTexture, GL_TEXTURE_2D, *get_texture(context));
      }

      GLuint* get_texture(fan::opengl::context_t* context) {
        return &context->image_list[texture_reference].texture_id;
      }

      bool load(fan::opengl::context_t* context, const fan::webp::image_info_t image_info, load_properties_t p = load_properties_t()) {

        create_texture(context);
        bind_texture(context);

        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p.filter);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p.filter);

        size = image_info.size;

        context->opengl.call(context->opengl.glTexImage2D, GL_TEXTURE_2D, 0, p.internal_format, size.x, size.y, 0, p.format, p.type, image_info.data);

        //context->opengl.call(context->opengl.glGenerateMipmap, GL_TEXTURE_2D);
        
        return 0;
      }

      bool load(fan::opengl::context_t* context, const fan::string& path, const load_properties_t& p = load_properties_t()) {

        #if fan_assert_if_same_path_loaded_multiple_times

        static std::unordered_map<fan::string, bool> existing_images;

        if (existing_images.find(path) != existing_images.end()) {
          fan::throw_error("image already existing " + path);
        }

        existing_images[path] = 0;

        #endif

        fan::webp::image_info_t image_info;
        if (fan::webp::load(path, &image_info)) {
          return 0;
        }
        bool ret = load(context, image_info, p);
        fan::webp::free_image(image_info.data);
        //fan::webp::free_image(image_info.data); leaks and double free sometimes
        return ret;
      }
      bool load(fan::opengl::context_t* context, fan::color* colors, const fan::vec2ui& size_, load_properties_t p = load_properties_t()) {

        create_texture(context);
        bind_texture(context);

        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p.filter);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p.filter);

        size = size_;

        context->opengl.call(context->opengl.glTexImage2D, GL_TEXTURE_2D, 0, fan::opengl::GL_RGBA32F, size.x, size.y, 0, p.format, fan::opengl::GL_FLOAT, (uint8_t*)colors);

        return 0;
      }

      void reload_pixels(fan::opengl::context_t* context, const fan::webp::image_info_t& image_info, const load_properties_t& p = load_properties_t()) {

        bind_texture(context);

        size = image_info.size;
        context->opengl.call(context->opengl.glTexImage2D, GL_TEXTURE_2D, 0, p.internal_format, size.x, size.y, 0, p.format, p.type, image_info.data);
      }

      void unload(fan::opengl::context_t* context) {
        erase_texture(context);
      }

      // creates single colored text size.x*size.y sized
      void create(fan::opengl::context_t* context, const fan::color& color, const fan::vec2& size_, load_properties_t p = load_properties_t()) {
        size = size_;

        uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * (size.x * size.y * fan::color::size()));
        for (int y = 0; y < size_.y; y++) {
          for (int x = 0; x < size_.x; x++) {
            for (int p = 0; p < fan::color::size(); p++) {
              *pixels = color[p] * 255;
              pixels++;
            }
          }
        }

        pixels -= size.x * size.y * fan::color::size();

        create_texture(context);
        bind_texture(context);

        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p.filter);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p.filter);

        context->opengl.call(context->opengl.glTexImage2D, GL_TEXTURE_2D, 0, p.internal_format, size.x, size.y, 0, p.format, p.type, pixels);

        free(pixels);

        context->opengl.call(context->opengl.glGenerateMipmap, GL_TEXTURE_2D);
      }

      void create_missing_texture(fan::opengl::context_t* context, load_properties_t p = load_properties_t()) {
        uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * (2 * 2 * fan::color::size()));
        uint32_t pixel = 0;

        pixels[pixel++] = 0;
        pixels[pixel++] = 0;
        pixels[pixel++] = 0;
        pixels[pixel++] = 255;

        pixels[pixel++] = 255;
        pixels[pixel++] = 0;
        pixels[pixel++] = 220;
        pixels[pixel++] = 255;

        pixels[pixel++] = 255;
        pixels[pixel++] = 0;
        pixels[pixel++] = 220;
        pixels[pixel++] = 255;

        pixels[pixel++] = 0;
        pixels[pixel++] = 0;
        pixels[pixel++] = 0;
        pixels[pixel++] = 255;

        p.visual_output = fan::opengl::GL_REPEAT;

        create_texture(context);
        bind_texture(context);

        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p.filter);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p.filter);

        size = fan::vec2i(2, 2);

        context->opengl.call(context->opengl.glTexImage2D, GL_TEXTURE_2D, 0, p.internal_format, 2, 2, 0, p.format, p.type, pixels);

        free(pixels);

        context->opengl.call(context->opengl.glGenerateMipmap, GL_TEXTURE_2D);
      }

      fan::_vec4<fan::vec2> calculate_aspect_ratio(const fan::vec2& size, f32_t scale) {

        fan::_vec4<fan::vec2> tc = {
          fan::vec2(0, 1),
          fan::vec2(1, 1),
          fan::vec2(1, 0),
          fan::vec2(0, 0)
        };

        f32_t a = size.x / size.y;
        fan::vec2 n = size.normalize();

        for (uint32_t i = 0; i < 8; i++) {
          if (size.x < size.y) {
            tc[i % 4][i / 4] *= n[i / 4] / a * scale;
          }
          else {
            tc[i % 4][i / 4] *= n[i / 4] * a * scale;
          }
        }
        return tc;
      }

      void get_pixel_data(fan::opengl::context_t* context, void* data, fan::opengl::GLenum format) {
        bind_texture(context);

        context->opengl.call(
          context->opengl.glGetTexImage, 
          fan::opengl::GL_TEXTURE_2D,
          0,
          format,
          fan::opengl::GL_UNSIGNED_BYTE,
          data
        );
      }

      fan::opengl::image_list_NodeReference_t texture_reference;
    //public:
      fan::vec2i size;
    };
  }
}


fan::opengl::image_list_NodeReference_t::image_list_NodeReference_t(fan::opengl::image_t* image) {
  NRI = image->texture_reference.NRI;
}
#endif