#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/webp.h)

namespace fan {
  namespace opengl {

    struct image_t {

      struct load_properties_defaults {
        static constexpr uint32_t visual_output = GL_CLAMP_TO_BORDER;
        static constexpr uintptr_t internal_format = GL_RGBA;
        static constexpr uintptr_t format = GL_RGBA;
        static constexpr uintptr_t type = GL_UNSIGNED_BYTE;
        static constexpr uintptr_t filter = GL_NEAREST;
      };

      struct load_properties_t {
        uint32_t visual_output = load_properties_defaults::visual_output;
        uintptr_t internal_format = load_properties_defaults::internal_format;
        uintptr_t format = load_properties_defaults::format;
        uintptr_t type = load_properties_defaults::type;
        uintptr_t filter = load_properties_defaults::filter;
      };

      void load(fan::opengl::context_t* context, const fan::webp::image_info_t image_info, load_properties_t p = load_properties_t()) {

        context->opengl.call(context->opengl.glGenTextures, 1, &texture);
        context->opengl.call(context->opengl.glBindTexture, GL_TEXTURE_2D, texture);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p.filter);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p.filter);

        size = image_info.size;

        context->opengl.call(context->opengl.glTexImage2D, GL_TEXTURE_2D, 0, p.internal_format, size.x, size.y, 0, p.format, p.type, image_info.data);

        context->opengl.call(context->opengl.glGenerateMipmap, GL_TEXTURE_2D);
        context->opengl.call(context->opengl.glBindTexture, GL_TEXTURE_2D, 0);
      }

      void load(fan::opengl::context_t* context, const std::string_view path, const load_properties_t& p = load_properties_t()) {

        #if fan_assert_if_same_path_loaded_multiple_times

        static std::unordered_map<std::string, bool> existing_images;

        if (existing_images.find(path) != existing_images.end()) {
          fan::throw_error("image already existing " + path);
        }

        existing_images[path] = 0;

        #endif

        load(context, fan::webp::load(path), p);
      }

      void reload_pixels(fan::opengl::context_t* context, image_t image, const fan::webp::image_info_t& image_info, const load_properties_t& p = load_properties_t()) {
        image.size = image_info.size;
        context->opengl.call(context->opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, image.texture);
        context->opengl.call(context->opengl.glTexImage2D, GL_TEXTURE_2D, 0, p.internal_format, image.size.x, image.size.y, 0, p.format, p.type, image_info.data);
      }

      void unload(fan::opengl::context_t* context, image_t image) {
        #if fan_debug >= fan_debug_low
        if (image == 0 || image.texture == 0) {
          fan::throw_error("texture does not exist");
        }
        #endif
        context->opengl.glDeleteTextures(1, &image.texture);

        #if fan_debug >= fan_debug_low
        image.texture = 0;
        #endif
      }

      // creates single colored text size.x*size.y sized
      void create(fan::opengl::context_t* context, fan::vec2ui size, const fan::color& color, load_properties_t p = load_properties_t()) {
        uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * (size.x * size.y * fan::color::size()));
        for (int y = 0; y < size.y; y++) {
          for (int x = 0; x < size.x; x++) {
            for (int p = 0; p < fan::color::size(); p++) {
              *pixels = color[p] * 255;
              pixels++;
            }
          }
        }

        pixels -= size.x * size.y * fan::color::size();

        context->opengl.call(context->opengl.glGenTextures, 1, &texture);

        context->opengl.call(context->opengl.glBindTexture, GL_TEXTURE_2D, texture);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, p.visual_output);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p.filter);
        context->opengl.call(context->opengl.glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p.filter);

        size = size;

        context->opengl.call(context->opengl.glTexImage2D, GL_TEXTURE_2D, 0, p.internal_format, size.x, size.y, 0, p.format, p.type, pixels);

        free(pixels);

        context->opengl.call(context->opengl.glGenerateMipmap, GL_TEXTURE_2D);
        context->opengl.call(context->opengl.glBindTexture, GL_TEXTURE_2D, 0);
      }

      uint32_t texture;
      fan::vec2i size;
    };
  }
}