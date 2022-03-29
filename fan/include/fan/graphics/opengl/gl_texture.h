#pragma once

#include <fan/graphics/opengl/gl_image.h>


namespace fan {
  namespace opengl {
    
    // creates single colored text size.x*size.y sized
    static image_t* create_texture(fan::opengl::context_t* context, fan::vec2ui size, const fan::color& color, fan::opengl::image_load_properties_t p = fan::opengl::image_load_properties_t()) {
      uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * (size.x * size.y * fan::color::size()));
      for (int y = 0; y < size.y; y++) {
        for (int x = 0; x < size.x; x++) {
          for (int p = 0; p < fan::color::size(); p++) {
            *pixels = color[p] * 255;
            pixels++;
          }
        }
      }

      image_t* image = new image_t;

      pixels -= size.x * size.y * fan::color::size();

      context->opengl.glGenTextures(1, &image->texture);

			context->opengl.glBindTexture(GL_TEXTURE_2D, image->texture);
			context->opengl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, p.visual_output);
			context->opengl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, p.visual_output);
			context->opengl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p.filter);
			context->opengl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p.filter);

      image->size = size;

			context->opengl.glTexImage2D(GL_TEXTURE_2D, 0, p.internal_format, image->size.x, image->size.y, 0, p.format, p.type, pixels);

      free(pixels);

			context->opengl.glGenerateMipmap(GL_TEXTURE_2D);
			context->opengl.glBindTexture(GL_TEXTURE_2D, 0);

      return image;
    }

  }
}