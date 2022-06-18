#pragma once

#include _FAN_PATH(graphics/opengl/2D/objects/letter_renderer.h)

namespace fan_2d {
  namespace graphics {

    struct text_renderer_t {

      using letter_t = fan_2d::graphics::letter_t<int, int>;

      struct properties_t {
        f32_t font_size = 0.1;
        fan::vec2 position = 0;
        fan::color color = fan::colors::white;
        fan::utf16_string text;
      };

      static void cb(letter_t* l, uint32_t src, uint32_t dst, void *p) {
        fan::print(src, dst);
      }

      void open(fan::opengl::context_t* context, font_t* font) {
        letter_ids.open();
        int x;
        letters.open(context, font, (letter_t::move_cb_t)cb, &x);
      }
      void close(fan::opengl::context_t* context) {
        for (uint32_t i = 0; i < letter_ids.size(); i++) {
          letter_ids[i].close();
        }
        letter_ids.close();
        letters.close(context);
      }

      f32_t convert_font_size(fan::opengl::context_t* context, f32_t font_size) {
        return font_size / letters.font->info.size;
      }

      fan::vec2 get_text_size(fan::opengl::context_t* context, const fan::utf16_string& text, f32_t font_size) {
        fan::vec2 text_size = 0;

        text_size.y = letters.font->info.line_height;

        f32_t width = 0;

        for (int i = 0; i < text.size(); i++) {

          auto letter = letters.font->info.characters[text[i]];

          if (i == text.size() - 1) {
            width += letter.glyph.size.x;
          }
          else {
            width += letter.metrics.advance;
          }
        }

        text_size.x = std::max(width, text_size.x);

        return text_size * convert_font_size(context, font_size);
      }

      uint32_t push_back(fan::opengl::context_t* context, properties_t properties) {
        letter_t::properties_t p;
        p.color = properties.color;
        p.font_size = properties.font_size;
        uint32_t id = letter_ids.resize(letter_ids.size() + 1);
        letter_ids[id].open();

        fan::vec2 text_size = get_text_size(context, properties.text, properties.font_size);
        f32_t left = properties.position.x - text_size.x / 2;

        fan::vec2 matrix_ratio = context->viewport_size / context->viewport_size.max();

        for (uint32_t i = 0; i < properties.text.size(); i++) {
          p.letter_id = letters.font->decode_letter(properties.text[i]);
          auto letter_info = letters.font->info.get_letter_info(properties.text[i], properties.font_size);
          
          p.position = fan::vec2(left - letter_info.metrics.offset.x, properties.position.y) + (fan::vec2(letter_info.metrics.size.x, -letter_info.metrics.size.y) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));


          letter_ids[id].push_back(letters.push_back(context, p));
          left += letter_info.metrics.advance;
        }
        return 0;
      }

      void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
        letters.bind_matrices(context, matrices);
      }

      void enable_draw(fan::opengl::context_t* context) {
        letters.enable_draw(context);
      }
      void disable_draw(fan::opengl::context_t* context) {
        letters.disable_draw(context);
      }

      letter_t letters;
      fan::hector_t<fan::hector_t<uint32_t>> letter_ids;
    };

  }
}