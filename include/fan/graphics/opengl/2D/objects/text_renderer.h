#pragma once

#include _FAN_PATH(graphics/opengl/2D/objects/letter_renderer.h)

namespace fan_2d {
  namespace opengl {

    struct text_renderer_t {

      using letter_t = fan_2d::graphics::letter_t;

      struct properties_t {

        using type_t = text_renderer_t;

        f32_t font_size = 0.1;
        fan::vec2 position = 0;
        fan::color color = fan::colors::white;
        fan::color outline_color;
        f32_t outline_size;
        fan::utf16_string text;
      };

      struct id_t{
        id_t(fan::opengl::cid_t* cid) {
          block = cid->id / letter_t::max_instance_size;
          instance = cid->id % letter_t::max_instance_size;
        }
        uint32_t block;
        uint32_t instance;
      };

      void open(fan::opengl::context_t* context) {
        letter_ids.open();
        e.amount = 0;
      }
      void close(fan::opengl::context_t* context) {
        for (uint32_t i = 0; i < letter_ids.size(); i++) {
          letter_ids[i].close();
        }
        letter_ids.close();
      }

      f32_t convert_font_size(fan::opengl::context_t* context, letter_t* letters, f32_t font_size) {
        return font_size / letters->font->info.size;
      }

      fan::vec2 get_text_size(fan::opengl::context_t* context, letter_t* letters, const fan::utf16_string& text, f32_t font_size) {
        fan::vec2 text_size = 0;

        text_size.y = letters->font->info.line_height;

        f32_t width = 0;

        for (int i = 0; i < text.size(); i++) {

          auto letter = letters->font->info.characters[text[i]];

          if (i == text.size() - 1) {
            width += letter.glyph.size.x;
          }
          else {
            width += letter.metrics.advance;
          }
        }

        text_size.x = std::max(width, text_size.x);

        return text_size * convert_font_size(context, letters, font_size);
      }

      uint32_t push_back(fan::opengl::context_t* context, letter_t* letters, properties_t properties) {
        typename letter_t::properties_t p;
        p.color = properties.color;
        p.font_size = properties.font_size;
        uint32_t id;
        if (e.amount != 0) {
          id = e.id0;
          e.id0 = *(uint32_t*)&letter_ids[e.id0];
          e.amount--;
        }
        else {
          id = letter_ids.resize(letter_ids.size() + 1);
        }
        letter_ids[id].open();

        fan::vec2 text_size = get_text_size(context, letters, properties.text, properties.font_size);
        f32_t left = properties.position.x - text_size.x / 2;

        fan::vec2 matrix_ratio = context->viewport_size / context->viewport_size.max();

        for (uint32_t i = 0; i < properties.text.size(); i++) {
          p.letter_id = letters->font->decode_letter(properties.text[i]);
          auto letter_info = letters->font->info.get_letter_info(properties.text[i], properties.font_size);
          
          p.position = fan::vec2(left - letter_info.metrics.offset.x, properties.position.y) + (fan::vec2(letter_info.metrics.size.x, properties.font_size - letter_info.metrics.size.y) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));

          letter_ids[id].resize(letter_ids[id].size() + 1);
          letters->push_back(context, &letter_ids[id][letter_ids[id].size() - 1], p);
          left += letter_info.metrics.advance;
        }
        return id;
      }
      void erase(fan::opengl::context_t* context, letter_t* letters, uint32_t id) {
        for (uint32_t i = 0; i < letter_ids[id].size(); i++) {
          letters->erase(context, &letter_ids[id][i]);
        }
        letter_ids[id].close();
        *(uint32_t*)&letter_ids[id] = e.id0;
        e.id0 = id;
        e.amount++;
      }

      //template <typename T>
      //T get(fan::opengl::context_t* context, letter_t* letters, uint32_t id, T letter_t::instance_t::*member) {
      //  return letters->blocks[id.block].uniform_buffer.get_instance(context, id.instance)->*member;
      //  for (uint32_t i = 0; i < letter_ids[id].size(); i++) {
      //    letters->erase(context, &letter_ids[id][i]);
      //  }
      //}
      template <typename T, typename T2>
      void set(fan::opengl::context_t* context, letter_t* letters, uint32_t id, T letter_t::instance_t::*member, const T2& value) {
        for (uint32_t i = 0; i < letter_ids[id].size(); i++) {
          letters->set(context, &letter_ids[id][i], member, value);
        }
      }

      struct{
        uint16_t id0;

        uint32_t amount;
      }e;

      fan::hector_t<fan::hector_t<fan::opengl::cid_t>> letter_ids;
    };
  }
}