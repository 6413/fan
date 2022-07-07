#pragma once

#include _FAN_PATH(graphics/opengl/2D/objects/letter_renderer.h)

namespace fan_2d {
  namespace opengl {

    template <typename T_user_global_data>
    struct text_renderer_t {

      using user_global_data_t = T_user_global_data;

    #pragma pack(push, 1)
      struct letter_data_t {
        uint16_t id0;
        uint16_t id1;
      };
    #pragma pack(pop)

      using letter_t = fan_2d::graphics::letter_t<user_global_data_t, letter_data_t>;

      struct properties_t {

        using type_t = text_renderer_t;

        f32_t font_size = 0.1;
        fan::vec2 position = 0;
        fan::color color = fan::colors::white;
        fan::color outline_color;
        f32_t outline_size;
        fan::utf16_string text;
      };

      static void cb(letter_t* l, uint32_t src, uint32_t dst, letter_data_t *letter_data) {
        text_renderer_t* tr = OFFSETLESS(l, text_renderer_t, letters);
        tr->letter_ids[letter_data->id0][letter_data->id1] = dst;
      }

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
        p.data.id0 = id;

        fan::vec2 text_size = get_text_size(context, letters, properties.text, properties.font_size);
        f32_t left = properties.position.x - text_size.x / 2;

        fan::vec2 matrix_ratio = context->viewport_size / context->viewport_size.max();

        for (uint32_t i = 0; i < properties.text.size(); i++) {
          p.letter_id = letters->font->decode_letter(properties.text[i]);
          auto letter_info = letters->font->info.get_letter_info(properties.text[i], properties.font_size);
          
          p.position = fan::vec2(left - letter_info.metrics.offset.x, properties.position.y) + (fan::vec2(letter_info.metrics.size.x, 0) / 2 + fan::vec2(letter_info.metrics.offset.x, -letter_info.metrics.offset.y));


          p.data.id1 = letter_ids[id].size();
          letter_ids[id].push_back(letters->push_back(context, p));
          left += letter_info.metrics.advance;
        }
        return id;
      }
      void erase(fan::opengl::context_t* context, letter_t* letters, uint32_t id) {
        for (uint32_t i = 0; i < letter_ids[id].size(); i++) {
          letters->erase(context, letter_ids[id][i]);
        }
        letter_ids[id].close();
        *(uint32_t*)&letter_ids[id] = e.id0;
        e.id0 = id;
        e.amount++;
      }

      struct{
        uint16_t id0;

        uint32_t amount;
      }e;

      fan::hector_t<fan::hector_t<uint32_t>> letter_ids;
    };
  }
}