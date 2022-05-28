#pragma once

#include _FAN_PATH(graphics/gui/rectangle_box_sized.h)
#include _FAN_PATH(graphics/opengl/2D/gui/text_renderer.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct rectangle_text_box_sized_properties {

        fan::utf16_string text;

        fan::utf16_string place_holder;

        fan::vec2 position = 0;

        f32_t font_size = fan_2d::graphics::gui::defaults::font_size;

        text_position_e text_position = text_position_e::middle;

        fan_2d::graphics::gui::theme theme = fan_2d::graphics::gui::themes::deep_blue();

        fan::vec2 size = 0;

        fan::vec2 offset = 0;
        
        void* userptr;
      };

      struct rectangle_text_box_sized_t {

        using properties_t = rectangle_text_box_sized_properties;

        using inner_rect_t = fan::class_duplicator<fan_2d::opengl::rectangle_t, 0>;
        using outer_rect_t = fan::class_duplicator<fan_2d::opengl::rectangle_t, 1>;

        rectangle_text_box_sized_t() = default;

        void open(fan::opengl::context_t* context)
        {
          rbs.open(context);
          tr.open(context);
          m_store.open();
        }

        void close(fan::opengl::context_t* context)
        {
          rbs.close(context);
          tr.close(context);
          for (int i = 0; i < m_store.size(); i++) {
            m_store[i].m_properties.text.close();
            m_store[i].m_properties.place_holder.close();
          }
          m_store.close();
        }

        void push_back(fan::opengl::context_t* context, const properties_t& property)
        {
          store_t store;

          store.m_properties.place_holder.open();
          store.m_properties.text.open();

          store.m_properties.font_size = property.font_size;
          *store.m_properties.place_holder = property.font_size;
          *store.m_properties.text = property.text;
          store.m_properties.text_position = property.text_position;
          store.m_properties.userptr = property.userptr;

          m_store.push_back(store);

          const auto str = property.place_holder.empty() ? property.text : property.place_holder;

          fan_2d::opengl::gui::text_renderer_t::properties_t text_properties;

          switch (property.text_position) {
            case text_position_e::left:
            {

              text_properties.text = str;
              text_properties.font_size = property.font_size;
              text_properties.position = fan::vec2(
                property.position.x + property.theme.button.outline_thickness - property.size.x * 0.5,
                property.position.y + property.theme.button.outline_thickness
              ) + property.offset;
              text_properties.text_color = property.place_holder.empty() ? property.theme.button.text_color : defaults::text_color_place_holder;
              text_properties.outline_color = property.theme.button.text_outline_color;
              text_properties.outline_size = property.theme.button.text_outline_size;

              tr.push_back(context, text_properties);

              break;
            }
            case text_position_e::middle:
            {
              text_properties.text = str;
              text_properties.font_size = property.font_size;
              text_properties.position = property.position + property.offset;
              text_properties.text_color = property.text.size() && property.text[0] != '\0' ? property.theme.button.text_color : defaults::text_color_place_holder;

              text_properties.outline_color = property.theme.button.text_outline_color;
              text_properties.outline_size = property.theme.button.text_outline_size;

              tr.push_back(context, text_properties);

              break;
            }
          }

          inner_rect_t::properties_t rect_properties;
          rect_properties.position = property.position;
          rect_properties.size = property.size;
          //rect_properties.rotation_point = rect_properties.position;
          rect_properties.color = property.theme.button.color;

          fan_2d::graphics::gui::rectangle_box_sized_t::properties_t rbsp;
          rbsp.offset = property.offset;
          rbsp.theme = property.theme;
          rbsp.position = property.position;
          rbsp.size = property.size;
          rbs.push_back(context, rbsp);
        }

        void draw(fan::opengl::context_t* context) {
          rbs.draw(context);

          tr.draw(context);
        }

        void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
          rbs.set_position(context, i, position);
        }

        bool inside(fan::opengl::context_t* context, uintptr_t i, const fan::vec2& position) const
        {
          return rbs.inside(context, i, position);
        }

        void set_text(fan::opengl::context_t* context, uint32_t i, const fan::utf16_string& text) {
          switch (m_store[i].m_properties.text_position) {
            case text_position_e::left:
            {
              tr.set_text(context, i, text);
              break;
            }
            case text_position_e::middle:
            {
              tr.set_text(context, i, text);
              break;
            }
          }
        }

        void set_size(fan::opengl::context_t* context, uint32_t i, const fan::vec2& size) {
          rbs.set_size(context, i, size);
        }

        fan::color get_text_color(fan::opengl::context_t* context, uint32_t i) const
        {
          return tr.get_text_color(context, i);
        }

        void set_text_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color)
        {
          tr.set_text_color(context, i, color);
        }

        fan::vec2 get_position(fan::opengl::context_t* context, uint32_t i) const
        {
          return rbs.get_position(context, i);
        }

        fan::vec2 get_size(fan::opengl::context_t* context, uint32_t i) const
        {
          return rbs.get_size(context, i);
        }

        f32_t get_font_size(fan::opengl::context_t* context, uint32_t i) const
        {
          return tr.get_font_size(context, i);
        }

        fan::color get_color(fan::opengl::context_t* context, uint32_t i) const
        {
          return rbs.get_color(context, i);
        }

        fan::utf16_string get_text(fan::opengl::context_t* context, uint32_t i) const {
          return tr.get_text(context, i);
        }

        fan_2d::graphics::gui::src_dst_t get_cursor(fan::opengl::context_t* context, uint32_t i, uint32_t x, uint32_t y)
        {
          f32_t font_size = get_font_size(context, i);
          f32_t line_height = fan_2d::opengl::gui::text_renderer_t::get_line_height(context, font_size);

          fan::vec2 src, dst;

          if (m_store[i].m_properties.text_position == fan_2d::graphics::gui::text_position_e::left) {

            src += (this->get_position(context, i) + fan::vec2(0, this->get_size(context, i).y / 2) + rbs.m_store[i].m_properties.theme->button.outline_thickness);

            src.y -= line_height / 2;

            src += rbs.m_store[i].m_properties.offset;
          }
          else if (m_store[i].m_properties.text_position == fan_2d::graphics::gui::text_position_e::middle) {
            auto text_size = tr.get_text_size(context, i);
            src += this->get_position(context, i) + fan::vec2(-text_size.x / 2, 0);
          }

          uint32_t offset = 0;

          auto str = tr.get_text(context, i);

          for (int j = 0; j < y; j++) {
            while (str[offset++] != '\n') {
              if (offset >= str.size()) {
                throw std::runtime_error("string didnt have endline");
              }
            }
          }

          for (int j = 0; j < x; j++) {
            wchar_t wc = str[j + offset];
            if (wc == '\n') {
              continue;
            }

            auto letter_info = fan_2d::opengl::gui::text_renderer_t::get_letter_info(context, fan::utf16_string(wc).to_utf8().data(), font_size);
            src.x += letter_info.metrics.advance;
          }

          src.y += line_height * y;

          dst = fan::vec2(cursor_properties::line_thickness / 2, line_height / 2);

          return { src, dst };
        }

        fan::vec2 get_text_starting_point(fan::opengl::context_t* context, uint32_t i) const
        {
          fan::vec2 src;

          if (m_store[i].m_properties.text_position == fan_2d::graphics::gui::text_position_e::left) {
            src = this->get_position(context, i);
            src.y += tr.font.characters['\n'].metrics.size.y * tr.convert_font_size(context, get_font_size(context, i));
            src += rbs.m_store[i].m_properties.offset;
          }
          else if (m_store[i].m_properties.text_position == fan_2d::graphics::gui::text_position_e::middle) {
            auto text_size = tr.get_text_size(context, tr.get_text(context, i), tr.get_font_size(context, i));
            text_size.y = fan_2d::opengl::gui::text_renderer_t::get_line_height(context, get_font_size(context, i));
            src = this->get_position(context, i) - text_size * 0.5;
          }

          return src;
        }

        properties_t get_property(fan::opengl::context_t* context, uint32_t i) const
        {
          return *(properties_t*)&m_store[i].m_properties;
        }

        void set_offset(fan::opengl::context_t* context, uint32_t i, const fan::vec2& offset) {
          rbs.m_store[i].m_properties.offset = offset;
          tr.set_position(context, i, get_position(context, i) + offset);
        }

        uintptr_t size(fan::opengl::context_t* context) const {
          return rbs.size(context);
        }

        void erase(fan::opengl::context_t* context, uint32_t i)
        {
          rbs.erase(context, i);
          tr.erase(context, i);
          m_store.erase(i);
        }

        void erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end)
        {
          rbs.erase(context, begin, end);
          tr.erase(context, begin, end);
          m_store.erase(begin, end);
        }

        void clear(fan::opengl::context_t* context) {
          rbs.clear(context);
          tr.clear(context);
          m_store.clear();
        }

        void update_theme(fan::opengl::context_t* context, uint32_t i) {
          rbs.update_theme(context, i);
        }

        void set_theme(fan::opengl::context_t* context, uint32_t i, const fan_2d::graphics::gui::theme& theme_) {
          rbs.set_theme(context, i, theme_);
        }

        void enable_draw(fan::opengl::context_t* context) {
          rbs.enable_draw(context);
          tr.enable_draw(context);
        }

        void disable_draw(fan::opengl::context_t* context) {
          rbs.disable_draw(context);
          tr.disable_draw(context);
        }

        fan_2d::graphics::gui::rectangle_box_sized_t rbs;
        fan_2d::opengl::gui::text_renderer_t tr;

        struct p_t {
          fan::utf16_string_ptr_t text;
          fan::utf16_string_ptr_t place_holder;
          f32_t font_size = fan_2d::graphics::gui::defaults::font_size;
          text_position_e text_position = text_position_e::middle;
          void* userptr;
        };

        struct store_t {
          p_t m_properties;
        };

        fan::hector_t<store_t> m_store;
      };
    }
  }
}