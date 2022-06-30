#pragma once

#include _FAN_PATH(graphics/gui/rectangle_box_sized.h)
#include _FAN_PATH(graphics/opengl/2D/objects/text_renderer.h)

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

      template <typename T_user_global_data, typename T_user_instance_data>
      struct rectangle_text_box_sized_t {

        using properties_t = rectangle_text_box_sized_properties;

        using box_sized_t = fan_2d::graphics::gui::rectangle_box_sized_t<T_user_global_data, T_user_instance_data>;

        rectangle_text_box_sized_t() = default;

        void open(fan::opengl::context_t* context, fan_2d::graphics::font_t* font, box_sized_t::rect_t::move_cb_t move_cb_, const T_user_global_data& gd)
        {
          rbs.open(context, move_cb_, gd);
          tr.open(context, font);
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
          store.m_properties.offset = property.offset;
          store.m_properties.userptr = property.userptr;

          m_store.push_back(store);

          const auto str = property.place_holder.empty() ? property.text : property.place_holder;

          fan_2d::opengl::text_renderer_t::properties_t text_properties;

          switch (property.text_position) {
            case text_position_e::left:
            {
              fan::vec2 text_size = tr.get_text_size(context, property.text, property.font_size);
              text_properties.text = str;
              text_properties.font_size = property.font_size;
              text_properties.position = fan::vec2(
                property.position.x - property.size.x + text_size.x * 0.5,
                property.position.y
              ) + property.offset;
              text_properties.color = property.place_holder.empty() || str[0] != '\0' ? property.theme.button.text_color : defaults::text_color_place_holder;
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
              text_properties.color = property.text.size() || property.text[0] != '\0' ? property.theme.button.text_color : defaults::text_color_place_holder;

              text_properties.outline_color = property.theme.button.text_outline_color;
              text_properties.outline_size = property.theme.button.text_outline_size;

              tr.push_back(context, text_properties);

              break;
            }
          }

          typename box_sized_t::properties_t rbsp;
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
          switch (m_store[i].m_properties.text_position) {
            case text_position_e::middle: {
              tr.set_position(context, i, position + m_store[i].m_properties.offset);
              break;
            }
            case text_position_e::left: {
              tr.set_position(context, i, fan::vec2(position.x + get_size(context, i).x, position.y) + m_store[i].m_properties.offset);
              break;
            }
          }
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

              fan::vec2 position = get_position(context, i);
              fan::vec2 size = get_size(context, i);
              fan::vec2 text_size = tr.get_text_size(context, i);
              tr.set_position(context, i, 
                fan::vec2(
                  position.x - size.x + text_size.x * 0.5,
                  position.y
                ) + m_store[i].m_properties.offset
              );
              
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
        void set_font_size(fan::opengl::context_t* context, uint32_t i, f32_t font_size) {
          tr.set_font_size(context, i, font_size);
        }
        f32_t get_outline_size(fan::opengl::context_t* context, uint32_t i) const {
          return tr.get_outline_size(context, i);
        }
        void set_outline_size(fan::opengl::context_t* context, uint32_t i, f32_t outline_size) {
          tr.set_outline_size(context, i, outline_size);
        }
        fan::color get_text_color(fan::opengl::context_t* context, uint32_t i, uint32_t j = 0) const {
          return tr.get_text_color(context, i, j);
        }
        void set_text_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color) {
          tr.set_text_color(context, i, color);
        }
        fan::color get_outline_color(fan::opengl::context_t* context, uint32_t i) const {
          return tr.get_outline_color(context, i);
        }
        void set_outline_color(fan::opengl::context_t* context, uint32_t i, const fan::color& outline_color) {
          tr.set_outline_color(context, i, outline_color);
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
          f32_t line_height = fan_2d::opengl::text_renderer_t::get_line_height(context, font_size);

          fan::vec2 src = 0, dst = 0;

          auto text_size = tr.get_text_size(context, i);
          src += tr.get_position(context, i) + fan::vec2(-text_size.x / 2, 0) + rbs.m_store[i].m_properties.offset;

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
            if (j + offset >= str.size()) {
              break;
            }
            wchar_t wc = str[j + offset];
            if (wc == '\n') {
              continue;
            }

            auto letter_info = fan_2d::opengl::text_renderer_t::get_letter_info(context, fan::utf16_string(wc).to_utf8().data(), font_size);
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
            text_size.y = fan_2d::opengl::text_renderer_t::get_line_height(context, get_font_size(context, i));
            src = this->get_position(context, i) - text_size * 0.5;
          }

          return src;
        }

        fan::vec2 set_offset(fan::opengl::context_t* context, uint32_t i) const {
          return rbs.m_store[i].m_properties.offset;
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

        fan_2d::graphics::gui::theme get_theme(fan::opengl::context_t* context, uint32_t i) const {
          return rbs.get_theme(context, i);
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

        void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
          rbs.m_box.m_shader.bind_matrices(context, matrices);
          tr.bind_matrices(context, matrices);
        }

        void* get_userptr(fan::opengl::context_t* context, uint32_t i) {
          return m_store[i].m_properties.userptr;
        }
        void* set_userptr(fan::opengl::context_t* context, uint32_t i, void* userptr) {
          return m_store[i].m_properties.userptr = userptr;
        }

         // IO +

        void write_out(fan::opengl::context_t* context, fan::io::file::file_t* f) const {
				  rbs.write_out(context, f);
          tr.write_out(context, f);
          uint64_t count = m_store.size() * sizeof(store_t);
          fan::io::file::write(f, &count, sizeof(count), 1);
					fan::io::file::write(f, m_store.data(), sizeof(store_t) * m_store.size(), 1);
			  }
        void write_in(fan::opengl::context_t* context, FILE* f) {
          rbs.write_in(context, f);
          tr.write_in(context, f);
          uint64_t count;
          fan::io::file::read(f, &count, sizeof(count), 1);
          m_store.resize(count / sizeof(store_t));
          fan::io::file::read(f, m_store.data(), count, 1);
			  }

        // IO -

        box_sized_t rbs;
        fan_2d::opengl::text_renderer_t tr;

        struct p_t {
          fan::utf16_string_ptr_t text;
          fan::utf16_string_ptr_t place_holder;
          f32_t font_size = fan_2d::graphics::gui::defaults::font_size;
          text_position_e text_position = text_position_e::middle;
          fan::vec2 offset;
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