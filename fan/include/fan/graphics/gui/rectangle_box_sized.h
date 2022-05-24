#pragma once

#include <fan/graphics/gui/themes.h>
#include <fan/graphics/opengl/2D/objects/rectangle.h>
#include <fan/graphics/opengl/2D/gui/text_renderer.h>

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct rectangle_box_sized_properties {
        fan::vec2 position = 0;
        fan_2d::graphics::gui::theme theme = fan_2d::graphics::gui::themes::deep_blue();
        fan::vec2 size = 0;
        fan::vec2 offset = 0;
        void* userptr;
      };

      struct rectangle_box_sized_t {

        using properties_t = rectangle_box_sized_properties;

        rectangle_box_sized_t() = default;

        void open(fan::opengl::context_t* context)
        {
          m_box.open(context);
          m_store.open();
        }

        void close(fan::opengl::context_t* context)
        {
          m_box.close(context);
          for (int i = 0; i < m_store.size(); i++) {
            m_store[i].m_properties.theme.close();
          }
          m_store.close();
        }

        void push_back(fan::opengl::context_t* context, const properties_t& property)
        {
          store_t store;

          store.m_properties.theme.open();

          store.m_properties.offset = property.offset;
          *store.m_properties.theme = property.theme;
          store.m_properties.userptr = property.userptr;

          m_store.push_back(store);

          fan_2d::opengl::rectangle_t::properties_t rect_properties;
          rect_properties.color = property.theme.button.outline_color;
          rect_properties.position = property.position;
          rect_properties.size = property.size;
          m_box.push_back(context, rect_properties);
          rect_properties.color = property.theme.button.color;
          rect_properties.size -= property.theme.button.outline_thickness;
          m_box.push_back(context, rect_properties);
        }

        void draw(fan::opengl::context_t* context)
        {
          m_box.draw(context);
        }

        void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
          m_box.set_position(context, i * 2, position);
          m_box.set_position(context, i * 2 + 1, position);
        }

        bool inside(fan::opengl::context_t* context, uintptr_t i, const fan::vec2& position) const {
          return m_box.inside(context, i * 2, position);
        }

        void set_size(fan::opengl::context_t* context, uint32_t i, const fan::vec2& size) {
          m_box.set_size(context, i * 2, size);
          m_box.set_size(context, i * 2 + 1, size - m_store[i].m_properties.theme->button.outline_thickness);
        }

        fan::vec2 get_position(fan::opengl::context_t* context, uint32_t i) const
        {
          return m_box.get_position(context, i * 2);
        }

        fan::vec2 get_size(fan::opengl::context_t* context, uint32_t i) const
        {
          return m_box.get_size(context, i * 2);
        }

        fan::color get_color(fan::opengl::context_t* context, uint32_t i) const
        {
          return m_box.get_color(context, i * 2);
        }

        properties_t get_property(fan::opengl::context_t* context, uint32_t i) const
        {
          return *(properties_t*)&m_store[i].m_properties;
        }

        uintptr_t size(fan::opengl::context_t* context) const
        {
          return m_box.size(context) / 2;
        }

        void erase(fan::opengl::context_t* context, uint32_t i)
        {
          m_box.erase(context, i * 2, i * 2 + 2);

          m_store.erase(i);
        }

        void erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end)
        {
          m_box.erase(context, begin * 2, end * 2);
          m_store.erase(begin, end);
        }

        void clear(fan::opengl::context_t* context)
        {
          m_box.clear(context);
          m_store.clear();
        }

        void update_theme(fan::opengl::context_t* context, uint32_t i)
        {
          m_box.set_color(context, i * 2, m_store[i].m_properties.theme->button.outline_color);
          m_box.set_color(context, i * 2 + 1, m_store[i].m_properties.theme->button.color);
        }

        void set_theme(fan::opengl::context_t* context, uint32_t i, const fan_2d::graphics::gui::theme& theme_)
        {
          m_box.set_color(context, i * 2, theme_.button.outline_color);
          m_box.set_color(context, i * 2 + 1, theme_.button.color);
          *m_store[i].m_properties.theme.ptr = theme_;
        }

        void set_background_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color) {
          m_box.set_color(context, i * 2, color);
        }
        void set_foreground_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color) {
          m_box.set_color(context, i * 2 + 1, color);
        }

        void enable_draw(fan::opengl::context_t* context)
        {
          m_box.enable_draw(context);
        }

        void disable_draw(fan::opengl::context_t* context)
        {
          m_box.disable_draw(context);
        }

        // IO +

        void write_out(fan::opengl::context_t* context, FILE* f) const {
				  m_box.write_out(context, f);
          uint64_t count = m_store.size() * sizeof(store_t);
          fwrite(&count, sizeof(count), 1, f);
					fwrite(m_store.data(), sizeof(store_t) * m_store.size(), 1, f);
          for (uint32_t i = 0; i < count / sizeof(store_t); i++) {
            fwrite(m_store[i].m_properties.theme.ptr, sizeof(fan_2d::graphics::gui::theme), 1, f);
          }
			  }
        void write_in(fan::opengl::context_t* context, FILE* f) {
          m_box.write_in(context, f);
          uint64_t count;
          fread(&count, sizeof(count), 1, f);
          m_store.resize(count / sizeof(store_t));  
				  fread(m_store.data(), count, 1, f);
          for (uint32_t i = 0; i < count / sizeof(store_t); i++) {
            m_store[i].m_properties.theme.open();
            fread(m_store[i].m_properties.theme.ptr, sizeof(fan_2d::graphics::gui::theme), 1, f);
          }
			  }

        // IO -

        void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
          m_box.bind_matrices(context, matrices);
        }

        fan_2d::opengl::rectangle_t m_box;

      //protected:

        struct p_t {
          theme_ptr_t theme;
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