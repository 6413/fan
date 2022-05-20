#pragma once

#include <fan/graphics/gui.h>

namespace fan_2d {
  namespace graphics {
    namespace gui {
      namespace fgm {

        struct pile_t;

        struct editor_draw_types_t {

          fan::opengl::matrices_t gui_matrices;
          fan::opengl::matrices_t gui_properties_matrices;
          fan::graphics::viewport_t builder_viewport;
          fan::graphics::viewport_t properties_viewport;

          fan::vec2 properties_camera;

          struct constants {
            static constexpr fan::vec2 builder_viewport_size = fan::vec2(600, 600);
            static constexpr f32_t gui_size = 16;
            static constexpr f32_t properties_text_pad = 10;
            static constexpr f32_t scroll_speed = 10;

            static constexpr f32_t properties_box_pad = 160;
          };

          struct flags_t {
            static constexpr uint32_t moving = (1 << 0);
          };

          struct builder_draw_type_t {
            static constexpr uint32_t rectangle_text_button_sized = 0;
            static constexpr uint32_t text_renderer_clickable = 1;
          };

          struct click_collision_t {
            uint32_t builder_draw_type;
            uint32_t builder_draw_type_index;
          };

          bool is_inside_builder_viewport(pile_t* pile, const fan::vec2& position);
          bool is_inside_properties_viewport(pile_t* pile, const fan::vec2& position);

          bool click_collision(pile_t* pile, click_collision_t* click_collision_);

          void open_build_properties(pile_t* pile, click_collision_t click_collision_);
          void close_build_properties(pile_t* pile);

          void open(pile_t* pile);

          fan::vec2 builder_viewport_size;
          fan::vec2 origin_shapes;
          fan::vec2 origin_properties;

          uint32_t builder_draw_type;
          uint32_t builder_draw_type_index;

          uint32_t selected_draw_type;
          uint32_t selected_draw_type_index;

          fan::vec2 moving_position;

          uint8_t flags;

          struct depth_map_t {
            uint32_t builder_draw_type;
            uint32_t builder_draw_type_index;
          };

          fan::hector_t<depth_map_t> depth_map;

          fan_2d::graphics::line_t outline;
          fan_2d::graphics::gui::rectangle_text_button_sized_t builder_types;
          fan_2d::graphics::gui::text_renderer_t properties_button_text;
          fan_2d::graphics::gui::rectangle_text_button_sized_t properties_button;
        };

        struct builder_draw_types_t {

          void open(pile_t* pile);

          fan_2d::graphics::gui::rectangle_text_button_sized_t rtbs;
          uint32_t rtbs_id_counter;

          fan_2d::graphics::gui::text_renderer_clickable trc;
        };

        struct pile_t {

          fan::window_t window;
          fan::opengl::context_t context;

          editor_draw_types_t editor_draw_types;
          builder_draw_types_t builder_draw_types;

          void open() {
            
            window.open();
            context.init();
            context.bind_to_window(&window);
            context.set_viewport(0, window.get_size());
        
            window.add_resize_callback(this, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
              pile_t* pile = (pile_t*)userptr;

              pile->context.set_viewport(0, size);

              fan::vec2 window_size = pile->window.get_size();
              pile->editor_draw_types.gui_matrices.set_ortho(&pile->context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));
              pile->editor_draw_types.gui_properties_matrices.set_ortho(
                &pile->context, 
                fan::vec2(
                  0, 
                  window_size.x - pile->editor_draw_types.origin_properties.x
                ), 
                fan::vec2(
                0, 
                window_size.y - pile->editor_draw_types.origin_properties.y)
              );

              fan::graphics::viewport_t::properties_t vp;

              vp.size = fan::vec2(window_size.x, window_size.y) - pile->editor_draw_types.origin_properties; 
              vp.position = fan::vec2(pile->editor_draw_types.origin_properties.x, 0);

              pile->editor_draw_types.properties_viewport.set(&pile->context, vp);
              vp.position = 0;
              vp.size = window_size;
              pile->editor_draw_types.builder_viewport.set(&pile->context, vp);
            });

            editor_draw_types.gui_matrices.open();
            editor_draw_types.gui_properties_matrices.open();

            builder_draw_types.open(this);
            editor_draw_types.open(this);

            fan::vec2 window_size = window.get_size();
            editor_draw_types.gui_matrices.set_ortho(&context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));
            editor_draw_types.gui_properties_matrices.set_ortho(&context, fan::vec2(0, window_size.x - editor_draw_types.origin_properties.x), fan::vec2(0, window_size.y - editor_draw_types.origin_properties.y));
          }
        };

        #include <fan/graphics/gui/fgm/editor/editor.h>
        #include <fan/graphics/gui/fgm/builder.h>

      }
    }
  }
}