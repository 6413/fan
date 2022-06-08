case builder_draw_type_t::button: {
  switch (key_state) {
    case fan::key_state::press: {
      pile->editor.flags |= flags_t::moving;
      pile->editor.click_position = pile->window.get_mouse_position();

      fan_2d::graphics::gui::rectangle_text_button_sized_t::properties_t button_p;
      button_p.position = pile->window.get_mouse_position();
      button_p.size = pile->editor.builder_types.get_size(&pile->window, &pile->context, 0);
      button_p.theme = fan_2d::graphics::gui::themes::gray();
      button_p.font_size = constants::gui_size;
      pile->builder.button.push_back(&pile->window, &pile->context, button_p);

      pile->editor.depth_map_push(pile, builder_draw_type_t::button, pile->builder.button.size(&pile->window, &pile->context) - 1);

      pile->editor.builder_draw_type = editor_t::builder_draw_type_t::button;
      pile->editor.builder_draw_type_index = pile->builder.button.size(&pile->window, &pile->context) - 1;
      pile->editor.selected_type = editor_t::builder_draw_type_t::button;
      pile->editor.selected_type_index = pile->editor.builder_draw_type_index;

      break;
    }
    case fan::key_state::release: {
      pile->editor.flags &= ~flags_t::moving;

      switch (pile->editor.builder_draw_type) {
        case builder_draw_type_t::button: {

          // if object is not within builder_viewport we will delete it
          if (!pile->editor.is_inside_builder_viewport(
            pile, 
            pile->builder.button.get_position(
              &pile->window, 
              &pile->context,
              pile->editor.builder_draw_type_index
            ) +
            pile->builder.button.get_size(
              &pile->window, 
              &pile->context,
              pile->editor.builder_draw_type_index
            )
            ))
          {
            switch (pile->editor.builder_draw_type) {
              #include "erase_active.h"
            }
          }
          else {
            pile->editor.close_build_properties(pile);
            click_collision_t click_collision;
            click_collision.builder_draw_type = pile->editor.builder_draw_type;
            click_collision.builder_draw_type_index = pile->editor.builder_draw_type_index;
            pile->editor.open_build_properties(pile, click_collision);
          }
          break;
        }
      }
      break;
    }
  }
  break;
}