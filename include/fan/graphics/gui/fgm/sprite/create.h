case builder_draw_type_t::sprite: {
  switch (key_state) {
    case fan::key_state::press: {
      pile->editor.flags |= flags_t::moving;
      pile->editor.click_position = pile->window.get_mouse_position();

      fan_2d::graphics::sprite_t::properties_t sprite_p;
      sprite_p.position = pile->window.get_mouse_position();
      sprite_p.size = pile->editor.builder_types.get_size(&pile->window, &pile->context, 0);
      sprite_p.image.create_missing_texture(&pile->context);
      sprite_p.texture_coordinates = sprite_p.image.calculate_aspect_ratio(sprite_p.size, 3);
      pile->builder.sprite.push_back(&pile->context, sprite_p);

      pile->editor.depth_map_push(pile, builder_draw_type_t::sprite, pile->builder.sprite.size(&pile->context) - 1);

      pile->editor.builder_draw_type = editor_t::builder_draw_type_t::sprite;
      pile->editor.builder_draw_type_index = pile->builder.sprite.size(&pile->context) - 1;
      pile->editor.selected_type = editor_t::builder_draw_type_t::sprite;
      pile->editor.selected_type_index = pile->editor.builder_draw_type_index;

      break;
    }
    case fan::key_state::release: {
      pile->editor.flags &= ~flags_t::moving;

      switch (pile->editor.builder_draw_type) {
        case builder_draw_type_t::sprite: {

          // if object is not within builder_viewport we will delete it
          if (!pile->editor.is_inside_builder_viewport(
            pile, 
            pile->builder.sprite.get_position(
            &pile->context,
            pile->editor.builder_draw_type_index
            ) +
            pile->builder.sprite.get_size(
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