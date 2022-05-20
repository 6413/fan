void editor_draw_types_t::open(pile_t* pile) {

  #include "open.h"
  #include "set_callbacks.h"
}

void editor_draw_types_t::open_build_properties(pile_t* pile, click_collision_t click_collision_)
{
  selected_draw_type = click_collision_.builder_draw_type;
  selected_draw_type_index = click_collision_.builder_draw_type_index;

  switch (click_collision_.builder_draw_type) {    
    #include <fan/graphics/gui/fgm/rectangle_sized_text_button/open_build_properties.h>
    #include <fan/graphics/gui/fgm/text_renderer_clickable/open_build_properties.h>
  }
}

void editor_draw_types_t::close_build_properties(pile_t* pile)
{
  pile->editor_draw_types.properties_button.clear(&pile->window, &pile->context);
  pile->editor_draw_types.properties_button_text.clear(&pile->context);
}

bool editor_draw_types_t::is_inside_builder_viewport(pile_t* pile, const fan::vec2& position)
{
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, 0, builder_viewport_size);
}

bool editor_draw_types_t::is_inside_properties_viewport(pile_t* pile, const fan::vec2& position)
{
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, origin_properties, pile->window.get_size());
}

bool editor_draw_types_t::click_collision(pile_t* pile, click_collision_t* click_collision_)
{
  for (uint32_t i = pile->editor_draw_types.depth_map.size(); i--; ) {

    switch (pile->editor_draw_types.depth_map[i].builder_draw_type) {
      case builder_draw_type_t::rectangle_text_button_sized: {
        if (pile->builder_draw_types.rtbs.inside(&pile->window, &pile->context, pile->editor_draw_types.depth_map[i].builder_draw_type_index, pile->window.get_mouse_position())) {
          click_collision_->builder_draw_type = builder_draw_type_t::rectangle_text_button_sized;
          click_collision_->builder_draw_type_index = pile->editor_draw_types.depth_map[i].builder_draw_type_index;
          return true;
        }
        break;
      }
      case builder_draw_type_t::text_renderer_clickable: {
        if (pile->builder_draw_types.trc.inside(&pile->window, &pile->context, pile->editor_draw_types.depth_map[i].builder_draw_type_index, pile->window.get_mouse_position())) {
          click_collision_->builder_draw_type = builder_draw_type_t::text_renderer_clickable;
          click_collision_->builder_draw_type_index = pile->editor_draw_types.depth_map[i].builder_draw_type_index;
          return true;
        }
        break;
      }
    }
  }
  return false;
}