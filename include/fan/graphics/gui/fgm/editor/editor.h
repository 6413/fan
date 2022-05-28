void editor_t::open(pile_t* pile) {

  #include "open.h"
  #include "set_callbacks.h"
}

inline void fan_2d::graphics::gui::fgm::editor_t::update_resize_rectangles(pile_t* pile)
{
  fan::vec2 positions[8];
  
  switch (pile->editor.selected_type) {
    #include <fan/graphics/gui/fgm/rectangle_sized_text_button/corners.h>
    #include <fan/graphics/gui/fgm/text_renderer_clickable/corners.h>
  }

  for (uint32_t i = 0; i < 8; i++) {
    pile->editor.resize_rectangles.set_position(&pile->window, &pile->context, i, positions[i]);
  }
}

void editor_t::open_build_properties(pile_t* pile, click_collision_t click_collision_)
{
  selected_type = click_collision_.builder_draw_type;
  selected_type_index = click_collision_.builder_draw_type_index;

  fan::vec2 positions[8];

  switch (selected_type) {
    #include <fan/graphics/gui/fgm/rectangle_sized_text_button/corners.h>
    #include <fan/graphics/gui/fgm/text_renderer_clickable/corners.h>
  }

  for (uint32_t i = 0; i < 8; i++) {
    fan_2d::graphics::gui::rectangle_button_sized_t::properties_t p;
    p.position = positions[i];
    p.size = constants::resize_rectangle_size;
    pile->editor.resize_rectangles.push_back(&pile->window, &pile->context, p);
  }

  switch (click_collision_.builder_draw_type) {    
    #include <fan/graphics/gui/fgm/rectangle_sized_text_button/open_build_properties.h>
    #include <fan/graphics/gui/fgm/text_renderer_clickable/open_build_properties.h>
    #include "fgm.h"
  }
}

void editor_t::close_build_properties(pile_t* pile)
{
  if (pile->editor.flags & flags_t::ignore_properties_close) {
    return;
  }

  pile->editor.properties_button.clear(&pile->window, &pile->context);
  pile->editor.properties_button_text.clear(&pile->context);
  pile->editor.resize_rectangles.clear(&pile->window, &pile->context);
}

bool editor_t::is_inside_builder_viewport(pile_t* pile, const fan::vec2& position)
{
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, 0, builder_viewport_size);
}

inline bool fan_2d::graphics::gui::fgm::editor_t::is_inside_types_viewport(pile_t* pile, const fan::vec2& position)
{
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, fan::vec2(builder_viewport_size.x, 0), fan::vec2(pile->window.get_size().x, origin_properties.y));
}

bool editor_t::is_inside_properties_viewport(pile_t* pile, const fan::vec2& position)
{
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, origin_properties, pile->window.get_size());
}

bool editor_t::click_collision(pile_t* pile, click_collision_t* click_collision_)
{
  int64_t frontest = -1;

  depth_t front;
  
  for (uint32_t i = 0; i < pile->editor.depth_map.size(); i++) {
    if (pile->editor.depth_map[i].depth > frontest) {
      switch (pile->editor.depth_map[i].type) {
        case builder_draw_type_t::rectangle_text_button_sized: {
          if (!pile->builder.rtbs.inside(&pile->window, &pile->context, pile->editor.depth_map[i].index, pile->window.get_mouse_position())) {
            continue;
          }
          break;
        }
        case builder_draw_type_t::text_renderer_clickable: {
          if (!pile->builder.trc.inside(&pile->window, &pile->context, pile->editor.depth_map[i].index, pile->window.get_mouse_position())) {
            continue;
          }
          break;
        }
      }
      frontest = pile->editor.depth_map[i].depth;
      front = pile->editor.depth_map[i];
    }
  }

  if (frontest == -1) {
    return false;
  }

  click_collision_->builder_draw_type = front.type;
  click_collision_->builder_draw_type_index = front.index;
  return true;
}