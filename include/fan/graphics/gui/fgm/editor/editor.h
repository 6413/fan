void editor_t::open(pile_t* pile) {

  #include "open.h"
  #include "set_callbacks.h"
#include "fgm.h"
}

inline void fan_2d::graphics::gui::fgm::editor_t::push_resize_rectangles(pile_t* pile, click_collision_t click_collision_)
{
  selected_type = click_collision_.builder_draw_type;
  selected_type_index = click_collision_.builder_draw_type_index;

  fan::vec2 positions[8];

  switch (selected_type) {
    #include _FAN_PATH(graphics/gui/fgm/includes/corners.h)
  }

  for (uint32_t i = 0; i < 8; i++) {
    fan_2d::graphics::gui::rectangle_button_t::properties_t p;
    p.position = positions[i];
    p.size = constants::resize_rectangle_size;
    p.theme = fan_2d::graphics::gui::themes::deep_blue(0.8);
    p.theme.button.outline_size = 0.001;
    pile->editor.resize_rectangles.push_back(&pile->window, &pile->context, p);
  }
}

inline void fan_2d::graphics::gui::fgm::editor_t::update_resize_rectangles(pile_t* pile)
{
  if (!pile->editor.resize_rectangles.size(&pile->window, &pile->context)) {
    return;
  }

  fan::vec2 positions[8];
  
  switch (pile->editor.selected_type) {
    #include _FAN_PATH(graphics/gui/fgm/includes/corners.h)
  }

  for (uint32_t i = 0; i < 8; i++) {
    pile->editor.resize_rectangles.set_position(&pile->window, &pile->context, i, positions[i]);
  }
}

inline void fan_2d::graphics::gui::fgm::editor_t::editor_erase_active(pile_t* pile)
{
  for (uint32_t i = 0; i < pile->editor.depth_map.size(); i++) {
    if (pile->editor.depth_map[i].type == pile->editor.selected_type &&
        pile->editor.depth_map[i].index == pile->editor.selected_type_index) {
      pile->editor.depth_map.erase(i);

      for (uint32_t j = i; j < pile->editor.depth_map.size(); j++) {
        if (pile->editor.selected_type != pile->editor.depth_map[j].type) {
          continue;
        }
        pile->editor.depth_map[j].index--;
      }

      switch (pile->editor.selected_type) {
        case builder_draw_type_t::hitbox: {
          pile->editor.hitbox_ids.erase(pile->editor.hitbox_ids.begin() + pile->editor.selected_type_index);
          break;
        }
        case builder_draw_type_t::button: {
          pile->editor.button_ids.erase(pile->editor.button_ids.begin() + pile->editor.selected_type_index);
          break;
        }
      }

      pile->editor.close_build_properties(pile);
      break;
    }
  }
}

inline void fan_2d::graphics::gui::fgm::editor_t::print(pile_t* pile, const std::string& message)
{
  fan::print(message);
}

inline bool fan_2d::graphics::gui::fgm::editor_t::check_for_colliding_hitbox_ids(const std::string& id)
{
  for (uint32_t i = 0; i < hitbox_ids.size(); i++) {
    if (id == hitbox_ids[i]) {
      return true;
    }
  }
  return false;
}

inline bool fan_2d::graphics::gui::fgm::editor_t::check_for_colliding_button_ids(const std::string& id)
{
  for (uint32_t i = 0; i < button_ids.size(); i++) {
    if (id == button_ids[i]) {
      return true;
    }
  }
  return false;
}

inline fan::vec2 fan_2d::graphics::gui::fgm::editor_t::get_mouse_position(pile_t* pile)
{
  return fan::transform_mouse_position(&pile->window);
}

inline void fan_2d::graphics::gui::fgm::editor_t::depth_map_push(pile_t* pile, uint32_t type, uint32_t index)
{
  depth_t depth;
  depth.depth = pile->editor.depth_index++;
  depth.type = type;
  depth.index = index;
  pile->editor.depth_map.push_back(depth);
}

void editor_t::open_build_properties(pile_t* pile, click_collision_t click_collision_)
{
  push_resize_rectangles(pile, click_collision_);

  switch (click_collision_.builder_draw_type) {    
    #include _FAN_PATH(graphics/gui/fgm/includes/open_build_properties.h)
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
  pile->editor.selected_type = fan::uninitialized;
}

bool editor_t::is_inside_builder_viewport(pile_t* pile, const fan::vec2& position)
{
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, 0, builder_viewport_dst);
}

inline bool fan_2d::graphics::gui::fgm::editor_t::is_inside_types_viewport(pile_t* pile, const fan::vec2& position)
{
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, builder_viewport_src, fan::vec2(1, 0.5));
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
      fan::vec2 mp = pile->editor.get_mouse_position(pile);
      uint32_t idx = pile->editor.depth_map[i].index;

      switch (pile->editor.depth_map[i].type) {
        case builder_draw_type_t::sprite: {
          if (!pile->builder.sprite.inside(&pile->context, idx, mp)) {
            continue;
          }
          break;
        }
        case builder_draw_type_t::text_renderer: {
          if (!pile->builder.tr.inside(&pile->context, idx, mp)) {
            continue;
          }
          break;
        }
        case builder_draw_type_t::hitbox: {
          if (!pile->builder.hitbox.inside(&pile->context, idx, mp)) {
            continue;
          }
          break;
        }
        case builder_draw_type_t::button: {
          if (!pile->builder.button.inside(&pile->window, &pile->context, idx, mp)) {
            continue;
          }
          break;
        }
        default: {
          fan::throw_error("not added shape to collision");
        }
      }
      if (frontest < pile->editor.depth_map[i].depth) {
        frontest = pile->editor.depth_map[i].depth;
        front = pile->editor.depth_map[i];
      }
    }
  }

  if (frontest == -1) {
    return false;
  }

  click_collision_->builder_draw_type = front.type;
  click_collision_->builder_draw_type_index = front.index;
  return true;
}