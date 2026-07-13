export module fan.graphics.editor:selection;

import std;

import fan.types.vector;
import fan.types.color;
import fan.math;
import fan.math.intersection;
import fan.window.input;
import fan.graphics.common_context;
import fan.graphics.shapes;
import fan.graphics.gui.base;

export namespace fan::graphics::editor {
  struct gizmo_t {
    template <typename T>
    bool manipulate(std::vector<T*>& objects, const fan::vec2& camera_pos, f32_t zoom, const fan::vec2& viewport_center, f32_t snap) {
      if (objects.empty()) return false;
      bool changed = false;

      fan::vec2 min_pos = objects[0]->get_position() - objects[0]->get_size();
      fan::vec2 max_pos = objects[0]->get_position() + objects[0]->get_size();
      for (auto* obj : objects) {
        fan::vec2 p = obj->get_position();
        fan::vec2 s = obj->get_size();
        fan::vec2 cmin = p - s, cmax = p + s;
        if (cmin.x < min_pos.x) min_pos.x = cmin.x;
        if (cmin.y < min_pos.y) min_pos.y = cmin.y;
        if (cmax.x > max_pos.x) max_pos.x = cmax.x;
        if (cmax.y > max_pos.y) max_pos.y = cmax.y;
      }
      fan::vec2 group_pos = (min_pos + max_pos) / 2.f;
      fan::vec2 group_size = (max_pos - min_pos) / 2.f;

      auto* draw_list = gui::get_window_draw_list();

      fan::vec2 p_min = (group_pos - group_size - camera_pos) * zoom + viewport_center;
      fan::vec2 p_max = (group_pos + group_size - camera_pos) * zoom + viewport_center;
      draw_list->AddRect(p_min, p_max, fan::color(1.f, 0.5f, 0.f, 1.f).get_gui_color(), 0.0f, 0, 2.0f);

      gui::push_id("shape_drag");
      gui::set_cursor_screen_pos(p_min);
      gui::invisible_button("##shape_drag", p_max - p_min);

      if (gui::is_item_active()) {
        if (!is_dragging) {
          is_dragging = true;
          multi_drag_start.clear();
          multi_drag_start.reserve(objects.size());
          for (auto* obj : objects) {
            multi_drag_start.push_back(obj->get_position());
          }
          drag_mouse_start = gui::get_mouse_pos();
        }
        fan::vec2 mouse_world_delta = (gui::get_mouse_pos() - drag_mouse_start) / zoom;

        if (!fan::window::is_key_down(fan::key_left_shift) && snap > 0.0f) {
          fan::vec2 first_target = fan::vec2(multi_drag_start[0]) + mouse_world_delta;
          first_target.x = std::round(first_target.x / snap) * snap;
          first_target.y = std::round(first_target.y / snap) * snap;
          fan::vec2 snapped_delta = first_target - fan::vec2(objects[0]->get_position());
          if (snapped_delta != fan::vec2(0)) {
            for (auto* obj : objects) {
              obj->set_position(obj->get_position() + fan::vec3(snapped_delta, 0));
            }
          }
        } else {
          for (size_t i = 0; i < objects.size(); ++i) {
            fan::vec2 new_pos = fan::vec2(multi_drag_start[i]) + mouse_world_delta;
            objects[i]->set_position(fan::vec3(new_pos, objects[i]->get_position().z));
          }
        }
        changed = true;
      }
      gui::pop_id();

      if (objects.size() == 1) {
        T& shape = *objects[0];
        fan::vec2 shape_pos = shape.get_position();
        fan::vec2 shape_size = shape.get_size();

        for (int i = 0; i < handle_count; ++i) {
          gui::push_id(i);
          fan::vec2 screen_hp = ((shape_pos + shape_size * handle_dirs[i]) - camera_pos) * zoom + viewport_center;
          gui::set_cursor_screen_pos(screen_hp - handle_size);
          gui::invisible_button("##handle", fan::vec2(handle_size * 2.f));

          bool is_hovered = gui::is_item_hovered();
          bool is_active = gui::is_item_active();

          auto handle_color = is_active ? fan::color(1.f, 0.f, 0.f, 1.f) : (is_hovered ? fan::color(1.f, 0.8f, 0.f, 1.f) : fan::color(1.f, 1.f, 1.f, 1.f));
          draw_list->AddRectFilled(screen_hp - handle_size, screen_hp + handle_size, handle_color.get_gui_color());
          draw_list->AddRect(screen_hp - handle_size, screen_hp + handle_size, fan::colors::black.get_gui_color());

          if (is_active) {
            if (active_handle != i) {
              active_handle = i;
              start_size = shape_size;
              resize_anchor = shape_pos - handle_dirs[i] * start_size;
            }
            fan::vec2 mouse_world = (gui::get_mouse_pos() - viewport_center) / zoom + camera_pos;
            fan::vec2 dir = handle_dirs[i];
            fan::vec2 new_size = (mouse_world - resize_anchor).abs() / 2.f;

            if (dir.x == 0) new_size.x = start_size.x;
            if (dir.y == 0) new_size.y = start_size.y;
            new_size = fan::math::max(new_size, fan::vec2(1.0f));

            if (snap > 0.0f && !fan::window::is_key_down(fan::key_left_shift)) {
              new_size.x = std::round(new_size.x / snap) * snap;
              new_size.y = std::round(new_size.y / snap) * snap;
              new_size = fan::math::max(new_size, fan::vec2(snap));
            }

            fan::vec2 new_center = (mouse_world + resize_anchor) / 2.f;
            if (dir.x == 0) new_center.x = resize_anchor.x;
            if (dir.y == 0) new_center.y = resize_anchor.y;

            shape.set_position(fan::vec3(new_center, shape.get_position().z));
            shape.set_size(new_size);
            changed = true;
          }
          gui::pop_id();
        }
      }

      if (!gui::is_any_item_active()) {
        active_handle = -1;
        is_dragging = false;
      } else {
        is_dragging = true;
      }
      return changed;
    }

    static constexpr int handle_count = 8;
    static constexpr f32_t handle_size = 5.0f;
    static constexpr fan::vec2 handle_dirs[handle_count] = {
      {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}
    };
    int active_handle = -1;
    bool is_dragging = false;
    fan::vec2 drag_mouse_start;
    fan::vec2 resize_anchor;
    fan::vec2 start_size;
    std::vector<fan::vec2> multi_drag_start;
  };

  template <typename GlobalT>
  struct selection_t {
    template <typename FGM>
    void update(FGM& fgm, std::vector<std::unique_ptr<GlobalT>>& shape_list, const fan::vec2& mouse_pos, f32_t zoom) {
      if (fgm.viewport_settings.editor_hovered && fan::window::is_mouse_clicked() && gizmo.active_handle == -1 && !gizmo.is_dragging) {
        drag_start = mouse_pos;
        bool ctrl = fan::window::is_key_down(fan::key_left_control);
        bool hit_gizmo = false;

        if (!ctrl) {
          for (auto* obj : objects) {
            fan::vec2 expanded_size = obj->children[0].get_size() + fan::vec2(gizmo.handle_size / zoom);
            if (fan::math::d2::aabb_point_inside(drag_start, obj->children[0].get_position(), expanded_size)) {
              hit_gizmo = true;
              break;
            }
          }
        }

        if (!hit_gizmo) {
          GlobalT* top_hit_shape = nullptr;
          for (auto& ptr : shape_list) {
            if (fan::math::d2::aabb_point_inside(drag_start, ptr->children[0].get_position(), ptr->children[0].get_size())) {
              top_hit_shape = ptr.get();
            }
          }
          
          if (top_hit_shape) {
            if (ctrl) {
              auto existing = std::find(objects.begin(), objects.end(), top_hit_shape);
              if (existing != objects.end()) {
                top_hit_shape->disable_highlight();
                objects.erase(existing);
              } else {
                top_hit_shape->enable_highlight();
                objects.push_back(top_hit_shape);
              }
              moving_object = true;
            } else {
              if (std::find(objects.begin(), objects.end(), top_hit_shape) == objects.end()) {
                for (auto& sel : objects) { sel->disable_highlight(); }
                objects.clear();
                top_hit_shape->enable_highlight();
                objects.push_back(top_hit_shape);
              }
              moving_object = true;
            }
            if (fgm.current_shape) fgm.current_shape->disable_highlight();
            fgm.current_shape = objects.back();
            fgm.current_shape->enable_highlight();
          } else if (!ctrl) {
            for (auto& sel : objects) { sel->disable_highlight(); }
            objects.clear();
            if (fgm.current_shape) {
              fgm.current_shape->disable_highlight();
              fgm.current_shape = nullptr;
            }
          }
        } else {
          moving_object = true;
        }
      } else if (fgm.viewport_settings.editor_hovered && fan::window::is_mouse_down() && gizmo.active_handle == -1 && !gizmo.is_dragging) {
        if (moving_object && !objects.empty()) {
          fan::vec2 delta = mouse_pos - drag_start;
          if (!fan::window::is_key_down(fan::key_left_shift) && fgm.snap > 0.0f) {
            fan::vec3 pos = objects[0]->get_position() + fan::vec3(delta, 0);
            pos.x = std::round(pos.x / fgm.snap) * fgm.snap;
            pos.y = std::round(pos.y / fgm.snap) * fgm.snap;
            fan::vec2 snapped_delta = fan::vec2(pos) - fan::vec2(objects[0]->get_position());
            if (snapped_delta != fan::vec2(0)) {
              for (auto* obj : objects) { obj->set_position(obj->get_position() + fan::vec3(snapped_delta, 0)); }
              drag_start += snapped_delta;
            }
          } else if (delta != fan::vec2(0)) {
            for (auto* obj : objects) { obj->set_position(obj->get_position() + fan::vec3(delta, 0)); }
            drag_start = mouse_pos;
          }
        } else {
          fan::vec2 size = mouse_pos - drag_start;
          for (auto& i : objects) { i->disable_highlight(); }
          objects.clear();
          drag_box.set_position(drag_start + size / 2);
          drag_box.set_size(size / 2);
        }
      } else if (fan::window::is_mouse_released() && !fgm.shapes_window_hovered && !gui::is_any_item_active() && gizmo.active_handle == -1 && !gizmo.is_dragging) {
        bool hit_any = false;
        for (auto& ptr : shape_list) {
          if (drag_box.intersects(ptr->children[0])) {
            if (!moving_object && (drag_box.get_size().x >= 1 && drag_box.get_size().y >= 1)) {
              ptr->enable_highlight();
              objects.push_back(ptr.get());
            }
          }
          if (fan::math::d2::aabb_point_inside(mouse_pos, ptr->children[0].get_position(), ptr->children[0].get_size())) {
            hit_any = true;
          }
        }
        if (!hit_any && drag_box.get_size().x == 0) {
          for (auto& ptr : shape_list) {
            ptr->disable_highlight();
          }
          objects.clear();
        }
        drag_box.set_size(0);
        moving_object = false;
      }
    }

    std::vector<GlobalT*> objects;
    gizmo_t gizmo;
    fan::graphics::shape_t drag_box;
    fan::vec2 drag_start = 0;
    bool moving_object = false;
  };
}