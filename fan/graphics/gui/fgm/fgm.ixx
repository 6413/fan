module;

#if defined(FAN_GUI)

#include <fan/utility.h>

export module fan.graphics.editor;

import std;

import fan.types;
import fan.graphics;
import fan.graphics.gui.types;
import fan.graphics.gui.base;
import fan.graphics.gui;
import fan.file_dialog;
import fan.types.fstring;
import fan.physics.types;
import fan.memory;
import fan.io.file;

import fan.physics.b2_integration;
import fan.graphics.physics_shapes;
import :viewport;
import :selection;
import :properties_ui;
import :animation_system;
import :scene_serializer;
import :fgm_types;

export namespace fan::graphics::editor {
  struct fgm_t {
    fgm_t() = default;

    static constexpr f32_t scroll_speed = 1.2f;
    static constexpr auto max_depth = 0xff;
    static constexpr int max_path_input = 40;
    static constexpr int max_id_input = 20;
    static constexpr fan::vec2 default_button_size {100, 30};

    struct texturepack_image_t {
      fan::graphics::image_t image;
      fan::vec2 uv0;
      fan::vec2 uv1;
      std::wstring image_name;
      f32_t aspect_ratio;
    };

    f32_t snap = 32.f;

    fan::graphics::material_system_t material_system;

    void open_texturepack(const std::string& path) {
      gloco()->texture_pack.open_compiled(path);
      texturepack_images.clear();
      texturepack_images.reserve(gloco()->texture_pack.size());

      gloco()->texture_pack.iterate_loaded_images([this](auto& image) {
        texturepack_image_t tp_image;
        tp_image.image = gloco()->texture_pack.get_pixel_data(image.unique_id).image;
        auto& img_data = gloco()->image_get_data(gloco()->texture_pack.get_pixel_data(image.unique_id).image);
        fan::vec2 size = img_data.size;
        tp_image.uv0 = fan::vec2(image.position) / size;
        tp_image.uv1 = fan::vec2(tp_image.uv0) + fan::vec2(image.size) / size;
        tp_image.image_name = {image.name.begin(), image.name.end()};
        tp_image.aspect_ratio = (f32_t)image.size.x / image.size.y;
        texturepack_images.push_back(tp_image);
        texturepack_size = texturepack_size.max(fan::vec2(size));
        texturepack_single_image_size = texturepack_single_image_size.max(fan::vec2(image.size));
      });
    }

    void open(const std::string& texturepack_name, const std::string& asset_path) {
      content_browser.init(asset_path);
      content_browser.current_view_mode = gui::content_browser_t::view_mode_large_thumbnails;

      render_view.camera = gloco()->open_camera(fan::vec2(0, 1), fan::vec2(0, 1));
      render_view.viewport = gloco()->open_viewport(fan::vec2(0), fan::vec2(1));

      background = fan::graphics::sprite_t {{
        .render_view = &render_view,
        .position = 0,
        .size = 0,
        .image = gloco()->create_transparent_texture(),
      }};

      if (texturepack_name.size()) {
        open_texturepack(texturepack_name);
      }

      white_texture = fan::graphics::image_create(fan::colors::white);

      key_handle = gloco()->window.add_keys_callback([this](const auto& d) {
        if (d.state != fan::keyboard_state::press || gui::is_any_item_active()) return;
        if (d.key == fan::key_r && !is_playing) erase_selected();
      });

      gloco()->input.input_action.add_keycombo({fan::input::key_left_control, fan::input::key_space}, "toggle_content_browser");
      gloco()->input.input_action.add_keycombo({fan::input::key_left_control, fan::input::key_f}, "set_windowed_fullscreen");
      gloco()->input.input_action.add_keycombo({fan::input::key_left_control, fan::input::key_s}, "save_file");

      mouse_move_handle = gloco()->window.add_mouse_move_callback([this](const auto& d) {
        if (viewport_settings.move) {
          fan::vec2 move_off = (d.position - viewport_settings.offset) / fan::graphics::camera_get_zoom(render_view.camera);
          gloco()->camera_set_position(render_view.camera, viewport_settings.pos - move_off);
        }
      });

      button_handle = gloco()->window.add_buttons_callback([this](const auto& d) {
        if (gui::is_any_item_active()) return;
        switch (d.button) {
          case fan::mouse_middle: {
            viewport_settings.move = (bool)d.state;
            viewport_settings.offset = gloco()->get_mouse_position();
            viewport_settings.pos = gloco()->camera_get_position(render_view.camera);
            break;
          }
          case fan::mouse_scroll_up:
          case fan::mouse_scroll_down: {
            if (viewport_settings.editor_hovered) {
              f32_t current_zoom = fan::graphics::camera_get_zoom(render_view.camera);
              f32_t new_zoom = d.button == fan::mouse_scroll_up ? current_zoom * scroll_speed : current_zoom / scroll_speed;
              fan::graphics::camera_set_zoom(render_view.camera, new_zoom);
              update_line_thickness();
            }
            break;
          }
        }
      });

      axis_lines[0] = fan::graphics::line_t {{
        .render_view = &render_view,
        .src = fan::vec3(-0xfffff, 0, 0x1fff),
        .dst = fan::vec2(0xfffff, 0),
        .color = axis_x_color
      }};

      axis_lines[1] = fan::graphics::line_t {{
        .render_view = &render_view,
        .src = fan::vec3(0, -0xfffff, 0x1fff),
        .dst = fan::vec2(0, 0xfffff),
        .color = axis_y_color
      }};

      selection.drag_box = fan::graphics::rectangle_t {{
        .render_view = &render_view,
        .position = fan::vec3(0, 0, 0xffff - 0xff),
        .size = 0,
        .color = fan::color::from_rgba(0x3eb9ff44),
        .blending = true
      }};
    }

    void close() {
      shape_list.clear();
      physics_bodies.clear();
      segment_bodies.clear();
      segment_drag_idx = -1;
      segment_drag_shape = nullptr;
      close_cb();
      shape_original_json.clear();
      is_playing = false;
      scene_backup.clear();
    }

    void update_line_thickness() {
      f32_t line_thickness = std::max(2.0f / fan::graphics::camera_get_zoom(render_view.camera), 2.0f);
      axis_lines[0].set_thickness(line_thickness);
      axis_lines[1].set_thickness(line_thickness);
    }

    bool id_exists(const std::string& id) {
      for (auto& ptr : shape_list) {
        if (ptr->id == id) return true;
      }
      return false;
    }

    template <typename ShapeT>
    std::size_t push_shape(const fan::vec2& pos, const fan::vec2& size = 128) {
      constexpr std::uint16_t st =
        std::is_same_v<ShapeT, fan::graphics::sprite_t> ? fan::graphics::shapes::shape_type_t::sprite :
        std::is_same_v<ShapeT, fan::graphics::unlit_sprite_t> ? fan::graphics::shapes::shape_type_t::unlit_sprite :
        std::is_same_v<ShapeT, fan::graphics::rectangle_t> ? fan::graphics::shapes::shape_type_t::rectangle :
        fan::graphics::shapes::shape_type_t::light;

      auto node = std::make_unique<shapes_t::global_t>(st,
        ShapeT{{.render_view = &render_view, .position = pos, .size = size}},
        current_z, current_shape);

      if constexpr (std::is_same_v<ShapeT, fan::graphics::sprite_t>) {
        auto* ri = ((fan::graphics::shapes::sprite_t::ri_t*)node->children[0].GetData(fan::graphics::g_shapes->shaper));
        animations_application.current_animation_nr = ri->sprite_sheet_data.current_sprite_sheet;
        animations_application.current_animation_shape_nr = ri->sprite_sheet_data.shape_sprite_sheets;
      }

      if constexpr (std::is_same_v<ShapeT, fan::graphics::light_t>) {
      }

      shape_list.push_back(std::move(node));
      return shape_list.size() - 1;
    }

    void render_tree_with_unified_selection() {
      static int selection_mask = 0;
      int node_clicked = -1;
      static gui::tree_node_flags_t base_flags = gui::tree_node_flags_open_on_arrow | gui::tree_node_flags_open_on_double_click | gui::tree_node_flags_span_avail_width;

      for (std::size_t idx = 0; idx < shape_list.size(); ++idx) {
        auto& shape_instance = shape_list[idx];
        gui::tree_node_flags_t node_flags = base_flags;
        if ((selection_mask & (1 << idx)) != 0) node_flags |= gui::tree_node_flags_selected;
        if (shape_instance->children.size() <= 1) node_flags |= gui::tree_node_flags_leaf;

        std::string_view shape_name = shape_instance->children.empty() ? std::string_view("Node") : fan::graphics::shape_names[shape_instance->children[0].get_shape_type()];
        if (shape_instance->physics.collision_shape == 1) {
          shape_name = "Segment Collider";
        }
        bool node_open = gui::tree_node_ex((void*)(std::intptr_t)idx, node_flags, "%.*s %ld", static_cast<int>(shape_name.length()), shape_name.data(), (std::intptr_t)idx);

        if (gui::is_item_clicked() && !gui::is_item_toggled_open()) {
          node_clicked = (std::intptr_t)idx;
          if (current_shape) current_shape->disable_highlight();
          current_shape = shape_instance.get();
          current_shape->enable_highlight();
        }

        if (node_open) {
          if (shape_instance->children.size() > 1) {
            render_child_nodes(node_clicked, shape_instance->children, selection_mask, base_flags);
          }
          gui::tree_pop();
        }
      }

      if (node_clicked != -1) {
        selection_mask = gui::get_io().KeyCtrl ? (selection_mask ^ (1 << node_clicked)) : (1 << node_clicked);
      }
    }

    void render_child_nodes(int& node_clicked, std::vector<fan::graphics::shapes::shape_t>& children, int& selection_mask, gui::tree_node_flags_t base_flags) {
      int child_index = 0;
      for (auto& child : children) {
        gui::tree_node_flags_t node_flags = base_flags;
        if ((selection_mask & (1 << (std::intptr_t)child.NRI)) != 0) node_flags |= gui::tree_node_flags_selected;
        if (child_index + 1 >= children.size()) node_flags |= gui::tree_node_flags_leaf;

        bool node_open = gui::tree_node_ex((void*)(std::intptr_t)child.NRI, node_flags, "%s %u", fan::graphics::shape_names[child.get_shape_type()], child.NRI);

        if (gui::is_item_clicked() && !gui::is_item_toggled_open()) {
            node_clicked = (std::intptr_t)child.NRI;
            for (auto& ptr : shape_list) {
              if (ptr->children[0] == child) {
                if (current_shape) current_shape->disable_highlight();
                current_shape = ptr.get();
                current_shape->enable_highlight();
                break;
              }
            }
        }
        if (node_open) gui::tree_pop();
        child_index++;
      }
    }

    void render_viewport(f32_t zoom) {
      if (!gui::begin("Editor", nullptr, gui::window_flags_menu_bar | gui::window_flags_no_background | gui::window_flags_no_scrollbar | gui::window_flags_no_scroll_with_mouse)) { gui::end(); return; }

      gui::push_style_color(gui::col_button, is_playing ? fan::color(0.8f, 0.2f, 0.2f, 1.f) : fan::color(0.2f, 0.7f, 0.2f, 1.f));
      if (gui::button(is_playing ? "Stop" : "Play", fan::vec2(100, 50))) {
        is_playing = !is_playing;
        if (is_playing) {
          selection.objects.clear();
          for (auto& ptr : shape_list) {
            ptr->dynamic_props.base_color = ptr->get_color();
          }
          invalidate_current();
          scene_backup = scene_serializer_t::save_to_string(*this);
          init_physics_scene();
        } else {
          destroy_physics_scene();
          std::string backup = scene_backup;
          close();
          scene_serializer_t::load_from_string(*this, backup);
        }
      }
      gui::pop_style_color();
      gui::same_line();

      fan::vec2 viewport_size = gui::get_window_size();
      editor_pos = gui::get_window_pos();

      fan::vec2 camera_pos = gloco()->camera_get_position(render_view.camera);
      auto world_size = viewport_size / zoom;

      background.set_position(camera_pos);
      background.set_size(world_size / 2);
      f32_t tile_size = 128.0f;
      fan::vec2 tc_size = world_size / tile_size;
      background.set_tc_size(tc_size);
      fan::vec2 tc_offset = camera_pos / tile_size;
      tc_offset.x -= std::floor(tc_offset.x);
      tc_offset.y -= std::floor(tc_offset.y);
      background.set_tc_position(tc_offset - tc_size / 2);

      viewport_settings.editor_hovered = gui::is_window_hovered();

      auto& style = gui::get_style();
      fan::vec2 vMin = gui::get_window_content_region_min() + gui::get_window_pos();
      fan::vec2 vMax = gui::get_window_content_region_max() + gui::get_window_pos();

      viewport_size = vMax - vMin + fan::vec2(style.WindowPadding) * 2;
      gloco()->viewport_set(render_view.viewport, vMin - fan::vec2(style.WindowPadding), viewport_size);
      gloco()->camera_set_ortho(render_view.camera, fan::vec2(-viewport_size.x / 2, viewport_size.x / 2), fan::vec2(-viewport_size.y / 2, viewport_size.y / 2));
      viewport_settings.size = viewport_size;
      viewport_settings.start_pos = vMin;

      if (!is_playing && viewport_settings.editor_hovered) {
        fan::vec2 mouse_world_seg = viewport_t::get_mouse_position(viewport_settings, gui::get_mouse_pos(), zoom, fan::vec2(style.WindowPadding));

        if (segment_drag_idx >= 0 && segment_drag_shape != nullptr) {
          f32_t inv_sa = std::sin(-segment_drag_shape->children[0].get_angle().z);
          f32_t inv_ca = std::cos(-segment_drag_shape->children[0].get_angle().z);
          fan::vec2 delta = mouse_world_seg - fan::vec2(segment_drag_shape->get_position());
          fan::vec2 local_mouse(inv_ca * delta.x - inv_sa * delta.y, inv_sa * delta.x + inv_ca * delta.y);
          auto& pts = segment_drag_shape->physics.segment_points;

          if (segment_drag_idx < (int)pts.size()) {
            if (fan::window::is_mouse_down(fan::mouse_left)) {
              pts[segment_drag_idx] = local_mouse;
            } else {
              segment_drag_idx = -1;
              segment_drag_shape = nullptr;
            }
          } else {
            segment_drag_idx = -1;
            segment_drag_shape = nullptr;
          }
        } else {
          segment_drag_idx = -1;
          segment_drag_shape = nullptr;
          if (fan::window::is_mouse_clicked(fan::mouse_left) && selection.gizmo.active_handle == -1) {
            int global_closest = -1;
            f32_t global_closest_d = 12.f;
            shapes_t::global_t* best_shape = nullptr;

            for (auto& shape_ptr : shape_list) {
              if (!shape_ptr->physics.enabled || shape_ptr->physics.collision_shape != 1) continue;
              
              f32_t inv_sa = std::sin(-shape_ptr->children[0].get_angle().z);
              f32_t inv_ca = std::cos(-shape_ptr->children[0].get_angle().z);
              fan::vec2 delta = mouse_world_seg - fan::vec2(shape_ptr->get_position());
              fan::vec2 local_mouse(inv_ca * delta.x - inv_sa * delta.y, inv_sa * delta.x + inv_ca * delta.y);
              auto& pts = shape_ptr->physics.segment_points;
              
              for (std::size_t i = 0; i < pts.size(); ++i) {
                f32_t d = (pts[i] - local_mouse).length();
                if (d < global_closest_d) { 
                  global_closest_d = d; 
                  global_closest = (int)i; 
                  best_shape = shape_ptr.get();
                }
              }
            }

            if (best_shape) {
              auto& pts = best_shape->physics.segment_points;
              if (fan::window::is_key_down(fan::key_left_control)) {
                pts.erase(pts.begin() + global_closest);
              } else {
                segment_drag_idx = global_closest;
                segment_drag_shape = best_shape;
              }
            } else if (current_shape && current_shape->physics.enabled && current_shape->physics.collision_shape == 1) {
              if (fan::window::is_key_down(fan::key_left_shift)) {
                f32_t inv_sa = std::sin(-current_shape->children[0].get_angle().z);
                f32_t inv_ca = std::cos(-current_shape->children[0].get_angle().z);
                fan::vec2 delta = mouse_world_seg - fan::vec2(current_shape->get_position());
                fan::vec2 local_mouse(inv_ca * delta.x - inv_sa * delta.y, inv_sa * delta.x + inv_ca * delta.y);
                current_shape->physics.segment_points.push_back(local_mouse);
              }
            }
          }
        }
      }

      if (!selection.objects.empty()) {
        fan::vec2 viewport_center = viewport_settings.start_pos - fan::vec2(style.WindowPadding) + viewport_settings.size / 2.f;
        if (segment_drag_idx >= 0) {
          selection.gizmo.is_dragging = false;
        } else {
          selection.gizmo.manipulate(selection.objects, camera_pos, zoom, viewport_center, snap);
        }

        for (auto* obj : selection.objects) {
          if (!obj->physics.enabled) continue;
          fan::vec2 sz = obj->get_size() * obj->physics.hitbox_size;
          fan::vec2 pos = obj->get_position();
          f32_t angle = obj->children[0].get_angle().z;
          auto* dl = gui::get_window_draw_list();
          auto c = fan::color(0.f, 1.f, 0.3f, 1.f).get_gui_color();
          f32_t ca = std::cos(angle), sa = std::sin(angle);
          auto to_screen = [&](const fan::vec2& p) -> fan::vec2 {
            return ((p - camera_pos) * zoom + viewport_center);
          };
          auto rotate = [&](const fan::vec2& local) -> fan::vec2 {
            return pos + fan::vec2(ca * local.x - sa * local.y, sa * local.x + ca * local.y);
          };
          if (obj->physics.collision_shape == 1) {
            auto& pts = obj->physics.segment_points;
            auto pc = fan::color(1.f, 0.8f, 0.f, 1.f).get_gui_color();
            for (std::size_t pi = 0; pi + 1 < pts.size(); ++pi) {
              fan::vec2 a = to_screen(rotate(pts[pi]));
              fan::vec2 b = to_screen(rotate(pts[pi + 1]));
              dl->AddLine(a, b, c, 2.f);
            }
            if (pts.size() > 2) {
              fan::vec2 a = to_screen(rotate(pts.back()));
              fan::vec2 b = to_screen(rotate(pts.front()));
              dl->AddLine(a, b, c, 2.f);
            }
            for (auto& pt : pts) {
              dl->AddCircleFilled(to_screen(rotate(pt)), 4.f, pc);
            }
          }
          else switch (obj->physics.shape_type) {
            case 0: {
              fan::vec2 corners[4];
              for (int i = 0; i < 4; ++i) {
                fan::vec2 local((i & 1) ? sz.x : -sz.x, (i & 2) ? sz.y : -sz.y);
                corners[i] = to_screen(rotate(local));
              }
              dl->AddLine(corners[0], corners[1], c, 2.f);
              dl->AddLine(corners[1], corners[3], c, 2.f);
              dl->AddLine(corners[3], corners[2], c, 2.f);
              dl->AddLine(corners[2], corners[0], c, 2.f);
              break;
            }
            case 1: {
              fan::vec2 center = to_screen(pos);
              dl->AddCircle(center, sz.max() * zoom, c, 0, 2.f);
              break;
            }
            case 2: {
              f32_t half_h = sz.y;
              f32_t cap_r = sz.x;
              f32_t body_h = half_h - cap_r;
              fan::vec2 cw0 = rotate(fan::vec2(0, -body_h));
              fan::vec2 cw1 = rotate(fan::vec2(0, body_h));
              fan::vec2 dir = (cw1 - cw0).normalize();
              fan::vec2 perp(-dir.y, dir.x);
              f32_t a0 = std::atan2(dir.y, dir.x);
              auto arc = [&](fan::vec2 center, f32_t a_begin, f32_t a_end, fan::vec2* out_first, fan::vec2* out_last) {
                constexpr int segs = 10;
                fan::vec2 prev;
                for (int j = 0; j <= segs; ++j) {
                  f32_t t = (f32_t)j / segs;
                  f32_t ang = a_begin + (a_end - a_begin) * t;
                  fan::vec2 p = to_screen(center + fan::vec2(std::cos(ang), std::sin(ang)) * cap_r);
                  if (j == 0) *out_first = p;
                  if (j == segs) *out_last = p;
                  if (j > 0) dl->AddLine(prev, p, c, 2.f);
                  prev = p;
                }
              };
              fan::vec2 arc0_first, arc0_last, arc1_first, arc1_last;
              arc(cw0, a0 + fan::math::half_pi, a0 + fan::math::pi * 1.5f, &arc0_first, &arc0_last);
              dl->AddLine(arc0_last, to_screen(cw1 - perp * cap_r), c, 2.f);
              arc(cw1, a0 - fan::math::half_pi, a0 + fan::math::half_pi, &arc1_first, &arc1_last);
              dl->AddLine(arc1_last, arc0_first, c, 2.f);
              break;
            }
          }
        }
      }

      { // Always draw segment colliders so they are visible even when not focused
        auto* dl = gui::get_window_draw_list();
        fan::vec2 viewport_center = viewport_settings.start_pos - fan::vec2(style.WindowPadding) + viewport_settings.size / 2.f;
        for (auto& ptr : shape_list) {
          if (!ptr->physics.enabled || ptr->physics.collision_shape != 1) continue;
          fan::vec2 pos = ptr->get_position();
          f32_t angle = ptr->children[0].get_angle().z;
          f32_t ca = std::cos(angle), sa = std::sin(angle);
          auto to_screen = [&](const fan::vec2& p) -> fan::vec2 {
            return ((p - camera_pos) * zoom + viewport_center);
          };
          auto rotate = [&](const fan::vec2& local) -> fan::vec2 {
            return pos + fan::vec2(ca * local.x - sa * local.y, sa * local.x + ca * local.y);
          };
          auto& pts = ptr->physics.segment_points;
          auto lc = fan::color(0.f, 1.f, 0.3f, 1.f).get_gui_color();
          auto pc = fan::color(1.f, 0.8f, 0.f, 1.f).get_gui_color();
          for (std::size_t pi = 0; pi + 1 < pts.size(); ++pi) {
            fan::vec2 a = to_screen(rotate(pts[pi]));
            fan::vec2 b = to_screen(rotate(pts[pi + 1]));
            dl->AddLine(a, b, lc, 2.f);
          }
          if (pts.size() > 2) {
            dl->AddLine(to_screen(rotate(pts.back())), to_screen(rotate(pts.front())), lc, 2.f);
          }
          for (auto& pt : pts) {
            dl->AddCircleFilled(to_screen(rotate(pt)), 4.f, pc);
          }
        }
      }

      if (current_shape && current_shape->physics.enabled && current_shape->physics.collision_shape == 1) {
        auto* dl = gui::get_window_draw_list();
        fan::vec2 viewport_center = viewport_settings.start_pos - fan::vec2(style.WindowPadding) + viewport_settings.size / 2.f;
        fan::vec2 pos = current_shape->get_position();
        f32_t angle = current_shape->children[0].get_angle().z;
        f32_t ca = std::cos(angle), sa = std::sin(angle);
        auto to_screen = [&](const fan::vec2& p) -> fan::vec2 {
          return ((p - camera_pos) * zoom + viewport_center);
        };
        auto rotate = [&](const fan::vec2& local) -> fan::vec2 {
          return pos + fan::vec2(ca * local.x - sa * local.y, sa * local.x + ca * local.y);
        };
        auto& pts = current_shape->physics.segment_points;
        auto c = fan::color(1.f, 0.8f, 0.f, 1.f).get_gui_color();
        for (std::size_t i = 0; i < pts.size(); ++i) {
          auto sc = to_screen(rotate(pts[i]));
          f32_t r = (int)i == segment_drag_idx ? 6.f : 4.f;
          dl->AddCircleFilled(sc, r, c);
        }
      }
      if (current_shape && current_shape->physics.enabled && current_shape->physics.collision_shape == 1) {
        gui::push_style_color(gui::col_text, fan::color(1.f, 0.8f, 0.f, 1.f));
        gui::text("[Segment Edit] ", {.offset = {8, 8}});
        gui::same_line();
        gui::text("Click: add point | Drag: move point | Ctrl+Click: delete point");
        gui::pop_style_color();
      }

      gui::text(fan::to_string(zoom * 100) + " %", {.offset = {0.f, -gui::get_text_line_height_with_spacing()}, .align = align_e::bottom_right});
      fan::vec2 cursor_pos = (gui::get_mouse_pos() - viewport_settings.start_pos + fan::vec2(style.WindowPadding)) - viewport_settings.size / 2;
      std::string cursor_pos_str = cursor_pos.to_string(1);
      gui::text(cursor_pos_str.substr(1, cursor_pos_str.size() - 2), {.align = align_e::bottom_right});
      gui::set_cursor_pos(gui::get_cursor_start_pos());

      {
        content_browser.receive_drag_drop_target([&](const std::string& file) {
          if (fan::io::file::extension(file) == ".json") {
            fin(file);
          } else {
            auto image = gloco()->image_load(file);
            fan::vec2 original_size = gloco()->image_get_data(image).size;
            shape_list[push_shape<fan::graphics::sprite_t>(0, fan::vec2(128.f * (original_size.x / original_size.y), 128.f))]->children[0].set_image(image);
          }
        });

        gui::receive_drag_drop_target("FGM_TEXTUREPACK_DROP", [&](const std::string& path) {
          fan::graphics::texture_pack_t::ti_t ti;
          if (gloco()->texture_pack.qti(path, &ti)) {
            gui::print("non texturepack texture or failed to load texture:", path);
          } else {
            std::wstring wpath(path.begin(), path.end());
            auto found = std::find_if(texturepack_images.begin(), texturepack_images.end(), [&](const texturepack_image_t& img) { return wpath == img.image_name; });
            if (found != texturepack_images.end()) {
              auto& node = shape_list[push_shape<fan::graphics::sprite_t>(0, fan::vec2(128.f * found->aspect_ratio, 128.f))];
              node->children[0].load_tp(&ti);
              node->children[0].get_image_data().image_path = path;
            }
          }
        });

        if (fan::window::is_key_down(fan::key_left_control) && fan::window::is_key_clicked(fan::key_d)) {
          for (auto& i : selection.objects) {
            for (auto& child : i->children) {
              auto& dup = shape_list.emplace_back(std::make_unique<shapes_t::global_t>(child.get_shape_type(), child, current_z, current_shape));
              dup->physics = i->physics;
              dup->material_type = i->material_type;
              dup->original_image = i->original_image;
            }
          }
          selection.objects.clear();
        }

        if (gui::begin_menu_bar()) {
          if (gui::begin_menu("File")) {
            if (current_shape) selection.moving_object = false;
            if (gui::menu_item("Open..", "Ctrl+O")) {
              fan::graphics::open_file("json;fmm", [&](std::string_view p) { close(); fin(std::string(p)); }, []{});
            }
            if (gui::menu_item("Save", "Ctrl+S")) fout(previous_filename);
            if (gui::menu_item("Save as", "Ctrl+Shift+S")) {
              fan::graphics::save_file("json;fmm", [&](std::string_view p) { fout(std::string(p)); }, []{});
            }
            if (gui::menu_item("Quit")) {
              close();
              gui::end();
            }
            gui::end_menu();
          }
          gui::end_menu_bar();
        }
      }
      gui::end();
    }

    void render_settings_window() {
      if (!gui::begin("Settings")) { gui::end(); return; }
      gui::color_edit3("background", &gloco()->renderer_state.clear_color);
      gui::color_edit3("ambient", &gloco()->renderer_state.lighting.ambient);
      gui::drag("grid snap", &snap, 1, 0, std::numeric_limits<f32_t>::max(), gui::slider_flags_always_clamp);
      if (gui::checkbox("render axes", &render_axis_lines)) {
        axis_lines[0].set_color(axis_x_color * render_axis_lines);
        axis_lines[1].set_color(axis_y_color * render_axis_lines);
      }
      gui::end();
    }

    void render_shapes_window() {
      if (!gui::begin("Shapes", nullptr)) { gui::end(); return; }
      shapes_window_hovered = gui::is_window_hovered(gui::hovered_flags_allow_when_blocked_by_active_item);
      if (shapes_window_hovered && fan::window::is_mouse_clicked(fan::mouse_right)) gui::open_popup("ContextMenu");
      if (gui::begin_popup("ContextMenu")) {
        if (gui::begin_menu("Create")) {
          if (gui::begin_menu("Shapes")) {
            if (gui::menu_item("Sprite")) push_shape<fan::graphics::sprite_t>(0);
            if (gui::menu_item("Segment Collider")) {
              auto idx = push_shape<fan::graphics::rectangle_t>(0, fan::vec2(128));
              auto& node = *shape_list[idx];
              node.physics.enabled = true;
              node.physics.collision_shape = 1;
              node.physics.segment_points = { fan::vec2(-64, -64), fan::vec2(64, -64), fan::vec2(64, 64), fan::vec2(-64, 64) };
              node.children[0].set_color(fan::color(0, 0, 0, 0));
            }
            gui::end_menu();
          }
          if (gui::begin_menu("Lights")) {
            if (gui::menu_item("Point")) push_shape<fan::graphics::light_t>(0);
            gui::end_menu();
          }
          gui::end_menu();
        }
        gui::end_popup();
      }
      render_tree_with_unified_selection();
      gui::end();
    }

    void render_texturepack_window() {
      if (!gui::begin("Texture Pack")) { gui::end(); return; }
      if (gui::button("open")) {
        fan::graphics::open_file("ftp", [&](std::string_view p) { open_texturepack(std::string(p)); });
      }
      f32_t thumbnail_size = 128.0f;
      int column_count = std::max((int)(gui::get_content_region_avail().x / (thumbnail_size + 16.0f)), 1);
      gui::columns(column_count, 0, false);
      gui::push_style_color(gui::col_button, fan::color(0.f, 0.f, 0.f, 0.f));
      gui::push_style_color(gui::col_button_active, fan::color(0.f, 0.f, 0.f, 0.f));
      gui::push_style_color(gui::col_button_hovered, fan::color(0.3f, 0.3f, 0.3f, 0.3f));
      int idx = 0;
      for (auto& i : texturepack_images) {
        gui::push_id(idx++);
        gui::image_button("##", i.image, fan::vec2(thumbnail_size, thumbnail_size / i.aspect_ratio), i.uv0, i.uv1);
        gui::send_drag_drop_item("FGM_TEXTUREPACK_DROP", i.image_name);
        gui::next_column();
        gui::pop_id();
      }
      gui::pop_style_color(3);
      gui::end();
    }

    void render_animations_window() {
      if (!gui::begin("Shape Animations")) { gui::end(); return; }
      if (current_shape && current_shape->children.size() > 0) {
        auto it = shape_sprite_sheets.find(current_shape);
        if (it == shape_sprite_sheets.end()) {
          shape_sprite_sheets[current_shape].owner_shape = current_shape;
          it = shape_sprite_sheets.find(current_shape);
        }
        it->second.render_gui(current_shape);
      } else {
        gui::text("Select a shape to animate");
      }
      gui::end();
    }

    void render_animation_updates() {
      for (auto& [shape, anim] : shape_sprite_sheets) {
        if (anim.is_playing && anim.owner_shape && !anim.keyframes.empty()) {
          anim.update(gloco()->get_delta_time());
          anim.apply_to_shape(anim.owner_shape);
        }
      }
      if (is_playing) {
        for (auto& shape_ptr : shape_list) {
          if (shape_ptr->shape_type == fan::graphics::shapes::shape_type_t::light && shape_ptr->light_props.enable_flicker) {
            f32_t t = std::fmod(gloco()->time * shape_ptr->light_props.flicker_speed, 1.0f);
            f32_t intensity = std::lerp(shape_ptr->light_props.flicker_min, shape_ptr->light_props.flicker_max, fan::apply_ease((fan::ease_e)shape_ptr->light_props.ease_type, t));
            fan::color c = shape_ptr->get_color();
            c.a = intensity;
            shape_ptr->set_color(c);
          }
          if (shape_ptr->dynamic_props.target_color.r != 1.0f || shape_ptr->dynamic_props.target_color.g != 1.0f || shape_ptr->dynamic_props.target_color.b != 1.0f || shape_ptr->dynamic_props.target_color.a != 1.0f) {
            f32_t t = std::fmod(gloco()->time * shape_ptr->dynamic_props.variance_speed, 1.0f);
            f32_t factor = fan::apply_ease(static_cast<fan::ease_e>(shape_ptr->dynamic_props.ease_type), t);
            fan::color c;
            c.r = std::lerp(shape_ptr->dynamic_props.base_color.r, shape_ptr->dynamic_props.target_color.r, factor);
            c.g = std::lerp(shape_ptr->dynamic_props.base_color.g, shape_ptr->dynamic_props.target_color.g, factor);
            c.b = std::lerp(shape_ptr->dynamic_props.base_color.b, shape_ptr->dynamic_props.target_color.b, factor);
            c.a = std::lerp(shape_ptr->dynamic_props.base_color.a, shape_ptr->dynamic_props.target_color.a, factor);
            shape_ptr->set_color(c);
          }
        }
      }
    }

    void init_physics_scene() {
      saved_fgm_debug_cb = fan::physics::debug_draw_cb();
      fan::physics::debug_draw_cb() = [this](bool enabled, void*) {
        saved_fgm_debug_cb(enabled, &render_view);
      };
      fan::physics::gphysics()->debug.render_view = &render_view;
      for (auto& ptr : shape_list) {
        auto& p = ptr->physics;
        if (!p.enabled) continue;
        auto& child = ptr->children[0];
        fan::physics::shape_properties_t props;
        props.friction = p.friction;
        props.restitution = p.restitution;
        props.density = p.mass;
        props.is_sensor = p.is_sensor;
        auto angle = child.get_angle().z;
        auto pos = ptr->get_position();
        auto sz = ptr->get_size() * p.hitbox_size;
        if (p.collision_shape == 1) {
          segment_bodies.push_back(segment_body_t{
            fan::physics::gphysics()->create_segment(fan::vec2(pos), p.segment_points, (std::uint8_t)p.body_type, props)
          });
        }
        else switch (p.shape_type) {
          case 0: {
            physics_bodies.push_back(std::make_unique<fan::graphics::physics::rectangle_t>(
              fan::graphics::physics::rectangle_t::properties_t{
                .render_view = &render_view,
                .position = pos,
                .size = sz,
                .color = fan::colors::transparent,
                .angle = fan::vec3(0, 0, angle),
                .body_type = (std::uint8_t)p.body_type,
                .shape_properties = props
              }));
            break;
          }
          case 1: {
            f32_t radius = sz.max();
            physics_bodies.push_back(std::make_unique<fan::graphics::physics::circle_t>(
              fan::graphics::physics::circle_t::properties_t{
                .render_view = &render_view,
                .position = pos,
                .radius = radius,
                .color = fan::colors::transparent,
                .angle = fan::vec3(0, 0, angle),
                .body_type = (std::uint8_t)p.body_type,
                .shape_properties = props
              }));
            break;
          }
          case 2: {
            f32_t half_h = sz.y;
            f32_t cap_radius = sz.x;
            physics_bodies.push_back(std::make_unique<fan::graphics::physics::capsule_t>(
              fan::graphics::physics::capsule_t::properties_t{
                .render_view = &render_view,
                .position = pos,
                .center0 = fan::vec2(0, -half_h + cap_radius),
                .center1 = fan::vec2(0, half_h - cap_radius),
                .radius = cap_radius,
                .angle = fan::vec3(0, 0, angle),
                .color = fan::colors::transparent,
                .body_type = (std::uint8_t)p.body_type,
                .shape_properties = props
              }));
            break;
          }
        }
      }
      gloco()->update_physics(true);
    }

    struct segment_body_t {
      fan::physics::entity_t entity;
      fan::vec3 get_position() const { return fan::vec3(entity.get_position(), 0); }
      fan::vec3 get_angle() const { return fan::vec3(0); }
    };

    void destroy_physics_scene() {
      physics_bodies.clear();
      segment_bodies.clear();
      fan::physics::debug_draw_cb() = saved_fgm_debug_cb;
      gloco()->update_physics(false);
    }

    void step_physics() {
      if (physics_bodies.empty() && segment_bodies.empty()) return;
      std::size_t bi = 0, si = 0;
      for (auto& ptr : shape_list) {
        if (!ptr->physics.enabled) continue;
        if (ptr->physics.collision_shape == 1) {
          if (si >= segment_bodies.size()) break;
          auto& body = segment_bodies[si];
          ptr->set_position(fan::vec3(fan::vec2(body.get_position()), ptr->get_position().z));
          si++;
        } else {
          if (bi >= physics_bodies.size()) break;
          auto& body = physics_bodies[bi];
          ptr->set_position(fan::vec3(fan::vec2(body->get_position()), ptr->get_position().z));
          ptr->children[0].set_angle(body->get_angle());
          bi++;
        }
      }
    }

    void render() {
      f32_t zoom = fan::graphics::camera_get_zoom(render_view.camera);
      fan::vec2 mouse_world = viewport_t::get_mouse_position(viewport_settings, gui::get_mouse_pos(), zoom, fan::vec2(gui::get_style().WindowPadding));

      selection.update(*this, shape_list, mouse_world, zoom);

      if (gloco()->input.input_action.is_active("set_windowed_fullscreen")) gloco()->window.set_borderless();
      if (gloco()->input.input_action.is_active("toggle_content_browser")) render_content_browser = !render_content_browser;
      if (gloco()->input.input_action.is_active("save_file") && !is_playing) fout(previous_filename);

      if (render_content_browser) content_browser.render();

      step_physics();
      render_viewport(zoom);
      render_settings_window();
      properties_ui_t::render(*this, current_shape);
      properties_ui_t::render_materials(*this);
      render_shapes_window();
      render_texturepack_window();
      render_animations_window();
      render_animation_updates();
    }

    void fout(std::string filename) {
      scene_serializer_t::save(*this, filename);
    }

    void fin(const std::string& filename, const std::source_location& callers_path = std::source_location::current()) {
      scene_serializer_t::load(*this, filename);
    }

    void load_tp(shapes_t::global_t* node) {
      fan::graphics::texture_pack_t::ti_t ti;
      if (gloco()->texture_pack.qti(node->children[0].get_image_data().image_path, &ti)) {
        gui::print_warning("texturepack: non texturepack texture or failed to load texture:", node->children[0].get_image_data().image_path);
      } else {
        node->children[0].load_tp(&ti);
      }
    }

    void invalidate_current() {
      current_shape = nullptr;
    }

    void apply_material(shapes_t::global_t* node) {
      if (node->material_type == 1) {
        node->original_image = node->children[0].get_image();
        node->children[0].set_image(white_texture);
      } else {
        node->children[0].set_image(node->original_image);
      }
    }

    void erase_selected() {
      std::vector<std::size_t> indices;
      for (auto* obj : selection.objects) {
        for (std::size_t i = 0; i < shape_list.size(); ++i) {
          if (obj == shape_list[i].get()) {
            shape_original_json.erase(obj);
            indices.push_back(i);
            break;
          }
        }
      }
      std::sort(indices.begin(), indices.end(), std::greater<>());
      for (std::size_t i : indices) {
        if (is_playing && shape_list[i]->physics.enabled) {
          int bi = 0, si = 0;
          for (std::size_t k = 0; k < i; ++k) {
            if (!shape_list[k]->physics.enabled) continue;
            if (shape_list[k]->physics.collision_shape == 1) si++;
            else bi++;
          }
          if (shape_list[i]->physics.collision_shape == 1) {
            if (si < segment_bodies.size()) {
              segment_bodies[si].entity.destroy();
              segment_bodies.erase(segment_bodies.begin() + si);
            }
          } else {
            if (bi < physics_bodies.size()) {
              physics_bodies.erase(physics_bodies.begin() + bi);
            }
          }
        }
        shape_list.erase(shape_list.begin() + i);
      }
      selection.objects.clear();
      invalidate_current();
    }

    fan::graphics::shape_t axis_lines[2];
    fan::vec2 texturepack_size {};
    fan::vec2 texturepack_single_image_size {};
    std::vector<texturepack_image_t> texturepack_images;
    fan::graphics::render_view_t render_view;
    gui::content_browser_t content_browser {false};
    shapes_t::global_t* current_shape = nullptr;
    std::vector<std::unique_ptr<shapes_t::global_t>> shape_list;
    f32_t current_z = 1;
    std::uint32_t current_id = 0;
    bool render_content_browser = true;
    std::function<void()> close_cb = []() {};
    viewport_settings_t viewport_settings;
    selection_t<shapes_t::global_t> selection;
    bool shapes_window_hovered = false;
    gui::sprite_animations_t animations_application;
    fan::vec2 editor_pos = 0;
    fan::graphics::sprite_t background;
    std::vector<fan::graphics::shape_t> copy_buffer;
    std::string previous_filename;
    fan::graphics::image_t white_texture;
    fan::graphics::engine_t::keys_handle_t key_handle;
    fan::graphics::engine_t::mouse_move_handle_t mouse_move_handle;
    fan::graphics::engine_t::buttons_handle_t button_handle;
    
    std::unordered_map<shapes_t::global_t*, fan::json> shape_original_json;
    std::unordered_map<shapes_t::global_t*, shape_keyframe_animation_t> shape_sprite_sheets;

    static constexpr fan::color axis_x_color = (fan::colors::red / 2.f).set_alpha(0.8f);
    static constexpr fan::color axis_y_color = (fan::colors::green / 2.f).set_alpha(0.8f);
    bool render_axis_lines = true;
    bool is_playing = false;
    std::string scene_backup;
    std::vector<std::unique_ptr<fan::graphics::physics::base_shape_t>> physics_bodies;
    std::vector<segment_body_t> segment_bodies;
    int segment_drag_idx = -1;
    shapes_t::global_t* segment_drag_shape = nullptr;
    std::function<void(bool enabled, void* render_view)> saved_fgm_debug_cb;
  };
}
#endif