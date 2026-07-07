module;

#if defined(FAN_2D)
#endif

export module fan.graphics.gui.tilemap_editor.core:ui;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_2D)
#if defined(FAN_GUI) && defined(FAN_PHYSICS_2D)

import fan.physics.types;
import fan.texture_pack.tp0;
import fan.window.input;
import fan.types;
import fan.graphics.loco;
import fan.graphics.shapes;
import fan.graphics;
import fan.graphics.common_context;
import fan.graphics.gui.types;
import fan.graphics.gui.base;
import fan.graphics.gui;
import fan.graphics.gui.tilemap_editor.core;
import fan.graphics.gui.text_logger;
import fan.types.vector;
import fan.types.color;
import fan.math;
import fan.file_dialog;
import fan.print.error;
import fan.graphics.algorithm.raycast_grid;
import fan.math.intersection;
import fan.physics.b2_integration;
import fan.io.file;

namespace fan::graphics::gui::tilemap_editor::ui {

  template <typename enum_t, std::size_t N>
  bool combo_enum(const char* label, enum_t& val, const char* const (&names)[N]) {
    int idx = static_cast<int>(val);
    if (fan::graphics::gui::combo(label, &idx, N, [&](int i) -> const char* { return names[i]; })) {
      val = static_cast<enum_t>(idx);
      return true;
    }
    return false;
  }

  void draw_id_label(const std::string& id, const fan::vec3& world_pos, f32_t base_font_size, f32_t zoom, fan::graphics::render_view_t* render_view, fan::graphics::gui::draw_list_t* draw_list) {
    if (id.empty()) return;

    fan::vec2 screen_pos = fan::graphics::world_to_screen(fan::vec2(world_pos), render_view->viewport, render_view->camera);
    f32_t zoomed_font_size = base_font_size * zoom;
    f32_t fs_first = fan::graphics::gui::font_sizes[0];
    f32_t fs_last = fan::graphics::gui::font_sizes[std::size(fan::graphics::gui::font_sizes) - 1];

    zoomed_font_size = fan::math::clamp(zoomed_font_size, fs_first, fs_last);
    fan::graphics::gui::push_font(fan::graphics::gui::get_font(zoomed_font_size));

    std::string display_id;
    display_id.reserve(id.size() * 2);
    screen_pos = screen_pos.floor();
    for (char c : id) {
      display_id.push_back(c);
      if (c == '_') display_id.push_back('\n');
    }

    fan::vec2 text_size = fan::graphics::gui::text_size(display_id);
    screen_pos.x -= text_size.x * 0.5f;
    screen_pos.y -= text_size.y * 0.5f;
    screen_pos = screen_pos.floor();

    draw_list->AddText(screen_pos, fan::color(255, 255, 255, 200).get_gui_color(), display_id.c_str());
    fan::graphics::gui::pop_font();
  }

  void draw_id_labels(fte_t& editor) {
    auto* draw_list = fan::graphics::gui::get_foreground_draw_list();
    auto v = fan::graphics::ctx()->viewport_get(fan::graphics::ctx(), editor.render_view->viewport);
    fan::vec2 clip_min(v.position.x, v.position.y);
    fan::vec2 clip_max(v.position.x + v.size.x, v.position.y + v.size.y);
    draw_list->PushClipRect(clip_min, clip_max, true);

    f32_t base_font_size = 14.0f;
    f32_t zoom = fan::graphics::camera_get_zoom(editor.render_view->camera);

    for (auto& depth_pair : editor.spawn_marks) {
      for (auto& mark : depth_pair.second) draw_id_label(mark.id, mark.position, base_font_size, zoom, editor.render_view, draw_list);
    }
    for (auto& depth_pair : editor.physics_shapes) {
      for (auto& shape : depth_pair.second) draw_id_label(shape.id, shape.visual.get_position(), base_font_size, zoom, editor.render_view, draw_list);
    }
    draw_list->PopClipRect();
  }

  bool handle_editor_window(fte_t& editor, fan::vec2& editor_size) {
    if (fan::graphics::gui::begin_main_menu_bar()) {
      static std::string fn;
      if (fan::graphics::gui::begin_menu("File")) {
        if (fan::graphics::gui::menu_item("New")) {
          editor.reset_map();
        }
        if (fan::graphics::gui::menu_item("Open..")) {
          fan::graphics::open_file("fte,json", [&](std::string_view path) {
            fn = path;
            editor.fin(fn);
            fn.clear();
          });
        }
        if (fan::graphics::gui::menu_item("Save")) editor.fout(editor.previous_filename);
        if (fan::graphics::gui::menu_item("Save as")) {
          fan::graphics::save_file("fte,json", [&](std::string_view path) {
            fn = path;
            editor.fout(fn);
            fn.clear();
          });
        }
        if (fan::graphics::gui::menu_item("Quit")) fan::graphics::gui::end();
        fan::graphics::gui::end_menu();
      }

      if (fan::graphics::gui::begin_menu("Texture Pack")) {
        if (fan::graphics::gui::menu_item("Open..")) {
          fan::graphics::open_file("ftp", [&](std::string_view path) {
            fn = path;
            editor.open_texture_pack(fn);
            fn.clear();
          });
        }
        fan::graphics::gui::end_menu();
      }
      fan::graphics::gui::end_main_menu_bar();

      fan::graphics::gui::push_style_color(fan::graphics::gui::col_window_bg, fan::color(0, 0, 0, 0));
      fan::graphics::gui::begin("Tilemap Editor2", nullptr);
      fan::graphics::gui::pop_style_color();

      fan::vec2 viewport_size = fan::graphics::gui::get_content_region_avail();
      auto& style = fan::graphics::gui::get_style();
      fan::vec2 frame_padding = style.FramePadding;
      fan::vec2 real_viewport_size = viewport_size + frame_padding * 2 + fan::vec2(0, style.WindowPadding.y * 2);
      real_viewport_size.x = fan::math::clamp(real_viewport_size.x, 1.f, real_viewport_size.x);
      real_viewport_size.y = fan::math::clamp(real_viewport_size.y, 1.f, real_viewport_size.y);

      fan::graphics::camera_set_ortho(editor.render_view->camera, (fan::vec2(-real_viewport_size.x / 2, real_viewport_size.x / 2)), (fan::vec2(-real_viewport_size.y / 2, real_viewport_size.y / 2)));
      fan::graphics::gui::set_viewport(editor.render_view->viewport);
      editor_size = real_viewport_size;
      editor.viewport_settings.size = viewport_size;

      editor.viewport_settings.window_related_mouse_pos = fan::vec2(fan::vec2(fan::graphics::gui::get_window_pos()) + fan::vec2(fan::graphics::gui::get_window_size() / 2) + fan::vec2(0, style.WindowPadding.y * 2 - frame_padding.y * 2));

      fan::graphics::gui::set_font(fan::graphics::gui::get_font_size() * 1.5);
      fan::graphics::gui::text("brush type: "_str + editor.brush.type_names[(std::uint8_t)editor.brush.type]);
      fan::graphics::gui::text("brush depth: " + std::to_string((int)editor.brush.depth - fte_t::shape_depths_t::max_layer_depth / 2));
      fan::graphics::gui::pop_font();

      fan::vec2 prev_item_spacing = style.ItemSpacing;
      style.ItemSpacing = fan::vec2(0);
      fan::vec2 old_cursorpos = fan::graphics::gui::get_cursor_pos();
      fan::vec2 draw_start = fan::graphics::gui::get_mouse_pos();
      fan::vec2 cursor_pos = 0;
      cursor_pos.y = -(editor.tile_viewer_sprite_size.y / std::max(1.f, editor.current_tile_brush_count.y / 5.f));
      fan::graphics::gui::set_cursor_screen_pos(draw_start);

      for (auto& i : editor.current_tile_images) {
        int idx = 0;
        for (auto& j : i) {
          if (idx != 0) {
            cursor_pos.x += editor.tile_viewer_sprite_size.x / std::max(1.f, editor.current_tile_brush_count.x / 5.f);
            fan::graphics::gui::same_line();
          }
          else {
            cursor_pos.y += (editor.tile_viewer_sprite_size.y / std::max(1.f, editor.current_tile_brush_count.y / 5.f));
            cursor_pos.x = 0;
            fan::graphics::gui::set_cursor_screen_pos(fan::vec2(draw_start.x, fan::graphics::gui::get_cursor_screen_pos().y));
          }
          idx++;
          auto& img_data = fan::graphics::image_get_data(j.ti.image);
          fan::vec2 size = img_data.size;
          fan::graphics::gui::image_rotated(j.ti.image, (editor.tile_viewer_sprite_size / std::max(1.f, editor.current_tile_brush_count.x / 5.f)) * fan::graphics::camera_get_zoom(editor.render_view->camera), 360.f - fan::math::degrees(editor.brush.angle.z), j.ti.position / size, j.ti.position / size + j.ti.size / size, fan::color(1, 1, 1, 0.9));
        }
      }

      fan::graphics::gui::set_cursor_pos(old_cursorpos);
      style.ItemSpacing = prev_item_spacing;

      fan::vec2 cursor_position = fan::window::get_mouse_position();
      fan::vec2i grid_pos;
      if (editor.window_relative_to_grid(cursor_position, &grid_pos)) {
        std::string grid_str = (std::string)(grid_pos / (editor.tile_size * 2.f));
        std::string world_str = (std::string)grid_pos;
        
        std::string combined = "Grid: " + grid_str + "\nWorld: " + world_str;
        
        fan::graphics::gui::text(combined.c_str(), {
          .align = fan::graphics::gui::text_style_t::align_e::bottom_right
        });
      }

      if (fan::graphics::gui::begin("Layer window")) {
        for (auto& layer_pair : editor.visual_layers) {
          auto& layer = layer_pair.second; 
          layer.text.resize(32);
          std::uint16_t depth = layer_pair.first;
          auto fmt = ("Layer " + std::to_string(depth - fte_t::shape_depths_t::max_layer_depth / 2));

          if (fan::graphics::gui::toggle_button(("Visible " + fmt).c_str(), &layer.visible)) {
            auto iterate_positions = [&](auto l) {
              auto visual_found = editor.visual_layers.find(depth);
              if (visual_found == editor.visual_layers.end()) fan::throw_error("some weird bugs");
              for (auto& position_pair : visual_found->second.positions) {
                auto tiles_found = editor.map_tiles.find(position_pair.first);
                if (tiles_found == editor.map_tiles.end()) fan::throw_error("more some weird bugs");
                for (auto& tile_layer : tiles_found->second.layers) {
                  if (tile_layer.tile.position.z == depth) l(tile_layer);
                }
              }
            };

            if (layer.visible == false) {
              iterate_positions([&](fte_t::shapes_t::global_t::layer_t& l_layer) {
                auto vs_found = editor.visual_shapes.find(l_layer.tile.position);
                if (l_layer.tile.mesh_property != fte_t::mesh_property_t::none && vs_found != editor.visual_shapes.end()) {
                  vs_found->second.shape.set_size(0);
                }
                l_layer.shape.set_size(0);
              });
            }
            else {
              iterate_positions([&](fte_t::shapes_t::global_t::layer_t& l_layer) {
                auto vs_found = editor.visual_shapes.find(l_layer.tile.position);
                if (l_layer.tile.mesh_property != fte_t::mesh_property_t::none && vs_found != editor.visual_shapes.end()) {
                  vs_found->second.shape.set_size(l_layer.tile.size);
                }
                l_layer.shape.set_size(l_layer.tile.size);
              });
            }
            editor.modify_cb(0);
          }
          fan::graphics::gui::same_line();
          fan::graphics::gui::input_text(fmt.c_str(), &layer.text);
        }
      }
      fan::graphics::gui::end();
    }
    else {
      fan::graphics::viewport_zero(editor.render_view->viewport);
      return true;
    }

    editor.editor_settings.hovered = fan::graphics::gui::is_window_hovered();
    fan::graphics::gui::end();

    if (fan::window::is_key_clicked(fan::key_s) && fan::window::is_key_down(fan::key_left_control)) {
      editor.fout(editor.previous_filename);
    }

    if (fan::graphics::get_window().key_state(fan::key_shift) != -1) {
      fan::vec2 line_dst = fan::graphics::get_mouse_position(editor.render_view->camera, editor.render_view->viewport);
      bool control_pressed = fan::graphics::get_window().key_pressed(fan::key_left_control);

      if (control_pressed) line_dst = fan::math::snap_line_to_angle(editor.brush.line_src, line_dst, 45.0f);
      editor.visual_line.set_line(editor.brush.line_src, line_dst);

      if (fan::graphics::get_window().key_state(fan::mouse_left) == 1) {
        editor.brush.line_src = ((editor.brush.line_src + editor.tile_size) / (editor.tile_size * 2)).floor() * editor.tile_size * 2;
        fan::vec2 final_dst = line_dst;
        final_dst = ((final_dst + editor.tile_size) / (editor.tile_size * 2)).floor() * editor.tile_size * 2;

        if (final_dst.x - editor.brush.line_src.x > editor.tile_size.x * 2) final_dst.x += editor.tile_size.x * 2;
        if (final_dst.y - editor.brush.line_src.y > editor.tile_size.y * 2) final_dst.y += editor.tile_size.y * 2;

        fan::vec2 raycast_dst = control_pressed ? line_dst : final_dst;
        f32_t divider = 2.0001;

        if (editor.brush.line_src.y > final_dst.y && editor.brush.line_src.x < final_dst.x) divider = 1.9999;
        else if (editor.brush.line_src.y < final_dst.y && editor.brush.line_src.x > final_dst.x) divider = 1.9999;

        std::vector<fan::vec2i> raycast_positions = fan::graphics::algorithm::grid_raycast({editor.brush.line_src / 2 + editor.tile_size / divider, raycast_dst / 2 + editor.tile_size / divider}, editor.tile_size);

        for (fan::vec2i& pos : raycast_positions) {
          fan::vec2i p = pos * (editor.tile_size * 2);
          for (int i = 0; i < editor.brush.size.y; ++i) {
            for (int j = 0; j < editor.brush.size.x; ++j) {
              fan::vec2i tile_pos = p + fan::vec2i(j * editor.tile_size.x, i * editor.tile_size.y);
              editor.handle_tile_push(tile_pos, i, j);
            }
          }
        }
      }
    }
    else {
      editor.visual_line.set_line(-999999999, -999999999);
    }
    return false;
  }

  bool handle_editor_settings_window(fte_t& editor) {
    if (fan::graphics::gui::begin("Editor settings")) {
      if (fan::graphics::gui::input_int("map size", &editor.map_size)) editor.resize_map();
      if (fan::graphics::gui::input_int("tile size", &editor.tile_size)) {
        editor.resize_map();
        for (auto& i : editor.map_tiles) {
          for (auto& j : i.second.layers) {
            fan::vec2 s = j.shape.get_size();
            fan::vec2 sp = fan::vec2(j.shape.get_position());
            fan::vec2 p = editor.tile_size * ((sp / s));
            j.shape.set_position(p);
            j.shape.set_size(editor.tile_size);
          }
        }
      }
      if (fan::graphics::gui::checkbox("render grid", &editor.grid_visualize.render_grid)) {
        if (editor.grid_visualize.render_grid) editor.grid_visualize.grid.set_size(editor.map_size * (editor.tile_size * 2.f) / 2.f);
        else editor.grid_visualize.grid.set_size(0);
      }
    }
    fan::graphics::gui::end();
    return false;
  }

  void handle_tiles_window(fte_t& editor) {
    static f32_t zoom = 1;
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_button, fan::color(0.f, 0.f, 0.f, 0.f));
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_button_active, fan::color(0.f, 0.f, 0.f, 0.f));
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_button_hovered, fan::color(0.f, 0.f, 0.f, 0.f));
    fan::graphics::gui::push_style_var(fan::graphics::gui::style_var_frame_padding, fan::vec2(0, 0));
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_button, fan::color::rgb(31, 31, 31));
    fan::graphics::gui::push_style_color(fan::graphics::gui::col_window_bg, fan::color::rgb(31, 31, 31));

    if (fan::graphics::gui::begin("tiles", nullptr)) {
      
      fan::graphics::gui::dummy(fan::vec2(0, 24));

      auto build_and_save = [&editor](const std::string& target_path, bool load_existing, const std::vector<std::string>& paths) {
        fan::graphics::texture_pack::internal_t builder;
        builder.open();
        
        if (load_existing && !editor.texture_packs.empty()) {
          fan::graphics::texture_pack_t* tp = editor.texture_packs[0];
          tp->iterate_loaded_images([&](const fan::graphics::texture_pack_t::texture_minor_decoded_t& minor) {
            fan::graphics::texture_pack::internal_t::texture_properties_t props;
            props.image_name = minor.name;
            fan::vec2 img_size = tp->get_pixel_data(minor.unique_id).image.get_size();
            props.uv_pos = fan::vec2(minor.position) / img_size;
            props.uv_size = fan::vec2(minor.size) / img_size;
            builder.push_texture(tp->get_pixel_data(minor.unique_id).image, props);
          });
        }

        for (const auto& path : paths) {
          fan::graphics::texture_pack::internal_t::texture_properties_t props;
          props.image_name = path;
          builder.push_texture(path, props);
        }

        builder.process();
        builder.save_compiled(target_path);
        editor.open_texture_pack(target_path);
      };
      
      auto handle_add_images = [&editor, build_and_save](const std::vector<std::string>& paths) {
        if (editor.texture_packs.empty() || editor.texture_packs[0]->file_path.empty() || !fan::io::file::exists(editor.texture_packs[0]->file_path)) {
          fan::graphics::gui::print_error("No texture pack open. Select save location for new pack...");
          fan::graphics::save_file("ftp", [build_and_save, paths](std::string_view save_path) {
            auto fn = std::string(save_path);
            fan::io::file::ensure_extension(fn, ".ftp");
            build_and_save(std::string(save_path), false, paths);
          });
        } 
        else {
          build_and_save(editor.texture_packs[0]->file_path, true, paths);
        }
      };

      auto handle_content_browser_items = [&editor, handle_add_images](const std::vector<std::string>& payloads) {
        std::vector<std::string> paths;
        paths.reserve(payloads.size());
        for (const auto& payload : payloads) {
          std::filesystem::path p(payload);
          paths.push_back(p.is_absolute() ? payload : (std::filesystem::path(editor.content_browser.asset_path) / p).generic_string());
        }

        if (!paths.empty()) {
          handle_add_images(paths);
        }
      };

      fan::vec2 drop_min = fan::graphics::gui::get_window_pos();
      fan::vec2 drop_max = drop_min + fan::graphics::gui::get_window_size();
      fan::graphics::gui::item_add(fan::graphics::gui::rect_t(drop_min, drop_max), fan::graphics::gui::get_id("##tiles_drop_target"));
      fan::graphics::gui::receive_drag_drop_target("CONTENT_BROWSER_ITEMS", handle_content_browser_items);

      if (fan::graphics::gui::button("Add Image(s)")) {
        fan::graphics::open_files("webp,png,jpg", [handle_add_images](std::vector<std::string_view> paths) {
          std::vector<std::string> safe_paths;
          safe_paths.reserve(paths.size());
          for (auto p : paths) safe_paths.emplace_back(p);
          handle_add_images(safe_paths);
        });
      }

      fan::graphics::gui::receive_drag_drop_target("CONTENT_BROWSER_ITEMS", handle_content_browser_items);
      fan::graphics::gui::dummy(fan::vec2(0, 10));
      // ---------------------------------------------------------

      fan::graphics::gui::separator();

      if (fan::graphics::gui::is_window_hovered() && fan::window::is_key_down(fan::key_left_control)) {
        zoom *= 1.f + fan::graphics::gui::get_io().MouseWheel * 0.08f;
        zoom = fan::math::clamp(zoom, 0.01f, 8.f);
      }
      if (fan::graphics::gui::is_window_hovered() && fan::graphics::gui::is_mouse_dragging(fan::mouse_middle)) {
        fan::vec2 mouse_delta = fan::graphics::gui::get_mouse_drag_delta(fan::mouse_middle);
        fan::graphics::gui::reset_mouse_drag_delta(fan::mouse_middle);
        fan::graphics::gui::set_scroll_x(fan::graphics::gui::get_scroll_x() - mouse_delta.x);
        fan::graphics::gui::set_scroll_y(fan::graphics::gui::get_scroll_y() - mouse_delta.y);
      }

      f32_t x_size = fan::graphics::gui::get_content_region_avail().x;
      fan::graphics::gui::drag("original image width", &editor.original_image_width, 1, 0, 10000);

      auto& style = fan::graphics::gui::get_style();
      fan::vec2 prev_item_spacing = style.ItemSpacing;
      style.ItemSpacing = fan::vec2(0);
      editor.current_tile_brush_count = 0;

      int total_images = editor.texture_pack_images.size();
      int images_per_row = editor.texturepack_single_image_size.x <= 0 ? 0 : (editor.original_image_width / editor.texturepack_single_image_size.x);

      if (images_per_row == 0) {
        style.ItemSpacing = prev_item_spacing;
        fan::graphics::gui::end();
        fan::graphics::gui::pop_style_color(5);
        fan::graphics::gui::pop_style_var();
        return;
      }

      int rows_needed = std::max((total_images + images_per_row - 1) / images_per_row, 1);
      f32_t final_image_size = x_size / images_per_row;
      f32_t cell_size = std::max(final_image_size * zoom, 1.f);

      static fan::vec2 selection_end(-1, -1);
      static fan::vec2 min_rect = (std::uint32_t)~0;
      static fan::vec2 max_rect = -1;
      static fan::vec2 min_rect_draw = (std::uint32_t)~0;
      static fan::vec2 max_rect_draw = -1;
      static bool is_selecting = false;

      bool is_left_mouse_button_clicked = fan::window::is_mouse_clicked(0);
      bool is_left_mouse_drag = fan::window::is_mouse_down(0) && fan::graphics::gui::is_mouse_dragging(0);
      bool is_right_mouse_button_clicked = fan::window::is_mouse_clicked(1);
      bool is_right_mouse_drag = fan::window::is_mouse_down(1) && fan::graphics::gui::is_mouse_dragging(1);
      bool is_left_ctrl_key_pressed = fan::window::is_key_down(fan::key_left_control);
      bool is_left_mouse_button_released = fan::window::is_mouse_released(0);

      fan::vec2 sprite_size = fan::vec2(cell_size);
      fan::vec2 initial_pos = fan::graphics::gui::get_cursor_screen_pos();
      auto* draw_list = fan::graphics::gui::get_window_draw_list();

      for (std::uint32_t i = 0; i < editor.texture_pack_images.size(); i++) {
        auto& node = editor.texture_pack_images[i];
        fan::vec2i grid_index(i % images_per_row, i / images_per_row);
        fan::vec2 cursor_pos_global = initial_pos + fan::vec2(grid_index) * cell_size;
        auto& img_data = fan::graphics::image_get_data(node.ti.image);
        fan::vec2 size = img_data.size;
        fan::vec2 node_position = fan::vec2(node.ti.position);
        fan::vec2 node_size = fan::vec2(node.ti.size);
        fan::vec2 uv0 = 0;
        fan::vec2 uv1 = 1;
        bool image_is_atlas = (size.x > node_size.x || size.y > node_size.y) &&
          (node_position + node_size).x <= size.x &&
          (node_position + node_size).y <= size.y;
        if (image_is_atlas) {
          uv0 = node_position / size;
          uv1 = (node_position + node_size) / size;
        }

        fan::graphics::gui::set_cursor_screen_pos(cursor_pos_global);
        fan::graphics::gui::invisible_button((std::string("##ibutton") + std::to_string(i)).c_str(), sprite_size);

        fan::vec2 image_size = sprite_size;
        f32_t max_node_size = std::max(node_size.x, node_size.y);
        if (max_node_size > 0) {
          image_size = node_size / max_node_size * cell_size;
        }
        fan::vec2 image_pos = cursor_pos_global + (sprite_size - image_size) / 2.f;
        draw_list->AddImage((fan::graphics::gui::texture_id_t)fan::graphics::image_get_handle(node.ti.image), image_pos, image_pos + image_size, uv0, uv1);

        if (editor.current_image_indices.find(grid_index) != editor.current_image_indices.end()) {
          draw_list->AddRect(cursor_pos_global, cursor_pos_global + sprite_size, 0xff0077ff, 0, 0, 1);
        }

        if (!is_selecting && fan::math::d2::aabb_point_inside(cursor_pos_global, fan::graphics::gui::get_mouse_pos() - sprite_size / 2, sprite_size / 2)) {
          draw_list->AddRect(cursor_pos_global, cursor_pos_global + sprite_size, 0xff0077ff, 0, 0, 3);
        }

        bool is_mouse_hovered = fan::graphics::gui::is_item_hovered(fan::graphics::gui::hovered_flags_rect_only);

        if (is_mouse_hovered && is_left_mouse_drag) {
          min_rect_draw = min_rect_draw.min(cursor_pos_global);
          max_rect_draw = max_rect_draw.max(cursor_pos_global);
          min_rect = min_rect.min(fan::vec2(grid_index));
          max_rect = max_rect.max(fan::vec2(grid_index));
        }

        if (is_mouse_hovered && is_left_mouse_button_clicked && !is_left_mouse_drag && !is_left_ctrl_key_pressed) {
          editor.current_image_indices.clear();
          editor.current_tile_images.clear();
          editor.current_image_indices[grid_index] = i;
        }
        else if (is_mouse_hovered && is_left_mouse_button_clicked && is_left_ctrl_key_pressed) {
          auto found = editor.current_image_indices.find(grid_index);
          if (found != editor.current_image_indices.end()) editor.current_image_indices.erase(found);
          else editor.current_image_indices[grid_index] = i;
        }
        else if (is_mouse_hovered && is_left_mouse_drag) is_selecting = true;
        else if ((is_right_mouse_button_clicked || is_right_mouse_drag) && is_mouse_hovered) {
          auto found = editor.current_image_indices.find(grid_index);
          if (found != editor.current_image_indices.end()) editor.current_image_indices.erase(found);
          if (editor.current_image_indices.empty()) editor.current_tile_images.clear();
        }
      }

      fan::vec2 cursor_grid = fan::graphics::gui::get_mouse_pos() - initial_pos;
      cursor_grid /= sprite_size;
      cursor_grid = cursor_grid.floor();

      if (is_selecting) {
        selection_end = fan::graphics::gui::get_mouse_pos();
        fan::vec2 max_rect_draw_adjusted = max_rect_draw + sprite_size;
        max_rect_draw_adjusted = max_rect_draw_adjusted.min(initial_pos + cursor_grid * sprite_size + sprite_size);
        if (min_rect != (std::uint32_t)~0 && max_rect != -1) draw_list->AddRect(min_rect_draw, max_rect_draw_adjusted, 0xff0077ff);

        if (is_left_mouse_button_released) {
          is_selecting = false;
          min_rect = (std::uint32_t)~0; max_rect = -1;
          min_rect_draw = (std::uint32_t)~0; max_rect_draw = -1;
        }
      }

      if (min_rect != (std::uint32_t)~0 && max_rect != -1) {
        for (int y = min_rect.y; y <= std::min(max_rect.y, cursor_grid.y); ++y) {
          for (int x = min_rect.x; x <= std::min(max_rect.x, cursor_grid.x); ++x) {
            int idx = y * images_per_row + x;
            if (idx >= 0 && idx < total_images) {
              editor.current_image_indices[fan::vec2i(x, y)] = idx;
            }
          }
        }
      }

      fan::graphics::gui::set_cursor_screen_pos(initial_pos);
      fan::graphics::gui::dummy(fan::vec2(images_per_row * cell_size, rows_needed * cell_size));

      style.ItemSpacing = prev_item_spacing;
      if (editor.current_image_indices.size()) editor.current_tile_images.clear();

      int prev_y = -1;
      int y = -1;
      int x = 0;
      for (auto& i : editor.current_image_indices) {
        if (prev_y != i.first.y) {
          editor.current_tile_images.resize(editor.current_tile_images.size() + 1);
          prev_y = i.first.y;
          editor.current_tile_brush_count.x = std::max(editor.current_tile_brush_count.x, x);
          x = 0;
          y++;
        }
        editor.current_tile_images[y].push_back(editor.texture_pack_images[i.second]);
        x++;
      }

      if (fan::graphics::gui::is_window_hovered() && fan::window::is_key_clicked(fan::key_r)) {
        if (!editor.current_image_indices.empty()) {
          fan::graphics::gui::open_popup("Delete Selected Tiles?");
        }
      }

      if (fan::graphics::gui::begin_popup_modal("Delete Selected Tiles?", fan::graphics::gui::window_flags_always_auto_resize)) {
        fan::graphics::gui::text("Permanently delete the selected tiles from the texture pack?");
        fan::graphics::gui::separator();

        if (fan::graphics::gui::button("Yes", fan::vec2(120, 0))) {
          std::vector<std::string> to_delete;
          for (const auto& [grid, idx] : editor.current_image_indices) {
            auto minor = (*editor.texture_packs[0])[editor.texture_pack_images[idx].ti.unique_id];
            to_delete.push_back(minor.name);
          }

          fan::graphics::texture_pack::internal_t builder;
          builder.open();
          
          fan::graphics::texture_pack_t tp(editor.texture_packs[0]->file_path);
          tp.iterate_loaded_images([&](const fan::graphics::texture_pack_t::texture_minor_decoded_t& minor) {
            if (std::find(to_delete.begin(), to_delete.end(), minor.name) == to_delete.end()) {
              fan::graphics::texture_pack::internal_t::texture_properties_t props;
              props.image_name = minor.name;
              fan::vec2 img_size = tp.get_pixel_data(minor.unique_id).image.get_size();
              props.uv_pos = fan::vec2(minor.position) / img_size;
              props.uv_size = fan::vec2(minor.size) / img_size;
              builder.push_texture(tp.get_pixel_data(minor.unique_id).image, props);
            }
          });
          builder.process();
          builder.save_compiled(editor.texture_packs[0]->file_path);
          editor.open_texture_pack(editor.texture_packs[0]->file_path);

          editor.current_image_indices.clear();
          editor.current_tile_images.clear();

          fan::graphics::gui::close_current_popup();
        }
        
        fan::graphics::gui::same_line();
        if (fan::graphics::gui::button("No", fan::vec2(120, 0))) {
          fan::graphics::gui::close_current_popup();
        }
        
        fan::graphics::gui::end_popup();
      }

      editor.current_tile_brush_count.x = std::max(editor.current_tile_brush_count.x, x);
      editor.current_tile_brush_count.y = y;
    }
    fan::graphics::gui::end();
    fan::graphics::gui::pop_style_color(5);
    fan::graphics::gui::pop_style_var();
  }

  void handle_tile_settings_window(fte_t& editor) {
    if (fan::graphics::gui::begin("Tile settings", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      if (editor.current_tile.layer != nullptr) {
        auto& layer = editor.current_tile.layer[editor.current_tile.layer_index];

        fan::vec2 offset = fan::vec2(layer.shape.get_position()) - editor.current_tile.position;
        if (fan::graphics::gui::drag("offset", &offset, 0.1, 0, 0)) layer.shape.set_position(fan::vec2(editor.current_tile.position) + offset);

        fan::vec2 tile_size = layer.shape.get_size();
        if (fan::graphics::gui::drag("tile size", &tile_size)) layer.shape.set_size(tile_size);

        std::string temp = layer.tile.id;
        temp.resize(fte_t::max_id_len);
        if (fan::graphics::gui::input_text("id", &temp)) layer.tile.id = temp.substr(0, std::strlen(temp.c_str()));

        fan::vec3 angle = layer.shape.get_angle();
        if (fan::graphics::gui::drag("angle", &angle, fan::math::radians(1))) layer.shape.set_angle(angle);

        fan::vec2 rotation_point = layer.shape.get_rotation_point();
        if (fan::graphics::gui::drag("rotation_point", &rotation_point, 0.1, -tile_size.max() * 2, tile_size.max() * 2)) layer.shape.set_rotation_point(rotation_point);

        std::uint32_t flags = layer.shape.get_flags();
        if (fan::graphics::gui::input_int("special flags", (int*)&flags, 1, 1)) layer.shape.set_flags(flags);

        fan::color color = layer.shape.get_color();
        if (fan::graphics::gui::color_edit4("color", &color)) layer.shape.set_color(color);

        int mesh_property = (int)layer.tile.mesh_property;
        if (fan::graphics::gui::slider("mesh flags", &mesh_property, 0, (int)fte_t::mesh_property_t::size - 1)) layer.tile.mesh_property = (fte_t::mesh_property_t)mesh_property;
      }
    }
    fan::graphics::gui::end();
  }

  void handle_brush_settings_window(fte_t& editor) {
    static bool set_default_focus = true;
    if (set_default_focus) fan::graphics::gui::set_next_window_focus();
    
    if (fan::graphics::gui::begin("Brush settings", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      set_default_focus = false;
      int idx = (int)editor.brush.depth - fte_t::shape_depths_t::max_layer_depth / 2;
      if (fan::graphics::gui::drag("depth", (int*)&idx, 1, 0, fte_t::shape_depths_t::max_layer_depth)) {
        editor.brush.depth = idx + fte_t::shape_depths_t::max_layer_depth / 2;
      }

      combo_enum("mode", editor.brush.mode, fte_t::brush_t::mode_names);
      combo_enum("type", editor.brush.type, fte_t::brush_t::type_names);

      if (fan::graphics::gui::slider("jitter", &editor.brush.jitter, 0, editor.brush.size.min())) {
        editor.grid_visualize.highlight_hover.set_size(editor.tile_size * editor.brush.size);
      }
      fan::graphics::gui::drag("jitter_chance", &editor.brush.jitter_chance, 1, 0, 0.01);

      combo_enum("dynamics angle", editor.brush.dynamics_angle, fte_t::brush_t::dynamics_names);
      combo_enum("dynamics color", editor.brush.dynamics_color, fte_t::brush_t::dynamics_names);

      if (fan::graphics::gui::drag("size", &editor.brush.size, 0.1)) editor.grid_visualize.highlight_hover.set_size(editor.tile_size * editor.brush.size);

      fan::graphics::gui::drag("tile size", &editor.brush.tile_size, 0.1);
      fan::graphics::gui::drag("angle", &editor.brush.angle, 0.1);

      std::string temp = editor.brush.id;
      temp.resize(fte_t::max_id_len);
      if (fan::graphics::gui::input_text("id", &temp)) editor.brush.id = temp.substr(0, std::strlen(temp.c_str()));

      fan::graphics::gui::color_edit4("color", &editor.brush.color);

      switch (editor.brush.type) {
        case fte_t::brush_t::type_e::light: {
          fan::graphics::gui::drag("flags", &editor.brush.flags, 0.1);
          break;
        }
        case fte_t::brush_t::type_e::physics_shape: {
          fan::graphics::gui::drag("offset", &editor.brush.offset, 0.1);

          combo_enum("Physics shape type", editor.brush.physics_type, fte_t::brush_t::physics_type_names);
          combo_enum("Physics body type", editor.brush.physics_body_type, fte_t::brush_t::physics_body_type_names);

          static bool default_bool = 0;
          if (fan::graphics::gui::toggle_button("Physics shape draw", &default_bool)) editor.brush.physics_draw = default_bool;

          static fan::physics::shape_properties_t sp;
          if (fan::graphics::gui::drag("Physics shape friction", &sp.friction, 0.01, 0, 1)) editor.brush.physics_shape_properties.friction = sp.friction;
          if (fan::graphics::gui::drag("Physics shape density", &sp.density, 0.01, 0, 1)) editor.brush.physics_shape_properties.density = sp.density;
          if (fan::graphics::gui::toggle_button("Physics shape fixed rotation", &sp.fixed_rotation)) editor.brush.physics_shape_properties.fixed_rotation = sp.fixed_rotation;
          if (fan::graphics::gui::toggle_button("Physics shape enable presolve events", &sp.presolve_events)) editor.brush.physics_shape_properties.presolve_events = sp.presolve_events;
          if (fan::graphics::gui::toggle_button("Is sensor", &sp.is_sensor)) editor.brush.physics_shape_properties.is_sensor = sp.is_sensor;
          break;
        }
        default: break;
      }
    }
    fan::graphics::gui::end();
  }

  void handle_lighting_settings_window(fte_t& editor) {
    if (fan::graphics::gui::begin("lighting settings", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      static fan::vec3 ambient = fan::graphics::get_lighting().ambient;
      if (fan::graphics::gui::color_edit3("ambient", &ambient)) fan::graphics::get_lighting().set_target(ambient);
    }
    fan::graphics::gui::end();
  }

  void handle_physics_settings_window(fte_t& editor) {
    if (fan::graphics::gui::begin("physics settings", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      fan::vec2 gravity = fan::physics::gphysics()->get_gravity();
      if (fan::graphics::gui::drag("gravity", &gravity, 0.01)) fan::physics::gphysics()->set_gravity(gravity);
    }
    fan::graphics::gui::end();
  }

  void handle_custom_tools_window(fte_t& editor) {
    if (fan::graphics::gui::begin("Custom Tools", nullptr, fan::graphics::gui::window_flags_no_focus_on_appearing)) {
      fan::graphics::gui::text("Map Stats:");
      fan::graphics::gui::text("Grid Size: " + std::to_string(editor.map_size.x) + "x" + std::to_string(editor.map_size.y));
      std::size_t total_tiles = 0;
      for (const auto& [pos, cell] : editor.map_tiles) total_tiles += cell.layers.size();
      fan::graphics::gui::text("Total Tiles: " + std::to_string(total_tiles));
      fan::graphics::gui::text("Active Layers: " + std::to_string(editor.visual_layers.size()));
      fan::graphics::gui::separator();
      if (fan::graphics::gui::button("Wipe Map Clean")) editor.reset_map();
    }
    fan::graphics::gui::end();
  }

  void handle_tile_brush(fte_t& editor) {
    if (!editor.editor_settings.hovered) return;

    static std::vector<fan::graphics::shape_t> select;
    static fan::vec2i copy_src;
    static fan::vec2i copy_dst;

    bool is_mouse_left_clicked = fan::window::is_mouse_clicked();
    bool is_mouse_left_down = fan::window::is_mouse_down();
    bool is_mouse_left_released = fan::window::is_mouse_released();
    bool is_mouse_right_clicked = fan::window::is_mouse_clicked(fan::mouse_right);
    bool is_mouse_right_down = fan::window::is_mouse_down(fan::mouse_right);

    auto handle_select_tiles = [&] {
      select.clear();
      fan::vec2i mouse_grid_pos;
      if (editor.mouse_to_grid(mouse_grid_pos)) {
        fan::vec2i src = copy_src;
        fan::vec2i dst = mouse_grid_pos;
        copy_dst = dst;
        fte_t::for_each_rect(src, dst, [&](int i, int j) {
          select.push_back(fan::graphics::unlit_sprite_t{ {
            .render_view = editor.render_view,
            .position = fan::vec3(fan::vec2(i, j) * editor.tile_size * 2, fte_t::shape_depths_t::cursor_highlight_depth),
            .size = editor.tile_size,
            .image = editor.grid_visualize.highlight_color,
            .blending = true
          } });
        });
      }
    };

    switch (editor.brush.mode) {
      case fte_t::brush_t::mode_e::draw: {
        switch (editor.brush.type) {
          case fte_t::brush_t::type_e::physics_shape: {
            if (is_mouse_left_clicked) {
              select.clear();
              fan::vec2i mouse_grid_pos;
              if (editor.mouse_to_grid(mouse_grid_pos)) copy_src = mouse_grid_pos;
            }
            if (is_mouse_left_down) handle_select_tiles();
            if (is_mouse_left_released) {
              select.clear();
              fan::vec2i mouse_grid_pos;
              if (editor.mouse_to_grid(mouse_grid_pos)) {
                fan::vec3 new_pos = fan::vec3(((fan::vec2(copy_src) + (fan::vec2(copy_dst) - fan::vec2(copy_src)) / 2) + editor.brush.offset / 2) * editor.tile_size * 2, editor.brush.depth);
                fan::vec2 new_size = (((copy_dst - copy_src) + fan::vec2(1, 1)) * fan::vec2(editor.tile_size)).abs() * editor.brush.tile_size;
                if (!editor.physics_shape_exists(new_pos, new_size)) {
                  auto& layers = editor.physics_shapes[editor.brush.depth];
                  layers.push_back({
                    .visual = fan::graphics::sprite_t{{
                      .render_view = editor.render_view,
                      .position = new_pos,
                      .size = new_size,
                      .image = editor.grid_visualize.collider_color,
                      .blending = true
                    }},
                    .type = editor.brush.physics_type,
                    .body_type = editor.brush.physics_body_type,
                    .draw = editor.brush.physics_draw,
                    .shape_properties = editor.brush.physics_shape_properties,
                    .id = editor.brush.id
                  });
                }
              }
            }
            if (!(is_mouse_right_clicked || is_mouse_right_down)) return;
            break;
          }
          default: break;
        }

        fan::vec2i position;
        bool is_ctrl_pressed = fan::graphics::get_window().key_pressed(fan::key_left_control);
        bool is_shift_pressed = fan::graphics::get_window().key_pressed(fan::key_left_shift);

        if (is_mouse_left_down && !is_ctrl_pressed && !is_shift_pressed && !fan::window::is_key_down(fan::key_t) && !fan::window::is_key_down(fan::key_5)) {
          editor.handle_tile_action(position, [&](auto...args) { return editor.handle_tile_push(args...); });
        }
        if (is_mouse_right_down) editor.handle_tile_action(position, [&](auto...args) { return editor.handle_tile_erase(args...); });
        break;
      }
      case fte_t::brush_t::mode_e::copy: {
        if (is_mouse_left_clicked) {
          select.clear();
          editor.copy_buffer.clear();
          fan::vec2i mouse_grid_pos;
          if (editor.mouse_to_grid(mouse_grid_pos)) copy_src = mouse_grid_pos;
        }
        if (is_mouse_left_down) handle_select_tiles();
        if (is_mouse_left_released) {
          select.clear();
          fan::vec2i mouse_grid_pos;
          if (editor.mouse_to_grid(mouse_grid_pos)) {
            fan::vec2 src = copy_src;
            fan::vec2 dst = copy_dst;
            int stepx = (src.x <= dst.x) ? 1 : -1;
            int stepy = (src.y <= dst.y) ? 1 : -1;
            editor.copy_buffer_region.x = std::max(1.f, std::abs((dst.x + stepx) - src.x));
            editor.copy_buffer_region.y = std::max(1.f, std::abs((dst.y + stepy) - src.y));

            if (src == dst) {
              auto found = editor.map_tiles.find(copy_src);
              if (found != editor.map_tiles.end()) {
                editor.copy_buffer.push_back(found->second);
                for (auto& i : editor.copy_buffer.back().layers) i.shape.set_size(0);
              }
            }
            else {
              fte_t::for_each_rect(src, dst, [&](int i, int j) {
                auto found = editor.map_tiles.find(fan::vec2i(i, j));
                if (found != editor.map_tiles.end()) {
                  editor.copy_buffer.push_back(found->second);
                  for (auto& layer : editor.copy_buffer.back().layers) layer.shape.set_size(0);
                }
                else editor.copy_buffer.push_back({});
              });
            }
          }
        }

        if (is_mouse_right_clicked) {
          fan::vec2i mouse_grid_pos;
          if (editor.mouse_to_grid(mouse_grid_pos)) {
            int index = 0;
            for (auto& i : editor.copy_buffer) {
              fan::vec2i current_pos = mouse_grid_pos + fan::vec2i(index % editor.copy_buffer_region.x, index / editor.copy_buffer_region.x);
              if (editor.is_in_constraints(current_pos * editor.tile_size * 2)) {
                auto& tile = editor.map_tiles[current_pos];
                tile = i;
                for (std::size_t k = 0; k < tile.layers.size(); ++k) {
                  auto& t = tile.layers[k];
                  fan::vec2 op = t.shape.get_position();
                  fan::vec2 offset = op - fan::vec2(t.tile.position) * editor.tile_size * 2 - editor.tile_size;
                  fan::vec2 draw_pos = current_pos * editor.tile_size * 2 + editor.tile_size + offset;

                  if (tile.layers[k].tile.position.z != editor.brush.depth) {
                    t.shape.set_position(fan::vec3(fan::vec2(draw_pos), t.tile.position.z));
                    t.shape.set_size(t.tile.size);
                    continue;
                  }
                  if (editor.is_in_constraints(draw_pos)) {
                    editor.visual_layers[t.tile.position.z].positions[current_pos] = 1;
                    t.shape.set_position(fan::vec3(fan::vec2(draw_pos), t.tile.position.z));
                    t.shape.set_size(t.tile.size);
                    if (t.tile.mesh_property == fte_t::mesh_property_t::light) {
                      editor.visual_shapes[fan::vec3(draw_pos, editor.brush.depth)].shape = fan::graphics::sprite_t{{
                        .render_view = editor.render_view,
                        .position = fan::vec3(draw_pos, editor.brush.depth + 1),
                        .size = tile.layers[k].tile.size,
                        .image = editor.grid_visualize.light_color,
                        .blending = true
                      }};
                    }
                  }
                }
              }
              index++;
            }
          }
        }
        break;
      }
    }
  }

  void handle_gui(fte_t& editor) {
    fan::vec2 editor_size;

    if (handle_editor_window(editor, editor_size)) {
      fan::graphics::gui::end();
      return;
    }
    if (handle_editor_settings_window(editor)) return;

    handle_tiles_window(editor);
    {
      handle_tile_settings_window(editor);
      handle_brush_settings_window(editor);
      handle_lighting_settings_window(editor);
      handle_physics_settings_window(editor);
      handle_custom_tools_window(editor);

      if (editor.editor_settings.hovered && fan::window::is_mouse_down()) {
        if (fan::graphics::get_window().key_pressed(fan::key_t)) editor.handle_pick_tile();
        else if (fan::graphics::get_window().key_pressed(fan::key_left_alt)) editor.handle_select_tile();
      }

      handle_tile_brush(editor);
    }

    editor.terrain_generator.render();
    draw_id_labels(editor);

    static bool render_content_browser = true;
    if (gloco()->input.input_action.is_active("toggle_content_browser")) {
      render_content_browser = !render_content_browser;
    }
    if (render_content_browser) {
      editor.content_browser.render();
    }
  }

}

void fte_t::render() {
  fan::graphics::gui::tilemap_editor::ui::handle_gui(*this);
}

#endif
#endif

#endif