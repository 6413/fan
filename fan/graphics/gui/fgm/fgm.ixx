module;

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
import fan.memory;
import fan.io.file;

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

#define BLL_set_AreWeInsideStruct 1
#include <fan/fan_bll_preset.h>
#define BLL_set_prefix shape_list
#define BLL_set_type_node uint32_t
#define BLL_set_NodeDataType shapes_t::global_t*
#define BLL_set_Link 1
#include <BLL/BLL.h>

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

      key_handle = gloco()->window.add_keys_callback([this](const auto& d) {
        if (d.state != fan::keyboard_state::press || gui::is_any_item_active()) return;
        if (d.key == fan::key_r) erase_current();
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
      auto it = shape_list.GetNodeFirst();
      while (it != shape_list.dst) {
        delete shape_list[it];
        it = it.Next(&shape_list);
      }
      shape_list.Clear();
      close_cb();
      background.erase();
      shape_original_json.clear();
    }

    void update_line_thickness() {
      f32_t line_thickness = std::max(2.0f / fan::graphics::camera_get_zoom(render_view.camera), 2.0f);
      axis_lines[0].set_thickness(line_thickness);
      axis_lines[1].set_thickness(line_thickness);
    }

    bool id_exists(const std::string& id) {
      auto it = shape_list.GetNodeFirst();
      while (it != shape_list.dst) {
        if (shape_list[it]->id == id) return true;
        it = it.Next(&shape_list);
      }
      return false;
    }

    shape_list_t::nr_t push_shape(uint16_t shape_type, const fan::vec2& pos, const fan::vec2& size = 128) {
      auto nr = shape_list.NewNodeLast();
      switch (shape_type) {
        case fan::graphics::shapes::shape_type_t::sprite: {
          shape_list[nr] = new shapes_t::global_t(shape_type, fan::graphics::sprite_t{{.render_view = &render_view, .position = pos, .size = size}}, current_z, current_shape);
          auto* ri = ((fan::graphics::shapes::sprite_t::ri_t*)shape_list[nr]->children[0].GetData(fan::graphics::g_shapes->shaper));
          animations_application.current_animation_nr = ri->sprite_sheet_data.current_sprite_sheet;
          animations_application.current_animation_shape_nr = ri->sprite_sheet_data.shape_sprite_sheets;
          break;
        }
        case fan::graphics::shapes::shape_type_t::unlit_sprite: {
          shape_list[nr] = new shapes_t::global_t(shape_type, fan::graphics::unlit_sprite_t{{.render_view = &render_view, .position = pos, .size = size}}, current_z, current_shape);
          break;
        }
        case fan::graphics::shapes::shape_type_t::rectangle: {
          shape_list[nr] = new shapes_t::global_t(shape_type, fan::graphics::rectangle_t{{.render_view = &render_view, .position = pos, .size = size}}, current_z, current_shape);
          break;
        }
        case fan::graphics::shapes::shape_type_t::light: {
          shape_list[nr] = new shapes_t::global_t(shape_type, fan::graphics::light_t{{.render_view = &render_view, .position = pos, .size = size}}, current_z, current_shape);
          shape_list[nr]->children.push_back(fan::graphics::circle_t {{
            .render_view = &render_view,
            .position = fan::vec3(pos, current_z),
            .radius = size.x,
            .color = fan::color(1, 1, 1, 0.0),
            .blending = true
          }});
          break;
        }
      }
      return nr;
    }

    void render_tree_with_unified_selection() {
      auto it = shape_list.GetNodeFirst();
      static int selection_mask = 0;
      int node_clicked = -1;
      static gui::tree_node_flags_t base_flags = gui::tree_node_flags_open_on_arrow | gui::tree_node_flags_open_on_double_click | gui::tree_node_flags_span_avail_width;

      while (it != shape_list.dst) {
        auto& shape_instance = shape_list[it];
        gui::tree_node_flags_t node_flags = base_flags;
        if ((selection_mask & (1 << (std::intptr_t)it.NRI)) != 0) node_flags |= gui::tree_node_flags_selected;
        if (shape_instance->children.size() <= 1) node_flags |= gui::tree_node_flags_leaf;

        std::string_view shape_name = shape_instance->children.empty() ? std::string_view("Node") : fan::graphics::shape_names[shape_instance->children[0].get_shape_type()];
        bool node_open = gui::tree_node_ex((void*)(intptr_t)it.NRI, node_flags, "%.*s %ld", static_cast<int>(shape_name.length()), shape_name.data(), (intptr_t)it.NRI);

        if (gui::is_item_clicked() && !gui::is_item_toggled_open()) {
          node_clicked = (intptr_t)it.NRI;
          if (current_shape) current_shape->disable_highlight();
          current_shape = shape_instance;
          current_shape->enable_highlight();
        }

        if (node_open) {
          if (shape_instance->children.size() > 1) {
            render_child_nodes(node_clicked, shape_instance->children, selection_mask, base_flags);
          }
          gui::tree_pop();
        }
        it = it.Next(&shape_list);
      }

      if (node_clicked != -1) {
        selection_mask = gui::get_io().KeyCtrl ? (selection_mask ^ (1 << node_clicked)) : (1 << node_clicked);
      }
    }

    void render_child_nodes(int& node_clicked, std::vector<fan::graphics::shapes::shape_t>& children, int& selection_mask, gui::tree_node_flags_t base_flags) {
      int child_index = 0;
      for (auto& child : children) {
        gui::tree_node_flags_t node_flags = base_flags;
        if ((selection_mask & (1 << (intptr_t)child.NRI)) != 0) node_flags |= gui::tree_node_flags_selected;
        if (child_index + 1 >= children.size()) node_flags |= gui::tree_node_flags_leaf;

        bool node_open = gui::tree_node_ex((void*)(intptr_t)child.NRI, node_flags, "%s %u", fan::graphics::shape_names[child.get_shape_type()], child.NRI);

        if (gui::is_item_clicked() && !gui::is_item_toggled_open()) {
          node_clicked = (intptr_t)child.NRI;
          auto it = shape_list.GetNodeFirst();
          while (it != shape_list.dst) {
            if (shape_list[it]->children[0] == child) {
              if (current_shape) current_shape->disable_highlight();
              current_shape = shape_list[it];
              current_shape->enable_highlight();
              break;
            }
            it = it.Next(&shape_list);
          }
        }
        if (node_open) gui::tree_pop();
        child_index++;
      }
    }

    void render() {
      f32_t zoom = fan::graphics::camera_get_zoom(render_view.camera);
      fan::vec2 mouse_world = viewport_t::get_mouse_position(viewport_settings, gui::get_mouse_pos(), zoom, fan::vec2(gui::get_style().WindowPadding));
      selection.update(*this, shape_list, mouse_world, zoom);

      if (gloco()->input.input_action.is_active("set_windowed_fullscreen")) gloco()->window.set_borderless();
      if (gloco()->input.input_action.is_active("toggle_content_browser")) render_content_browser = !render_content_browser;
      if (gloco()->input.input_action.is_active("save_file")) fout(previous_filename);

      if (render_content_browser) content_browser.render();

      if (gui::begin("Editor", nullptr, gui::window_flags_menu_bar | gui::window_flags_no_background)) {
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
        tc_offset.x -= floor(tc_offset.x);
        tc_offset.y -= floor(tc_offset.y);
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

        if (!selection.objects.empty()) {
          fan::vec2 viewport_center = viewport_settings.start_pos - fan::vec2(style.WindowPadding) + viewport_settings.size / 2.f;
          selection.gizmo.manipulate(*selection.objects[0], camera_pos, zoom, viewport_center, snap);
        }

        gui::text(fan::to_string(zoom * 100) + " %", {.offset = {0.f, -gui::get_text_line_height_with_spacing()}, .align = align_e::bottom_right});
        
        fan::vec2 cursor_pos = (gui::get_mouse_pos() - viewport_settings.start_pos + fan::vec2(style.WindowPadding)) - viewport_settings.size / 2;
        std::string cursor_pos_str = cursor_pos.to_string(1);
        gui::text(cursor_pos_str.substr(1, cursor_pos_str.size() - 2), {.align = align_e::bottom_right});
        gui::set_cursor_pos(gui::get_cursor_start_pos());

        content_browser.receive_drag_drop_target([&](const std::string& file) {
          if (fan::io::file::extension(file) == ".json") {
            fin(file);
          } else {
            auto image = gloco()->image_load(file);
            fan::vec2 original_size = gloco()->image_get_data(image).size;
            shape_list[push_shape(fan::graphics::shapes::shape_type_t::sprite, 0, fan::vec2(128.f * (original_size.x / original_size.y), 128.f))]->children[0].set_image(image);
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
              auto& node = shape_list[push_shape(fan::graphics::shapes::shape_type_t::sprite, 0, fan::vec2(128.f * found->aspect_ratio, 128.f))];
              node->children[0].load_tp(&ti);
              node->children[0].get_image_data().image_path = path;
            }
          }
        });

        if (fan::window::is_key_down(fan::key_left_control) && fan::window::is_key_clicked(fan::key_d)) {
          for (auto& i : selection.objects) {
            for (auto& child : i->children) {
              shape_list[shape_list.NewNodeLast()] = new shapes_t::global_t(child.get_shape_type(), child, current_z, current_shape);
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

      if (gui::begin("Settings")) {
        gui::color_edit3("background", &gloco()->renderer_state.clear_color);
        gui::color_edit3("ambient", &gloco()->renderer_state.lighting.ambient);
        gui::drag("grid snap", &snap, 1, 0, std::numeric_limits<f32_t>::max(), gui::slider_flags_always_clamp);
        if (gui::checkbox("render axes", &render_axis_lines)) {
          axis_lines[0].set_color(axis_x_color * render_axis_lines);
          axis_lines[1].set_color(axis_y_color * render_axis_lines);
        }
      }
      gui::end();

      properties_ui_t::render(*this, current_shape);

      if (gui::begin("Shapes", nullptr)) {
        shapes_window_hovered = gui::is_window_hovered(gui::hovered_flags_allow_when_blocked_by_active_item);
        if (shapes_window_hovered && fan::window::is_mouse_clicked(fan::mouse_right)) gui::open_popup("ContextMenu");
        if (gui::begin_popup("ContextMenu")) {
          if (gui::begin_menu("Create")) {
            if (gui::begin_menu("Shapes")) {
              if (gui::menu_item("Sprite")) push_shape(fan::graphics::shapes::shape_type_t::sprite, 0);
              gui::end_menu();
            }
            if (gui::begin_menu("Lights")) {
              if (gui::menu_item("Circle")) push_shape(fan::graphics::shapes::shape_type_t::light, 0);
              gui::end_menu();
            }
            gui::end_menu();
          }
          gui::end_popup();
        }
        render_tree_with_unified_selection();
      }
      gui::end();

      if (gui::begin("Texture Pack")) {
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
      }
      gui::end();

      if (gui::begin("Shape Animations")) {
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
      }
      gui::end();

      for (auto& [shape, anim] : shape_sprite_sheets) {
        if (anim.is_playing && anim.owner_shape && !anim.keyframes.empty()) {
          anim.update(gloco()->get_delta_time());
          anim.apply_to_shape(anim.owner_shape);
        }
      }
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

    void erase_current() {
      if (!current_shape) return;
      auto it = shape_list.GetNodeFirst();
      while (it != shape_list.dst) {
        if (current_shape == shape_list[it]) {
          shape_original_json.erase(current_shape);
          if (auto sel_it = std::find(selection.objects.begin(), selection.objects.end(), current_shape); sel_it != selection.objects.end()) {
            selection.objects.erase(sel_it);
          }
          delete shape_list[it];
          shape_list.unlrec(it);
          invalidate_current();
          break;
        }
        it = it.Next(&shape_list);
      }
    }

    fan::graphics::shape_t axis_lines[2];
    fan::vec2 texturepack_size {};
    fan::vec2 texturepack_single_image_size {};
    std::vector<texturepack_image_t> texturepack_images;
    fan::graphics::render_view_t render_view;
    gui::content_browser_t content_browser {false};
    shapes_t::global_t* current_shape = nullptr;
    shape_list_t shape_list;
    f32_t current_z = 1;
    uint32_t current_id = 0;
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
    fan::graphics::engine_t::keys_handle_t key_handle;
    fan::graphics::engine_t::mouse_move_handle_t mouse_move_handle;
    fan::graphics::engine_t::buttons_handle_t button_handle;
    
    std::unordered_map<shapes_t::global_t*, fan::json> shape_original_json;
    std::unordered_map<shapes_t::global_t*, shape_keyframe_animation_t> shape_sprite_sheets;

    static constexpr fan::color axis_x_color = (fan::colors::red / 2.f).set_alpha(0.8f);
    static constexpr fan::color axis_y_color = (fan::colors::green / 2.f).set_alpha(0.8f);
    bool render_axis_lines = true;
  };
}