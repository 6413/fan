struct fgm_t {
  fgm_t() {}

  static constexpr f32_t scroll_speed = 1.2;
  static constexpr auto max_depth = 0xff;
  static constexpr int max_path_input = 40;
  static constexpr int max_id_input = 20;
  static constexpr fan::vec2 default_button_size{100, 30};

  inline static fan::graphics::file_open_dialog_t open_tp_dialog;

  struct texturepack_image_t {
    fan::graphics::image_t image;
    fan::vec2 uv0;
    fan::vec2 uv1;
    std::wstring image_name;
    f32_t aspect_ratio;
  };

  struct shapes_t {
    struct global_t : fan::graphics::vfi_root_t, fan::graphics::gui::imgui_element_t {
      global_t() = default;

      template <typename T>
      global_t(uint16_t shape_type, fgm_t* fgm, const T& obj, bool shape_add = true) 
        : fan::graphics::gui::imgui_element_t() {
        T temp = obj;
        this->shape_type = shape_type;
        typename fan::graphics::shapes::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += shape_add ? fgm->current_z++ : 0;
        vfip.shape.rectangle->size = temp.get_size();
        vfip.shape.rectangle->angle = 0;
        vfip.shape.rectangle->rotation_point = 0;
        vfip.shape.rectangle->camera = fgm->render_view.camera;
        vfip.shape.rectangle->viewport = fgm->render_view.viewport;
        vfip.mouse_button_cb = [fgm, this](const auto& d) -> int {
          fgm->current_shape = this;
          return 0;
        };
        fan::graphics::vfi_root_t::set_root(vfip);
        if (shape_add) {
          temp.set_position(fan::vec3(fan::vec2(temp.get_position()), fgm->current_z - 1));
        }
        fan::graphics::vfi_root_t::push_child(temp);
        fgm->current_shape = this;
      }

      uint16_t shape_type = 0;
      std::string id;
      uint32_t group_id = 0;
    };
  };

#include <fan/graphics/gui/fgm/common.h>
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

  void open(const std::string& texturepack_name, const std::wstring& asset_path) {
    using namespace fan::graphics;

    content_browser.init(asset_path);
    content_browser.current_view_mode = gui::content_browser_t::view_mode_large_thumbnails;

    render_view.camera = gloco()->open_camera(fan::vec2(0, 1), fan::vec2(0, 1));
    render_view.viewport = gloco()->open_viewport(fan::vec2(0), fan::vec2(1));

    auto transparent_texture = gloco()->create_transparent_texture();
    background = fan::graphics::sprite_t{{
      .render_view = &render_view,
      .position = 0,
      .size = 0,
      .image = transparent_texture,
    }};
    
    if (texturepack_name.size()) {
      open_texturepack(texturepack_name);
    }

    key_handle = gloco()->window.add_keys_callback([this](const auto& d) {
      if (d.state != fan::keyboard_state::press || gui::is_any_item_active()) {
        return;
      }
      if (d.key == fan::key_r) {
        erase_current();
      }
    });

    gloco()->input_action.add_keycombo({fan::input::key_left_control, fan::input::key_space}, "toggle_content_browser");
    gloco()->input_action.add_keycombo({fan::input::key_left_control, fan::input::key_f}, "set_windowed_fullscreen");
    gloco()->input_action.add_keycombo({fan::input::key_left_control, fan::input::key_s}, "save_file");

    mouse_move_handle = gloco()->window.add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / fan::graphics::camera_get_zoom(render_view.camera);
        auto camera_pos = viewport_settings.pos - move_off;
        gloco()->camera_set_position(render_view.camera, camera_pos);
      }
    });

    button_handle = gloco()->window.add_buttons_callback([this](const auto& d) {
      switch (d.button) {
      case fan::mouse_middle: {
        viewport_settings.move = (bool)d.state;
        viewport_settings.offset = gloco()->get_mouse_position();
        viewport_settings.pos = gloco()->camera_get_position(render_view.camera);
        break;
      }
      case fan::mouse_scroll_up: {
        if (viewport_settings.editor_hovered) {
          fan::graphics::camera_set_zoom(render_view.camera, fan::graphics::camera_get_zoom(render_view.camera) * scroll_speed);
          update_line_thickness();
        }
        return;
      }
      case fan::mouse_scroll_down: {
        if (viewport_settings.editor_hovered) {
          fan::graphics::camera_set_zoom(render_view.camera, fan::graphics::camera_get_zoom(render_view.camera) / scroll_speed);
          update_line_thickness();
        }
        return;
      }
      }
    });

    xy_lines[0] = fan::graphics::line_t{{
      .render_view = &render_view,
      .src = fan::vec3(-0xfffff, 0, 0x1fff),
      .dst = fan::vec2(0xfffff, 0),
      .color = fan::colors::red / 2
    }};

    xy_lines[1] = fan::graphics::line_t{{
      .render_view = &render_view,
      .src = fan::vec3(0, -0xfffff, 0x1fff),
      .dst = fan::vec2(0, 0xfffff),
      .color = fan::colors::green / 2
    }};

    drag_select = fan::graphics::rectangle_t{{
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
  }

  void update_line_thickness() {
    f32_t line_thickness = std::max(2.0 / fan::graphics::camera_get_zoom(render_view.camera), 2.0);
    xy_lines[0].set_thickness(line_thickness);
    xy_lines[1].set_thickness(line_thickness);
    if (current_shape) {
      for (auto& i : current_shape->highlight) {
        for (auto& line : i) {
          if (line) {
            line.set_thickness(line_thickness);
          }
        }
      }
    }
  }

  bool id_exists(const std::string& id) {
    auto it = shape_list.GetNodeFirst();
    while (it != shape_list.dst) {
      if (shape_list[it]->id == id) {
        return true;
      }
      it = it.Next(&shape_list);
    }
    return false;
  }

  void open_properties(fgm_t::shapes_t::global_t* shape, const fan::vec2& editor_size) {
    using namespace fan::graphics;

    std::string shape_str = std::string("Shape name:") + fan::graphics::shape_names[shape->children[0].get_shape_type()];
    gui::text(shape_str);

    fan::vec3 pos = shape->get_position();
    if (gui::drag("shape position", &pos, 0.1f)) {
      pos.z = (int)pos.z;
      shape->set_position(pos);
    }

    fan::vec2 size = shape->get_size();
    if (gui::drag("shape size", &size, 0.1f)) {
      shape->set_size(size);
    }

    auto sti = shape->children[0].get_shape_type();
    if (sti == fan::graphics::shapes::shape_type_t::particles) {
      gui::shape_properties(shape->children[0]);
    }
    else {
      fan::color c = shape->get_color();
      if (gui::color_edit4("color", &c)) {
        shape->set_color(c);
      }

      fan::vec3 angle = shape->children[0].get_angle();
      angle.x = fan::math::degrees(angle.x);
      angle.y = fan::math::degrees(angle.y);
      angle.z = fan::math::degrees(angle.z);
      if (gui::slider("shape angle", &angle, 0, 360)) {
        angle = fan::math::radians(angle);
        shape->children[0].set_angle(angle);
      }
    }

    std::string& id = current_shape->id;
    std::string str = id;
    if (gui::input_text("id", &str)) {
      if (gui::is_item_deactivated_after_edit() && !id_exists(str)) {
        id = str;
      }
    }

    std::string id_str = std::to_string(current_shape->group_id);
    str = id_str;
    if (gui::input_text("group id", &str)) {
      if (gui::is_item_deactivated_after_edit()) {
        current_shape->group_id = std::stoul(str);
      }
    }

    switch (shape->children[0].get_shape_type()) {
    case fan::graphics::shapes::shape_type_t::unlit_sprite:
    case fan::graphics::shapes::shape_type_t::sprite: {
      fan::graphics::gui::render_texture_property(shape->children[0], 0, "Base texture", content_browser.asset_path);
      fan::graphics::gui::render_texture_property(shape->children[0], 1, "Normal map", content_browser.asset_path);
      fan::graphics::gui::render_texture_property(shape->children[0], 2, "Specular map", content_browser.asset_path);
      fan::graphics::gui::render_texture_property(shape->children[0], 3, "Occlusion map", content_browser.asset_path);

      int current_image_filter = gloco()->image_get_settings(shape->children[0].get_image()).min_filter;
      static const char* image_filters[] = {"nearest", "linear"};
      if (gui::combo("image filter", &current_image_filter, image_filters, std::size(image_filters))) {
        fan::graphics::image_load_properties_t ilp;
        ilp.min_filter = current_image_filter;
        ilp.mag_filter = current_image_filter;
        gloco()->image_set_settings(shape->children[0].get_image(), ilp);
        if (shape->children[0].get_images()[0].iic() == false) {
          gloco()->image_set_settings(shape->children[0].get_images()[0], ilp);
        }
      }

      std::string& current = shape->children[0].get_image_data().image_path;
      str = current;
      if (gui::input_text("image path", &str)) {
        if (gui::is_item_deactivated_after_edit()) {
          fan::graphics::texture_pack::ti_t ti;
          if (gloco()->texture_pack.qti(str.c_str(), &ti)) {
            fan::print_no_space("failed to load texture:", str);
          }
          else {
            current = str.substr(0, std::strlen(str.c_str()));
            auto& data = gloco()->texture_pack.get_pixel_data(ti.unique_id);
            if (shape->children[0].get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
              shape->children[0].load_tp(&ti);
            }
            else if (shape->children[0].get_shape_type() == fan::graphics::shapes::shape_type_t::unlit_sprite) {
              shape->children[0].load_tp(&ti);
            }
          }
        }
      }
      break;
    }
    }
  }

  shape_list_t::nr_t push_shape(uint16_t shape_type, const fan::vec2& pos, const fan::vec2& size = 128) {
    auto nr = shape_list.NewNodeLast();

    switch (shape_type) {
    case fan::graphics::shapes::shape_type_t::sprite: {
      shape_list[nr] = new shapes_t::global_t{
        fan::graphics::shapes::shape_type_t::sprite,
        this, fan::graphics::sprite_t{{
          .render_view = &render_view,
          .position = pos,
          .size = size
        }}};
      auto* ri = ((fan::graphics::shapes::sprite_t::ri_t*)shape_list[nr]->children[0].GetData(fan::graphics::g_shapes->shaper));
      animations_application.current_animation_nr = ri->sprite_sheet_data.current_sprite_sheet;
      animations_application.current_animation_shape_nr = ri->sprite_sheet_data.shape_sprite_sheets;
      break;
    }
    case fan::graphics::shapes::shape_type_t::unlit_sprite: {
      shape_list[nr] = new shapes_t::global_t{
        fan::graphics::shapes::shape_type_t::unlit_sprite,
        this, fan::graphics::unlit_sprite_t{{
          .render_view = &render_view,
          .position = pos,
          .size = size
        }}};
      break;
    }
    case fan::graphics::shapes::shape_type_t::rectangle: {
      shape_list[nr] = new shapes_t::global_t{
        fan::graphics::shapes::shape_type_t::rectangle,
        this, fan::graphics::rectangle_t{{
          .render_view = &render_view,
          .position = pos,
          .size = size
        }}};
      break;
    }
    case fan::graphics::shapes::shape_type_t::light: {
      shape_list[nr] = new shapes_t::global_t{
        fan::graphics::shapes::shape_type_t::light,
        this, fan::graphics::light_t{{
          .render_view = &render_view,
          .position = pos,
          .size = size
        }}};
      shape_list[nr]->push_child(fan::graphics::circle_t{{
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
    int node_index = 0;
    static int selection_mask = 0;
    int node_clicked = -1;
    static fan::graphics::gui::tree_node_flags_t base_flags =
      fan::graphics::gui::tree_node_flags_open_on_arrow |
      fan::graphics::gui::tree_node_flags_open_on_double_click |
      fan::graphics::gui::tree_node_flags_span_avail_width;

    while (it != shape_list.dst) {
      auto& shape_instance = shape_list[it];
      fan::graphics::gui::tree_node_flags_t node_flags = base_flags;
      const bool is_selected = (selection_mask & (1 << (intptr_t)it.NRI)) != 0;
      if (is_selected) {
        node_flags |= fan::graphics::gui::tree_node_flags_selected;
      }

      bool node_open = fan::graphics::gui::tree_node_ex(
        (void*)(intptr_t)it.NRI, node_flags, "Node %ld", (intptr_t)it.NRI);

      if (fan::graphics::gui::is_item_clicked() && !fan::graphics::gui::is_item_toggled_open()) {
        node_clicked = (intptr_t)it.NRI;
      }

      if (node_open) {
        render_child_nodes(node_clicked, shape_instance->children, selection_mask, base_flags);
        fan::graphics::gui::tree_pop();
      }
      it = it.Next(&shape_list);
      node_index++;
    }

    if (node_clicked != -1) {
      if (fan::graphics::gui::get_io().KeyCtrl) {
        selection_mask ^= (1 << node_clicked);
      }
      else {
        selection_mask = (1 << node_clicked);
      }
    }
  }

  void render_child_nodes(
    int& node_clicked,
    std::vector<fan::graphics::vfi_root_t::child_data_t>& children,
    int& selection_mask,
    fan::graphics::gui::tree_node_flags_t base_flags
  ) {
    int child_index = 0;
    for (auto& child : children) {
      fan::graphics::gui::tree_node_flags_t node_flags = base_flags;
      const bool is_selected = (selection_mask & (1 << (intptr_t)child.NRI)) != 0;
      if (is_selected) {
        node_flags |= fan::graphics::gui::tree_node_flags_selected;
      }

      if (child_index + 1 >= children.size()) {
        node_flags |= fan::graphics::gui::tree_node_flags_leaf;
      }

      bool node_open = fan::graphics::gui::tree_node_ex(
        (void*)(intptr_t)child.NRI,
        node_flags,
        "%s %u",
        fan::graphics::shape_names[child.get_shape_type()],
        child.NRI
      );

      if (fan::graphics::gui::is_item_clicked() && !fan::graphics::gui::is_item_toggled_open()) {
        node_clicked = (intptr_t)child.NRI;
        auto it = shape_list.GetNodeFirst();
        while (it != shape_list.dst) {
          auto& shape = shape_list[it];
          if (shape->children[0] == child) {
            if (current_shape) {
              current_shape->disable_highlight();
            }
            current_shape = shape;
            current_shape->enable_highlight();
            break;
          }
          it = it.Next(&shape_list);
        }
      }

      if (node_open) {
        fan::graphics::gui::tree_pop();
      }
      child_index++;
    }
  }

  fan::vec2 get_mouse_position() {
    using namespace fan::graphics;
    auto& style = gui::get_style();
    fan::vec2 pos = fan::vec2((gui::get_mouse_pos() - viewport_settings.start_pos + fan::vec2(style.WindowPadding)) - viewport_settings.size / 2);
    pos = fan::vec2(viewport_settings.pos) + pos / fan::graphics::camera_get_zoom(render_view.camera);
    return pos;
  }

  void render() {
    using namespace fan::graphics;

    if (viewport_settings.editor_hovered && fan::window::is_mouse_clicked() && !fan::graphics::vfi_root_t::moving_object) {
      drag_start = get_mouse_position();
    }
    else if (viewport_settings.editor_hovered && fan::window::is_mouse_down() && !fan::graphics::vfi_root_t::moving_object) {
      fan::vec2 size = get_mouse_position() - drag_start;
      for (auto& i : fan::graphics::vfi_root_t::selected_objects) {
        i->disable_highlight();
      }
      fan::graphics::vfi_root_t::selected_objects.clear();
      drag_select.set_position(drag_start + size / 2);
      drag_select.set_size(size / 2);
    }
    else if (fan::window::is_mouse_released() && !shapes_window_hovered && !gui::is_any_item_active()) {
      bool hit_any = false;
      auto it = shape_list.GetNodeFirst();
      while (it != shape_list.dst) {
        auto& shape = shape_list[it];
        if (drag_select.intersects(shape->children[0])) {
          if (!fan::graphics::vfi_root_t::moving_object &&
            (drag_select.get_size().x >= 1 && drag_select.get_size().y >= 1)) {
            shape->enable_highlight();
            fan::graphics::vfi_root_t::selected_objects.push_back(shape);
          }
        }
        if (fan_2d::collision::rectangle::point_inside_no_rotation(
          get_mouse_position(), shape->children[0].get_position(),
          shape->children[0].get_size()
        )) {
          hit_any = true;
          if (shape->children[0].get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
            auto* ri = ((fan::graphics::shapes::sprite_t::ri_t*)shape->children[0].GetData(fan::graphics::g_shapes->shaper));
            animations_application.current_animation_nr = ri->sprite_sheet_data.current_sprite_sheet;
            animations_application.current_animation_shape_nr = ri->sprite_sheet_data.shape_sprite_sheets;
          }
        }
        it = it.Next(&shape_list);
      }
      if (hit_any == false && drag_select.get_size().x == 0) {
        it = shape_list.GetNodeFirst();
        while (it != shape_list.dst) {
          auto& shape = shape_list[it];
          shape->disable_highlight();
          it = it.Next(&shape_list);
        }
        fan::graphics::vfi_root_t::selected_objects.clear();
      }
      drag_select.set_size(fan::vec2(0));
    }

    fan::vec2 editor_size;

    if (gloco()->input_action.is_active("set_windowed_fullscreen")) {
      gloco()->window.set_windowed_fullscreen();
    }

    if (gloco()->input_action.is_active("toggle_content_browser")) {
      render_content_browser = !render_content_browser;
    }
    if (gloco()->input_action.is_active("save_file")) {
      fout(previous_filename);
    }

    if (render_content_browser) {
      content_browser.render();
    }

    if (gui::begin("Editor", nullptr, gui::window_flags_menu_bar | gui::window_flags_no_background)) {
      fan::vec2 window_size = gloco()->window.get_size();
      fan::vec2 viewport_size = gui::get_window_size();
      fan::vec2 ratio = viewport_size / viewport_size.max();
      fan::vec2 s = viewport_size;

      editor_pos = gui::get_window_pos();

      f32_t zoom = fan::graphics::camera_get_zoom(render_view.camera);
      fan::vec2 ground_size = viewport_size * (1.0f / zoom);
      fan::vec2 camera_pos = gloco()->camera_get_position(render_view.camera);

      auto world_size = viewport_size / zoom;

      background.set_position(camera_pos);
      background.set_size(world_size / 2);

      f32_t tile_size = 128.0f;
      fan::vec2 tc_size = world_size / tile_size;
      background.set_tc_size(tc_size);

      fan::vec2 tc_offset = camera_pos / tile_size;
      tc_offset.x = tc_offset.x - floor(tc_offset.x);
      tc_offset.y = tc_offset.y - floor(tc_offset.y);

      background.set_tc_position(tc_offset - tc_size / 2);

      viewport_settings.editor_hovered = gui::is_window_hovered();
      fan::graphics::vfi_root_t::g_ignore_mouse = !viewport_settings.editor_hovered;

      auto& style = gui::get_style();
      fan::vec2 frame_padding = style.FramePadding;
      fan::vec2 viewport_pos = fan::vec2(0, frame_padding.y * 2);

      fan::vec2 vMin = gui::get_window_content_region_min();
      fan::vec2 vMax = gui::get_window_content_region_max();

      vMin.x += gui::get_window_pos().x;
      vMin.y += gui::get_window_pos().y;
      vMax.x += gui::get_window_pos().x;
      vMax.y += gui::get_window_pos().y;

      viewport_size = vMax - vMin + fan::vec2(style.WindowPadding) * 2;

      gloco()->viewport_set(
        render_view.viewport,
        vMin - fan::vec2(style.WindowPadding),
        viewport_size
      );

      gloco()->camera_set_ortho(
        render_view.camera,
        fan::vec2(-viewport_size.x / 2, viewport_size.x / 2),
        fan::vec2(-viewport_size.y / 2, viewport_size.y / 2)
      );

      viewport_settings.size = viewport_size;
      viewport_settings.start_pos = vMin;

      {
        std::string str = fan::to_string(gloco()->camera_get_zoom(render_view.camera) * 100);
        str += " %";
        gui::text_bottom_right(str, 1);
      }

      {
        fan::vec2 cursor_pos = ((gui::get_mouse_pos() - viewport_settings.start_pos + fan::vec2(style.WindowPadding)) - viewport_settings.size / 2);
        std::string cursor_pos_str = cursor_pos.to_string(1);
        std::string str = cursor_pos_str.substr(1, cursor_pos_str.size() - 2);
        gui::text_bottom_right(str.c_str(), 0);
      }

      gui::set_cursor_pos(gui::get_cursor_start_pos());

      content_browser.receive_drag_drop_target([&](const std::filesystem::path& fs) {
        auto file = fs.generic_string();
        auto extension = fan::io::file::extension(file);
        if (extension == ".json") {
          fin(file);
        }
        else {
          auto image = gloco()->image_load((fs).generic_string());
          fan::vec2 initial_size = 128.f;
          fan::vec2 original_size = gloco()->image_get_data(image).size;
          initial_size.x *= (original_size.x / original_size.y);
          auto nr = push_shape(fan::graphics::shapes::shape_type_t::sprite, get_mouse_position(), initial_size);
          shape_list[nr]->children[0].set_image(image);
        }
      });

      gui::receive_drag_drop_target("FGM_TEXTUREPACK_DROP", [&](const std::string& path) {
        fan::graphics::texture_pack_t::ti_t ti;
        if (gloco()->texture_pack.qti(path, &ti)) {
          fan::print_no_space("non texturepack texture or failed to load texture:", path);
        }
        else {
          auto image = ti.image;
          fan::vec2 initial_size = 128.f;
          std::wstring wpath(path.begin(), path.end());
          auto found = std::find_if(texturepack_images.begin(), texturepack_images.end(), [&](const texturepack_image_t& img) {
            return wpath == img.image_name;
          });
          if (found == texturepack_images.end()) {
            fan::print("some bug");
            return;
          }
          initial_size.x *= found->aspect_ratio;
          auto nr = push_shape(fan::graphics::shapes::shape_type_t::sprite, get_mouse_position(), initial_size);
          auto& node = shape_list[nr];
          node->children[0].load_tp(&ti);
          node->children[0].get_image_data().image_path = path;
        }
      });
    }

    if (fan::window::is_key_down(fan::key_left_control)) {
      if (fan::window::is_key_pressed(fan::key_d)) {
        for (auto& i : fan::graphics::vfi_root_t::selected_objects) {
          for (auto& child : i->children) {
            auto it = shape_list.NewNodeLast();
            auto& node = shape_list[it];
            node = new shapes_t::global_t{child.get_shape_type(), this, child};
          }
        }
        fan::graphics::vfi_root_t::selected_objects.clear();
      }
    }

    static fan::graphics::file_save_dialog_t save_file_dialog;
    static fan::graphics::file_open_dialog_t open_file_dialog;
    static std::string fn;

    if (fan::graphics::gui::begin_menu_bar()) {
      if (fan::graphics::gui::begin_menu("File")) {
        if (current_shape) {
          current_shape->move = false;
        }
        if (fan::graphics::gui::menu_item("Open..", "Ctrl+O")) {
          open_file_dialog.load("json;fmm", &fn);
        }
        if (fan::graphics::gui::menu_item("Save", "Ctrl+S")) {
          fout(previous_filename);
        }
        if (fan::graphics::gui::menu_item("Save as", "Ctrl+Shift+S")) {
          save_file_dialog.save("json;fmm", &fn);
        }
        if (fan::graphics::gui::menu_item("Quit")) {
          auto it = shape_list.GetNodeFirst();
          while (it != shape_list.dst) {
            delete shape_list[it];
            it = it.Next(&shape_list);
          }
          shape_list.Clear();
          close_cb();
          fan::graphics::gui::end();
        }
        fan::graphics::gui::end_menu();
      }
      fan::graphics::gui::end_menu_bar();
    }

    if (open_file_dialog.is_finished()) {
      if (fn.size() != 0) {
        auto it = shape_list.GetNodeFirst();
        while (it != shape_list.dst) {
          delete shape_list[it];
          it = it.Next(&shape_list);
        }
        shape_list.Clear();
        fin(fn);
      }
      open_file_dialog.finished = false;
      gui::end();
      return;
    }
    if (save_file_dialog.is_finished()) {
      if (fn.size() != 0) {
        fout(fn);
      }
      save_file_dialog.finished = false;
    }

    gui::end();

    if (gui::begin("Settings")) {
      if (gui::color_edit3("background", &gloco()->clear_color)) {
      }
      if (gui::color_edit3("ambient", &gloco()->lighting.ambient)) {
      }
      if (gui::drag("grid snap", &fan::graphics::vfi_root_t::snap, 1, 0, FLT_MAX, gui::slider_flags_always_clamp)) {
      }
    }
    gui::end();

    if (gui::begin("Properties")) {
      if (current_shape != nullptr) {
        open_properties(current_shape, editor_size);

        gui::new_line();
        gui::begin_child("properties_animations", 0, 1);
        gui::text("Animations");
        auto& shape = current_shape->children[0];
        fan::graphics::sprite_sheet_shape_id_t* shape_animation_nr = 0;
        if (shape.get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
          auto* ri = ((fan::graphics::shapes::sprite_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper));
          shape_animation_nr = &ri->sprite_sheet_data.shape_sprite_sheets;
          if (animations_application.current_animation_nr) {
            auto& anim = fan::graphics::get_sprite_sheet(animations_application.current_animation_nr);
          }
        }
        if (shape_animation_nr == nullptr) {
          goto g_end_animations;
        }
        {
          bool animation_changed = animations_application.render("CONTENT_BROWSER_ITEMS", *shape_animation_nr);
          if (shape.get_shape_type() == fan::graphics::shapes::shape_type_t::sprite && animations_application.current_animation_nr) {
            auto* ri = ((fan::graphics::shapes::sprite_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper));
            if (animations_application.current_animation_nr && animations_application.current_animation_shape_nr == *shape_animation_nr) {
              ri->sprite_sheet_data.current_sprite_sheet = animations_application.current_animation_nr;
            }
          }
          if (*shape_animation_nr && animations_application.current_animation_shape_nr == *shape_animation_nr && (animations_application.toggle_play_animation || animations_application.play_animation || animation_changed)) {
            if (shape.get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
              auto& anim = fan::graphics::get_sprite_sheet(animations_application.current_animation_nr);
              if (animation_changed && animations_application.current_animation_nr && animations_application.play_animation) {
                if (gloco()->is_image_valid(shape.get_image()) == false) {
                  shape.set_tc_position(0);
                  shape.set_tc_size(1);
                }
              }
              else if (anim.selected_frames.size()) {
              }

              auto& current_shape_anim = shape.get_sprite_sheet();
              if ((animations_application.toggle_play_animation || animation_changed) &&
                   animations_application.play_animation && 
                   current_shape_anim.selected_frames.size()) 
              {
                shape.play_sprite_sheet();
              }

              if (animations_application.toggle_play_animation && !animations_application.play_animation) {
                shape.stop_sprite_sheet();
              }
            }
          }
        }
      g_end_animations:
        gui::end_child();
      }
    }
    gui::end();

    if (fan::graphics::gui::begin("Shapes", nullptr)) {
      shapes_window_hovered = gui::is_window_hovered(gui::hovered_flags_allow_when_blocked_by_active_item);
      if (shapes_window_hovered && fan::window::is_mouse_clicked(fan::mouse_right)) {
        fan::graphics::gui::open_popup("ContextMenu");
      }
      if (fan::graphics::gui::begin_popup("ContextMenu")) {
        if (fan::graphics::gui::begin_menu("Create")) {
          if (fan::graphics::gui::begin_menu("Shapes")) {
            if (fan::graphics::gui::menu_item("Sprite")) {
              push_shape(fan::graphics::shapes::shape_type_t::sprite, 0);
            }
            fan::graphics::gui::end_menu();
          }
          if (fan::graphics::gui::begin_menu("Lights")) {
            if (fan::graphics::gui::menu_item("Circle")) {
              push_shape(fan::graphics::shapes::shape_type_t::light, 0);
            }
            fan::graphics::gui::end_menu();
          }
          fan::graphics::gui::end_menu();
        }
        fan::graphics::gui::end_popup();
      }
      render_tree_with_unified_selection();
    }

    static std::string filename;
    if (gui::begin("Texture Pack")) {
      if (gui::button("open")) {
        open_tp_dialog.load("ftp", &fn);
      }

      f32_t thumbnail_size = 128.0f;
      f32_t panel_width = gui::get_content_region_avail().x;
      f32_t padding = 16.0f;
      int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);

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
    if (open_tp_dialog.is_finished()) {
      if (fn.size() != 0) {
        open_texturepack(fn);
      }
      open_tp_dialog.finished = false;
    }
    gui::end();

    if (gui::begin("Shape Animations")) {
      if (current_shape && current_shape->children.size() > 0) {
        auto it = shape_sprite_sheets.find(current_shape);

        if (it == shape_sprite_sheets.end()) {
          // Create new animation and set owner immediately
          auto& new_anim = shape_sprite_sheets[current_shape];
          new_anim.owner_shape = current_shape;
          it = shape_sprite_sheets.find(current_shape);
        }

        it->second.render_gui(current_shape);
      }
      else {
        gui::text("Select a shape to animate");
      }
    }
    gui::end();

    for (auto& [shape, anim] : shape_sprite_sheets) {
      if (anim.is_playing && anim.owner_shape && !anim.keyframes.empty()) {
        anim.update(gloco()->delta_time);
        anim.apply_to_shape(anim.owner_shape);
      }
    }
  }

  void fout(std::string filename) {
    if (filename.empty()) {
      fan::graphics::gui::print_error("filename is empty. save file from 'File/Save as'");
      return;
    }
    if (!filename.ends_with(".json")) {
      filename += ".json";
    }
    previous_filename = filename;

    fan::json ostr;
    ostr["version"] = current_version;
    if (gloco()->lighting.ambient != fan::graphics::lighting_t().ambient) {
      ostr["lighting.ambient"] = gloco()->lighting.ambient;
    }
    if (gloco()->clear_color != fan::colors::black) {
      ostr["clear_color"] = gloco()->clear_color;
    }
    {
      auto animations_json = fan::graphics::sprite_sheet_serialize();
      if (animations_json.empty() == false) {
        ostr.update(animations_json, true);
      }
    }
    {
      fan::json shape_anims = fan::json::array();
      for (const auto& [shape, anim] : shape_sprite_sheets) {
        if (anim.keyframes.empty()) {
          continue;
        }
        fan::json anim_entry;
        auto it = shape_list.GetNodeFirst();
        int shape_index = 0;
        while (it != shape_list.dst) {
          if (shape_list[it] == shape) {
            anim_entry["shape_index"] = shape_index;
            anim_entry["animation"] = anim.serialize();
            shape_anims.push_back(anim_entry);
            break;
          }
          shape_index++;
          it = it.Next(&shape_list);
        }
      }
      if (!shape_anims.empty()) {
        ostr["shape_keyframe_animations"] = shape_anims;
      }
    }
    fan::json shapes = fan::json::array();
    auto it = shape_list.GetNodeFirst();
    while (it != shape_list.dst) {
      auto& shape_instance = shape_list[it];
      auto& shape = shape_instance->children[0];

      shapes_t::global_t defaults;

      fan::json shape_json = shape;

      if (shape_instance->id != defaults.id) {
        shape_json["id"] = shape_instance->id;
      }
      if (shape_instance->group_id != defaults.group_id) {
        shape_json["group_id"] = shape_instance->group_id;
      }
      shapes.push_back(shape_json);
      it = it.Next(&shape_list);
    }


    ostr["shapes"] = shapes;

    // set relative path from json to image
    ostr.find_and_iterate("image_path", [&filename](fan::json& value) {
      value = fan::io::file::relative_path(value.get<std::string>(), filename).generic_string();
    });

    fan::io::file::write(filename, ostr.dump(2), std::ios_base::binary);
    fan::graphics::gui::print_success("File saved to " + std::filesystem::absolute(filename).generic_string());
  }

  void load_tp(fgm_t::shape_list_NodeData_t& node) {
    fan::graphics::texture_pack_t::ti_t ti;
    if (gloco()->texture_pack.qti(node->children[0].get_image_data().image_path, &ti)) {
      fan::graphics::gui::print_warning("texturepack: non texturepack texture or failed to load texture:", node->children[0].get_image_data().image_path);
    }
    else {
      auto& data = gloco()->texture_pack.get_pixel_data(ti.unique_id);
      node->children[0].load_tp(&ti);
    }
  }

  void fin(const std::string& filename, const std::source_location& callers_path = std::source_location::current()) {
    using namespace fan::graphics;
    previous_filename = filename;

    std::string in;
    fan::io::file::read(filename, &in);
    fan::json json_in = fan::json::parse(in);
    if (json_in.contains("lighting.ambient")) {
      gloco()->lighting.ambient = json_in["lighting.ambient"];
    }
    if (json_in.contains("clear_color")) {
      gloco()->clear_color = json_in["clear_color"];
    }

    json_in.find_and_iterate("image_path", [&filename](fan::json& value) {
      std::filesystem::path json_path(filename);
      json_path = std::filesystem::absolute(json_path).parent_path();
      value = (json_path / std::filesystem::path(value.get<std::string>())).generic_string();
    });

    if (json_in.contains("animations")) {
      fan::graphics::sprite_sheets_parse(filename, json_in);
    }
    if (json_in.contains("shape_keyframe_animations")) {
      std::vector<std::pair<int, shape_keyframe_animation_t>> pending_animations;
      for (const auto& anim_entry : json_in["shape_keyframe_animations"]) {
        int shape_index = anim_entry["shape_index"].get<int>();
        shape_keyframe_animation_t anim;
        anim.deserialize(anim_entry["animation"]);
        pending_animations.push_back({ shape_index, anim });
      }

      for (const auto& [shape_index, anim] : pending_animations) {
        auto it = shape_list.GetNodeFirst();
        int current_index = 0;
        while (it != shape_list.dst) {
          if (current_index == shape_index) {
            shape_sprite_sheets[shape_list[it]] = anim;
            break;
          }
          current_index++;
          it = it.Next(&shape_list);
        }
      }
    }

    fan::graphics::shape_deserialize_t iterator;
    fan::graphics::shape_t shape;
    int i = 0;
    current_z = 0;
    std::string shapes = "shapes";
    if (json_in.contains("tiles")) {
      shapes = "tiles";
    }
    auto& shape_object = json_in;
    bool is_object = true;
    if (json_in.contains(shapes)) {
      shape_object = json_in[shapes];
      is_object = false;
    }
    else if (!json_in.contains("shape")) {
      gui::print_error("failed to parse .json file");
      return;
    }
    while (iterator.iterate(shape_object, &shape)) {
      shape.set_camera(render_view.camera);
      shape.set_viewport(render_view.viewport);
      current_z = std::max(current_z, shape.get_position().z);
      auto it = shape_list.NewNodeLast();
      auto& node = shape_list[it];
      switch (shape.get_shape_type()) {
      case fan::graphics::shapes::shape_type_t::sprite: {
        node = new fgm_t::shapes_t::global_t(
          fan::graphics::shapes::shape_type_t::sprite,
          this,
          shape,
          false
        );
        load_tp(node);
        node->children[0].get_image_data().image_path = shape.get_image_data().image_path;
        break;
      }
      case fan::graphics::shapes::shape_type_t::unlit_sprite: {
        node = new fgm_t::shapes_t::global_t(
          fan::graphics::shapes::shape_type_t::unlit_sprite,
          this,
          shape,
          false
        );
        load_tp(node);
        node->children[0].get_image_data().image_path = shape.get_image_data().image_path;
        break;
      }
      case fan::graphics::shapes::shape_type_t::rectangle: {
        node = new fgm_t::shapes_t::global_t(
          fan::graphics::shapes::shape_type_t::rectangle,
          this,
          shape,
          false
        );
        break;
      }
      case fan::graphics::shapes::shape_type_t::light: {
        node = new fgm_t::shapes_t::global_t(
          fan::graphics::shapes::shape_type_t::light,
          this,
          shape,
          false
        );
        node->push_child(fan::graphics::circle_t{{
          .render_view = &render_view,
          .position = shape.get_position(),
          .radius = shape.get_size().x,
          .color = shape.get_color(),
          .blending = true
        }});
        break;
      }
      case fan::graphics::shapes::shape_type_t::particles: {
        node = new fgm_t::shapes_t::global_t(
          fan::graphics::shapes::shape_type_t::particles,
          this,
          shape,
          false
        );
        break;
      }
      default: {
        gui::print("unimplemented shape type");
        break;
      }
      }
      if (!is_object) {
        const auto& shape_json = *(iterator.data.it - 1);
        if (shape_json.contains("id")) {
          node->id = shape_json["id"].get<std::string>();
        }
        if (shape_json.contains("group_id")) {
          node->group_id = shape_json["group_id"].get<uint32_t>();
        }
      }
    }
    ++current_z;
  }

  void invalidate_current() {
    current_shape = nullptr;
  }

  void erase_current() {
    if (current_shape == nullptr) {
      return;
    }

    auto it = shape_list.GetNodeFirst();
    while (it != shape_list.dst) {
      if (current_shape == shape_list[it]) {
        auto& selected_objs = fan::graphics::vfi_root_t::selected_objects;
        if (auto it = std::find(selected_objs.begin(), selected_objs.end(), current_shape); it != selected_objs.end()) {
          selected_objs.erase(it);
        }
        shape_list[it]->previous_focus = 0;
        delete shape_list[it];
        shape_list.unlrec(it);
        invalidate_current();
        break;
      }
      it = it.Next(&shape_list);
    }
  }

enum class rotation_position_e : uint8_t {
  center,
  top_left,
  top_middle,
  top_right,
  middle_left,
  middle_right,
  bottom_left,
  bottom_middle,
  bottom_right
};

struct shape_keyframe_animation_t {
  struct keyframe_t {
    static keyframe_t lerp(const keyframe_t& a, const keyframe_t& b, f32_t t) {
      keyframe_t result;
      result.time = a.time + (b.time - a.time) * t;
      result.position = a.position + (b.position - a.position) * t;
      result.size = a.size + (b.size - a.size) * t;
      result.angle = a.angle + (b.angle - a.angle) * t;
      result.rotation_pos = a.rotation_pos;
      return result;
    }

    f32_t time = 0.0f;
    fan::vec3 position = 0;
    fan::vec2 size = 0;
    fan::vec3 angle = 0;
    rotation_position_e rotation_pos = rotation_position_e::center;
  };

  fan::vec2 get_rotation_offset(rotation_position_e pos, const fan::vec2& half_size) const {
    switch (pos) {
      case rotation_position_e::top_left: return {-half_size.x, -half_size.y};
      case rotation_position_e::top_middle: return {0, -half_size.y};
      case rotation_position_e::top_right: return {half_size.x, -half_size.y};
      case rotation_position_e::middle_left: return {-half_size.x, 0};
      case rotation_position_e::middle_right: return {half_size.x, 0};
      case rotation_position_e::bottom_left: return {-half_size.x, half_size.y};
      case rotation_position_e::bottom_middle: return {0, half_size.y};
      case rotation_position_e::bottom_right: return {half_size.x, half_size.y};
      default: return {0, 0};
    }
  }

  void add_keyframe(const fan::vec3& pos, const fan::vec2& sz, const fan::vec3& ang, rotation_position_e rot_pos) {
    f32_t time = current_time;
    if (auto_increment_time && !keyframes.empty()) {
      time = keyframes.back().time + time_increment;
    }
    keyframe_t kf{time, pos, sz, ang, rot_pos};
    auto it = std::lower_bound(keyframes.begin(), keyframes.end(), kf,
      [](const keyframe_t& a, const keyframe_t& b) {
        return a.time < b.time;
      });
    keyframes.insert(it, kf);
    if (auto_increment_time) {
      current_time = time;
    }
  }

  void remove_keyframe(int index) {
    if (index >= 0 && index < keyframes.size()) {
      keyframes.erase(keyframes.begin() + index);
      if (selected_keyframe >= keyframes.size()) {
        selected_keyframe = keyframes.size() - 1;
      }
    }
  }

  keyframe_t get_current_frame() const {
    if (keyframes.empty()) {
      return keyframe_t{};
    }
    if (keyframes.size() == 1) {
      return keyframes[0];
    }

    for (size_t i = 0; i < keyframes.size() - 1; ++i) {
      if (current_time >= keyframes[i].time && current_time <= keyframes[i + 1].time) {
        f32_t delta = keyframes[i + 1].time - keyframes[i].time;
        if (delta <= 0.0f) {
          return keyframes[i];
        }
        f32_t t = (current_time - keyframes[i].time) / delta;
        return keyframe_t::lerp(keyframes[i], keyframes[i + 1], t);
      }
    }

    if (slerp_to_first && loop && keyframes.size() > 1) {
      f32_t last_time = keyframes.back().time;
      f32_t wrap_duration = slerp_duration;
      f32_t wrap_end = last_time + wrap_duration;
      
      if (current_time > last_time && current_time <= wrap_end) {
        f32_t t = (current_time - last_time) / wrap_duration;
        return keyframe_t::lerp(keyframes.back(), keyframes.front(), t);
      }
    }

    return keyframes.back();
  }

  void update(f32_t dt) {
    if (!is_playing || keyframes.empty() || !owner_shape) {
      return;
    }

    current_time += dt;
    f32_t max_time = keyframes.back().time;
    if (slerp_to_first && loop) {
      max_time += slerp_duration;
    }

    if (current_time > max_time) {
      if (loop) {
        current_time = 0.0f;
      }
      else {
        current_time = max_time;
        is_playing = false;
      }
    }
  }

  void apply_to_shape(shapes_t::global_t* shape) {
    if (keyframes.empty()) {
      return;
    }
    if (!shape || shape != owner_shape) {
      return;
    }
    
    auto saved_selected = fan::graphics::vfi_root_t::selected_objects;
    fan::graphics::vfi_root_t::selected_objects.clear();
    
    auto frame = get_current_frame();
    shape->set_position(frame.position);
    shape->children[0].set_size(frame.size);
    shape->children[0].set_angle(frame.angle);
    
    fan::vec2 offset = get_rotation_offset(frame.rotation_pos, frame.size);
    shape->children[0].set_rotation_point(offset);
    
    fan::graphics::vfi_root_t::selected_objects = saved_selected;
  }

  fan::json serialize() const {
    fan::json j;
    j["name"] = name;
    j["loop"] = loop;
    j["slerp_to_first"] = slerp_to_first;
    j["slerp_duration"] = slerp_duration;
    j["current_rotation_pos"] = static_cast<int>(current_rotation_pos);
    j["auto_increment_time"] = auto_increment_time;
    j["time_increment"] = time_increment;
    
    fan::json keyframes_json = fan::json::array();
    for (const auto& kf : keyframes) {
      fan::json kf_json;
      kf_json["time"] = kf.time;
      kf_json["position"] = kf.position;
      kf_json["size"] = kf.size;
      kf_json["angle"] = kf.angle;
      kf_json["rotation_pos"] = static_cast<int>(kf.rotation_pos);
      keyframes_json.push_back(kf_json);
    }
    j["keyframes"] = keyframes_json;
    
    return j;
  }

  void deserialize(const fan::json& j) {
    if (j.contains("name")) {
      name = j["name"].get<std::string>();
    }
    if (j.contains("loop")) {
      loop = j["loop"].get<bool>();
    }
    if (j.contains("slerp_to_first")) {
      slerp_to_first = j["slerp_to_first"].get<bool>();
    }
    if (j.contains("slerp_duration")) {
      slerp_duration = j["slerp_duration"].get<f32_t>();
    }
    if (j.contains("current_rotation_pos")) {
      current_rotation_pos = static_cast<rotation_position_e>(j["current_rotation_pos"].get<int>());
    }
    if (j.contains("auto_increment_time")) {
      auto_increment_time = j["auto_increment_time"].get<bool>();
    }
    if (j.contains("time_increment")) {
      time_increment = j["time_increment"].get<f32_t>();
    }
    
    keyframes.clear();
    selected_keyframe = -1;
    current_time = 0.0f;
    is_playing = false;
    
    if (j.contains("keyframes")) {
      for (const auto& kf_json : j["keyframes"]) {
        keyframe_t kf;
        kf.time = kf_json["time"].get<f32_t>();
        kf.position = kf_json["position"];
        kf.size = kf_json["size"];
        kf.angle = kf_json["angle"];
        kf.rotation_pos = static_cast<rotation_position_e>(kf_json["rotation_pos"].get<int>());
        keyframes.push_back(kf);
      }
    }
  }

  void render_gui(shapes_t::global_t* shape) {
    using namespace fan::graphics;

    gui::text(fan::format("Shape ptr: {} | Owner ptr: {} | Keyframes: {}", 
      (void*)shape, (void*)owner_shape, keyframes.size()).c_str());

    if (owner_shape != shape) {
      gui::text("ERROR: Wrong animation! This belongs to another shape.");
      gui::text("This should never happen - check map logic.");
      return;
    }

    gui::input_text("Name", &name);

    if (gui::button(is_playing ? "Stop" : "Play")) {
      is_playing = !is_playing;
      if (is_playing && current_time >= (keyframes.empty() ? 0 : keyframes.back().time)) {
        current_time = 0.0f;
      }
    }
    gui::same_line();
    if (gui::button("Reset")) {
      current_time = 0.0f;
    }
    gui::same_line();
    gui::checkbox("Loop", &loop);
    gui::same_line();
    gui::checkbox("Slerp to First", &slerp_to_first);

    gui::checkbox("Auto Increment Time", &auto_increment_time);
    if (auto_increment_time) {
      gui::drag("Time Increment", &time_increment, 0.01f, 0.0f, 10.0f);
    }

    if (slerp_to_first) {
      gui::drag("Slerp Duration", &slerp_duration, 0.01f, 0.0f, 10.0f);
    }

    f32_t max_time = keyframes.empty() ? 10.f : keyframes.back().time;
    if (slerp_to_first && loop) {
      max_time += slerp_duration;
    }
    if (gui::slider("Time##current_time", &current_time, 0.0f, std::max(max_time, 0.1f))) {
      apply_to_shape(shape);
    }

    gui::separator();
    gui::text("Keyframes: ", (int)keyframes.size());

    if (gui::button("Add Keyframe")) {
      add_keyframe(
        shape->get_position(),
        shape->children[0].get_size(),
        shape->children[0].get_angle(),
        current_rotation_pos
      );
    }
    gui::same_line();
    if (gui::button("Remove Selected") && selected_keyframe >= 0) {
      remove_keyframe(selected_keyframe);
    }

    gui::separator();
    gui::begin_child("keyframes_list", fan::vec2(0, 200), true);

    for (int i = 0; i < keyframes.size(); ++i) {
      gui::push_id(i);
      bool is_selected = (i == selected_keyframe);
      if (gui::selectable(fan::format("KF {}: {:.2f}s", i, keyframes[i].time).c_str(), is_selected)) {
        selected_keyframe = i;
        current_time = keyframes[i].time;
        apply_to_shape(shape);
      }
      gui::pop_id();
    }

    gui::end_child();

    if (selected_keyframe >= 0 && selected_keyframe < keyframes.size()) {
      gui::separator();
      auto& kf = keyframes[selected_keyframe];

      if (gui::drag("Time", &kf.time, 0.01f, 0.0f, FLT_MAX)) {
        std::sort(keyframes.begin(), keyframes.end(),
          [](const keyframe_t& a, const keyframe_t& b) {
            return a.time < b.time;
          });
      }

      if (gui::drag("Position", &kf.position, 0.1f)) {
        if (current_time == kf.time) {
          shape->set_position(kf.position);
        }
      }

      if (gui::drag("Size", &kf.size, 0.1f)) {
        if (current_time == kf.time) {
          shape->children[0].set_size(kf.size);
        }
      }

      fan::vec3 angle_deg = fan::math::degrees(kf.angle);
      if (gui::drag("Angle", &angle_deg, 1.0f)) {
        kf.angle = fan::math::radians(angle_deg);
        if (current_time == kf.time) {
          shape->children[0].set_angle(kf.angle);
        }
      }

      int rot_pos_idx = static_cast<int>(kf.rotation_pos);
      const char* rot_names[] = {
        "Center (0, 0)",
        "Top Left (-X, -Y)", "Top Middle (0, -Y)", "Top Right (+X, -Y)",
        "Middle Left (-X, 0)", "Middle Right (+X, 0)",
        "Bottom Left (-X, +Y)", "Bottom Middle (0, +Y)", "Bottom Right (+X, +Y)"
      };
      if (gui::combo("Rotation Position", &rot_pos_idx, rot_names, 9)) {
        kf.rotation_pos = static_cast<rotation_position_e>(rot_pos_idx);
        if (current_time == kf.time) {
          apply_to_shape(shape);
        }
      }
    }

    gui::separator();
    int global_rot_pos = static_cast<int>(current_rotation_pos);
    const char* rot_names[] = {
      "Center (0, 0)",
      "Top Left (-X, -Y)", "Top Middle (0, -Y)", "Top Right (+X, -Y)",
      "Middle Left (-X, 0)", "Middle Right (+X, 0)",
      "Bottom Left (-X, +Y)", "Bottom Middle (0, +Y)", "Bottom Right (+X, +Y)"
    };
    if (gui::combo("Default Rotation Position", &global_rot_pos, rot_names, 9)) {
      current_rotation_pos = static_cast<rotation_position_e>(global_rot_pos);
    }
  }

  std::vector<keyframe_t> keyframes;
  std::string name = "Animation";
  f32_t current_time = 0.0f;
  bool is_playing = false;
  bool loop = true;
  bool slerp_to_first = true;
  f32_t slerp_duration = 0.5f;
  bool auto_increment_time = true;
  f32_t time_increment = 0.5f;
  int selected_keyframe = -1;
  rotation_position_e current_rotation_pos = rotation_position_e::center;
  shapes_t::global_t* owner_shape = nullptr;
};


  std::unordered_map<shapes_t::global_t*, shape_keyframe_animation_t> shape_sprite_sheets;

  fan::graphics::shape_t xy_lines[2];
  fan::vec2 texturepack_size{};
  fan::vec2 texturepack_single_image_size{};
  std::vector<texturepack_image_t> texturepack_images;
  fan::graphics::render_view_t render_view;
  fan::graphics::gui::content_browser_t content_browser{false};
  shapes_t::global_t* current_shape = nullptr;
  shape_list_t shape_list;
  f32_t current_z = 1;
  uint32_t current_id = 0;
  bool render_content_browser = true;
  std::function<void()> close_cb = []() {};
  fan::vec2 drag_start;
  fan::graphics::shape_t drag_select;
  struct {
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 start_pos = 0;
    fan::vec2 offset = 0;
    bool editor_hovered = false;
  } viewport_settings;
  bool shapes_window_hovered = false;
  fan::graphics::gui::sprite_animations_t animations_application;
  fan::vec2 editor_pos = 0;
  fan::graphics::sprite_t background;
  std::vector<fan::graphics::shape_t> copy_buffer;
  std::string previous_filename;
  fan::graphics::engine_t::keys_handle_t key_handle;
  fan::graphics::engine_t::mouse_move_handle_t mouse_move_handle;
  fan::graphics::engine_t::buttons_handle_t button_handle;
};