struct fgm_t {
  fgm_t() {

  }

  static constexpr f32_t scroll_speed = 1.2;

  loco_t::shape_t xy_lines[2];

  inline static fan::graphics::file_open_dialog_t open_tp_dialog;


  fan::vec2 texturepack_size{};
  fan::vec2 texturepack_single_image_size{};
  struct texturepack_image_t {
    loco_t::image_t image;
    fan::vec2 uv0;// top left
    fan::vec2 uv1;// bottom right
    std::wstring image_name;
    f32_t aspect_ratio;
  };
  std::vector<texturepack_image_t> texturepack_images;
  void open_texturepack(const std::string& path) {
    gloco->texture_pack.open_compiled(path);
    texturepack_images.clear();
    texturepack_images.reserve(gloco->texture_pack.size());

    // loaded texturepack
    gloco->texture_pack.iterate_loaded_images([this](auto& image) {
      texturepack_image_t tp_image;
      tp_image.image = gloco->texture_pack.get_pixel_data(image.unique_id).image;

      auto& img_data = gloco->image_get_data(gloco->texture_pack.get_pixel_data(image.unique_id).image);
      fan::vec2 size = img_data.size;

      tp_image.uv0 = fan::vec2(image.position) / size;
      tp_image.uv1 = fan::vec2(tp_image.uv0) + fan::vec2(image.size) / size;
      tp_image.image_name = { image.name.begin(), image.name.end() };
      tp_image.aspect_ratio = (f32_t)image.size.x / image.size.y;

      texturepack_images.push_back(tp_image);
      texturepack_size = texturepack_size.max(fan::vec2(size));
      texturepack_single_image_size = texturepack_single_image_size.max(fan::vec2(image.size));
      });
  }

  void open(const std::string& texturepack_name, const std::wstring& asset_path) {

    content_browser.init(asset_path);
    content_browser.current_view_mode = fan::graphics::gui::content_browser_t::view_mode_large_thumbnails;

    render_view.camera = gloco->open_camera(
      fan::vec2(0, 1),
      fan::vec2(0, 1)
    );

    render_view.viewport = gloco->open_viewport(
      fan::vec2(0),
      fan::vec2(1)
    );

    // TODO leak
    auto transparent_texture = gloco->create_transparent_texture();

    background = fan::graphics::sprite_t{ {
      .render_view = &render_view,
      .position = 0,
      .size = 0,
      .image = transparent_texture,
    } };

    open_texturepack(texturepack_name);

    gloco->window.add_keys_callback([this](const auto& d) {
      if (d.state != fan::keyboard_state::press) {
        return;
      }
      if (ImGui::IsAnyItemActive()) {
        return;
      }

      switch (d.key) {
      case fan::key_r: {
        erase_current();
        break;
      }
      }
      });
    gloco->input_action.add_keycombo({ fan::input::key_left_control, fan::input::key_space }, "toggle_content_browser");
    gloco->input_action.add_keycombo({ fan::input::key_left_control, fan::input::key_f }, "set_windowed_fullscreen");

    gloco->window.add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / viewport_settings.zoom;
        auto camera_pos = viewport_settings.pos - move_off;
        gloco->camera_set_position(render_view.camera, camera_pos);
        //  background.set_position(fan::vec2(editor_pos) + camera_pos);
          /*fan::vec2 half_ground_size = ground_size * 0.5f;
          fan::vec2 top_left_world = camera_pos - half_ground_size;
          fan::vec2 tc_position = top_left_world / 64.f;*/
          //    background.set_tc_position(tc_position);
      }
      });

    gloco->window.add_buttons_callback([this](const auto& d) {

      f32_t old_zoom = viewport_settings.zoom;

      switch (d.button) {
      case fan::mouse_middle: {
        viewport_settings.move = (bool)d.state;
        fan::vec2 old_pos = viewport_settings.pos;
        viewport_settings.offset = gloco->get_mouse_position();
        viewport_settings.pos = gloco->camera_get_position(render_view.camera);
        break;
      }
      case fan::mouse_scroll_up: {
        if (viewport_settings.editor_hovered) {
          viewport_settings.zoom *= scroll_speed;
        }
        return;
      }
      case fan::mouse_scroll_down: {
        if (viewport_settings.editor_hovered) {
          viewport_settings.zoom /= scroll_speed;
        }
        return;
      }
      }
      });
    xy_lines[0] = fan::graphics::line_t{ {
      .render_view = &render_view,
      .src = fan::vec3(-0xffffff, 0, 0x1fff),
      .dst = fan::vec2(0xffffff, 0),
      .color = fan::colors::red / 2
    } };

    xy_lines[1] = fan::graphics::line_t{ {
        .render_view = &render_view,
        .src = fan::vec3(0, -0xffffff, 0x1fff),
        .dst = fan::vec2(0, 0xffffff),
        .color = fan::colors::green / 2
    } };

    drag_select = fan::graphics::rectangle_t{ {
        .render_view = &render_view,
        .position = fan::vec3(0, 0, 0xffff - 0xff),
        .size = 0,
        .color = fan::color::hex(0x3eb9ff44),
        .blending = true
    } };

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

  static constexpr auto editor_str = "Editor";
  static constexpr auto create_str = "Create";
  static constexpr auto properties_str = "Properties";

  static constexpr auto max_depth = 0xff;
  static constexpr int max_path_input = 40;

  static constexpr int max_id_input = 20;
  static constexpr fan::vec2 default_button_size{ 100, 30 };

  std::string previous_file_name;

  struct shapes_t {
    struct global_t : fan::graphics::vfi_root_t, fan::graphics::gui::imgui_element_t {

      global_t() = default;

      uint16_t shape_type = 0;

      template <typename T>
      global_t(uint16_t shape_type, fgm_t* fgm, const T& obj, bool shape_add = true) : fan::graphics::gui::imgui_element_t() {
        T temp = obj;
        this->shape_type = shape_type;
        typename loco_t::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += shape_add ? fgm->current_z++ : 0;
        vfip.shape.rectangle->size = temp.get_size();
        vfip.shape.rectangle->angle = 0;
        vfip.shape.rectangle->rotation_point = 0;
        vfip.shape.rectangle->camera = fgm->render_view.camera;
        vfip.shape.rectangle->viewport = fgm->render_view.viewport;
        vfip.mouse_button_cb = [fgm, this](const auto& d) -> int {
          fgm->event_type = event_type_e::move;
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

      // global data
      std::string id;
      uint32_t group_id = 0;
    };
  };
  //
#include <fan/graphics/gui/fgm/common.h>

//#define bcontainer_set_StoreFormat 1
//#define BLL_set_CPP_CopyAtPointerChange 1
#define BLL_set_AreWeInsideStruct 1
#include <fan/fan_bll_preset.h>
#define BLL_set_prefix shape_list
#define BLL_set_type_node uint32_t
#define BLL_set_NodeDataType shapes_t::global_t*
#define BLL_set_Link 1
#include <BLL/BLL.h>


  enum class event_type_e {
    none,
    add,
    remove,
    move,
    resize
  };


#define make_line(T, prop) \
  { \
    T v = shape->CONCAT(get_, prop)(); \
    ImGui::Indent();\
    if (gui::drag_float(STRINGIFY_DEFINE(prop), &v, 0.1, 0, FLT_MAX, "%.3f", gui::slider_flags_always_clamp)) { \
          shape->CONCAT(set_, prop)(v); \
    }\
    ImGui::Unindent(); \
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

    std::string shape_str = std::string("Shape name:") + gloco->shape_names[shape->children[0].get_shape_type()];
    ImGui::Text("%s", shape_str.c_str());

    {
      fan::vec3 v = shape->get_position();
      gui::indent();

      bool changed = false;


      gui::push_item_width(gui::calc_item_width() / 3.f);
      changed |= gui::drag_float("##position_x", &v.x, 0.1f, 0.0f, std::numeric_limits<float>::max(), "%.3f", gui::slider_flags_always_clamp);
      gui::same_line();
      changed |= gui::drag_float("##position_y", &v.y, 0.1f, 0.0f, std::numeric_limits<float>::max(), "%.3f", gui::slider_flags_always_clamp);
      gui::same_line();
      int z = static_cast<int>(v.z);
      if (gui::drag_int("##position_z", &z, 1.0f, 0, std::numeric_limits<int>::max(), "%d", gui::slider_flags_always_clamp)) {
        v.z = static_cast<float>(z);
        changed = true;
      }
      gui::same_line();
      gui::text("Position");

      gui::pop_item_width();

      if (changed) {
        v.z = (int)v.z; // if user types manually
        shape->set_position(v);
      }

      gui::unindent();
    }

    make_line(fan::vec2, size);
    fan::color c = shape->get_color();

    if (ImGui::ColorEdit4("color", c.data())) {
      shape->set_color(c);
    }

    {
      ImGui::Text("angle");
      ImGui::SameLine();
      fan::vec3 angle = shape->children[0].get_angle();
      angle.x = fan::math::degrees(angle.x);
      angle.y = fan::math::degrees(angle.y);
      angle.z = fan::math::degrees(angle.z);
      ImGui::SliderFloat3("##hidden_label1" "angle", angle.data(), 0, 360);
      angle = fan::math::radians(angle);
      shape->children[0].set_angle(angle);

    }

    {
      std::string& id = current_shape->id;
      std::string str = id;
      str.resize(max_id_input);
      ImGui::Text("id");
      ImGui::SameLine();
      if (ImGui::InputText("##hidden_label0" "id", str.data(), str.size())) {
        \
          if (ImGui::IsItemDeactivatedAfterEdit()) {
            std::string new_id = str.substr(0, std::strlen(str.c_str()));
            if (!id_exists(new_id)) {
              id = new_id;
            }
          }
      }
    }
    {
      std::string id = std::to_string(current_shape->group_id);
      std::string str = id;
      str.resize(max_id_input);
      ImGui::Text("group id");
      ImGui::SameLine();
      if (ImGui::InputText("##hidden_label0" "group id", str.data(), str.size())) {
        \
          if (ImGui::IsItemDeactivatedAfterEdit()) {
            current_shape->group_id = std::stoul(str);
          }
      }
    }
    switch (shape->children[0].get_shape_type()) {
    case loco_t::shape_type_t::unlit_sprite:
    case loco_t::shape_type_t::sprite: {
      {
        auto current_image = shape->children[0].get_image();
        fan::vec2 uv0 = shape->children[0].get_tc_position(), uv1 = shape->children[0].get_tc_size();
        uv1 += uv0;
        fan::graphics::gui::image(current_image, fan::vec2(64), uv0, uv1);
        gui::receive_drag_drop_target("CONTENT_BROWSER_ITEMS", [&](const std::string& path) {
          if (current_image != gloco->default_texture) {
            gloco->image_unload(current_image);
          }
          shape->children[0].set_image(gloco->image_load((std::filesystem::path(content_browser.asset_path) / path).string()));
          shape->children[0].set_tc_position(0);
          shape->children[0].set_tc_size(1);
          });
        ImGui::SameLine();
        ImGui::Text("Base texture");
      }

      {
        auto current_image = shape->children[0].get_images()[0];
        if (current_image.iic()) {
          current_image = gloco->default_texture;
        }
        fan::vec2 uv0 = shape->children[0].get_tc_position(), uv1 = shape->children[0].get_tc_size();
        uv1 += uv0;
        fan::graphics::gui::image(current_image, fan::vec2(64), uv0, uv1);
        gui::receive_drag_drop_target("CONTENT_BROWSER_ITEMS", [&](const std::string& path) {
          if (current_image != gloco->default_texture) {
            gloco->image_unload(current_image);
          }
          shape->children[0].set_images({ gloco->image_load((std::filesystem::path(content_browser.asset_path) / path).string()) });
          shape->children[0].set_tc_position(0);
          shape->children[0].set_tc_size(1);
          });
        ImGui::SameLine();
        ImGui::Text("Normal map");
      }
      {
        auto current_image = shape->children[0].get_images()[1];
        if (current_image.iic()) {
          current_image = gloco->default_texture;
        }
        fan::vec2 uv0 = shape->children[0].get_tc_position(), uv1 = shape->children[0].get_tc_size();
        uv1 += uv0;
        fan::graphics::gui::image(current_image, fan::vec2(64), uv0, uv1);
        gui::receive_drag_drop_target("CONTENT_BROWSER_ITEMS", [&](const std::string& path) {
          if (current_image != gloco->default_texture) {
            gloco->image_unload(current_image);
          }
          auto images = shape->children[0].get_images();
          images[1] = gloco->image_load((std::filesystem::path(content_browser.asset_path) / path).string());
          shape->children[0].set_images(images);
          shape->children[0].set_tc_position(0);
          shape->children[0].set_tc_size(1);
          });
        ImGui::SameLine();
        ImGui::Text("Specular map");
      }
      {
        auto current_image = shape->children[0].get_images()[2];
        if (current_image.iic()) {
          current_image = gloco->default_texture;
        }
        fan::vec2 uv0 = shape->children[0].get_tc_position(), uv1 = shape->children[0].get_tc_size();
        uv1 += uv0;
        fan::graphics::gui::image(current_image, fan::vec2(64), uv0, uv1);
        gui::receive_drag_drop_target("CONTENT_BROWSER_ITEMS", [&](const std::string& path) {
          if (current_image != gloco->default_texture) {
            gloco->image_unload(current_image);
          }
          auto images = shape->children[0].get_images();
          images[2] = gloco->image_load((std::filesystem::path(content_browser.asset_path) / path).string());
          shape->children[0].set_images(images);
          shape->children[0].set_tc_position(0);
          shape->children[0].set_tc_size(1);
          });
        ImGui::SameLine();
        ImGui::Text("Occlusion map");
      }

      {
        int current_image_filter = gloco->image_get_settings(shape->children[0].get_image()).min_filter;

        static const char* image_filters[] = {
          "nearest", "linear"
        };
        if (ImGui::Combo("image filter", &current_image_filter, image_filters, std::size(image_filters))) {
          fan::graphics::image_load_properties_t ilp;
          ilp.min_filter = current_image_filter;
          ilp.mag_filter = current_image_filter;
          gloco->image_set_settings(shape->children[0].get_image(), ilp);
          if (shape->children[0].get_images()[0].iic() == false) {
            gloco->image_set_settings(shape->children[0].get_images()[0], ilp);
          }
        }
      }

      std::string& current = shape->children[0].get_image_data().image_path;
      std::string str = current;
      str.resize(max_path_input);
      ImGui::Text("image name");
      ImGui::SameLine();
      if (ImGui::InputText("##hidden_label4", str.data(), str.size())) {
        if (ImGui::IsItemDeactivatedAfterEdit()) {
          loco_t::texturepack_t::ti_t ti;
          if (gloco->texture_pack.qti(str.c_str(), &ti)) {
            fan::print_no_space("failed to load texture:", str);
          }
          else {
            current = str.substr(0, std::strlen(str.c_str()));
            auto& data = gloco->texture_pack.get_pixel_data(ti.unique_id);
            if (shape->children[0].get_shape_type() == loco_t::shape_type_t::sprite) {
              shape->children[0].load_tp(&ti);
            }
            else if (shape->children[0].get_shape_type() == loco_t::shape_type_t::unlit_sprite) {
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
    case loco_t::shape_type_t::sprite: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::sprite,
        this, fan::graphics::sprite_t{{
          .render_view = &render_view,
          .position = pos,
          .size = size
        }} };
      auto* ri = ((loco_t::sprite_t::ri_t*)shape_list[nr]->children[0].GetData(gloco->shaper));
      animations_application.current_animation_nr = ri->current_animation;
      animations_application.current_animation_shape_nr = ri->shape_animations;
      break;
    }
    case loco_t::shape_type_t::unlit_sprite: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::unlit_sprite,
        this, fan::graphics::unlit_sprite_t{{
          .render_view = &render_view,
          .position = pos,
          .size = size
        }} };
      break;
    }
    case loco_t::shape_type_t::rectangle: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::rectangle,
        this, fan::graphics::rectangle_t{{
          .render_view = &render_view,
          .position = pos,
          .size = size
        }} };
      break;
    }
    case loco_t::shape_type_t::light: {
      shape_list[nr] = new shapes_t::global_t{
        loco_t::shape_type_t::light,
        this, fan::graphics::light_t{{
          .render_view = &render_view,
          .position = pos,
          .size = size
        }} };
      shape_list[nr]->push_child(fan::graphics::circle_t{ {
        .render_view = &render_view,
        .position = fan::vec3(pos, current_z),
        .radius = size.x,
        .color = fan::color(1, 1, 1, 0.0),
        .blending = true
      } });
      break;
    }
    }
    return nr;
  }

  void RenderTreeWithUnifiedSelection() {
    auto it = shape_list.GetNodeFirst();
    int nodeIndex = 0; // Unique identifier for all nodes

    static int selection_mask = 0; // Mask to track selected nodes
    int node_clicked = -1;
    static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;

    while (it != shape_list.dst) {
      auto& shape_instance = shape_list[it];
      std::string nodeStr = "Node " + std::to_string(nodeIndex);

      ImGuiTreeNodeFlags node_flags = base_flags;
      const bool is_selected = (selection_mask & (1 << (intptr_t)it.NRI)) != 0;
      if (is_selected)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      // Determine if this node is selected
      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)it.NRI, node_flags, "Node %ld", (intptr_t)it.NRI);

      if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        node_clicked = (intptr_t)it.NRI;
      if (node_open) {
        // Recursively handle child nodes
        RenderChildNodes(node_clicked, shape_instance->children, selection_mask, base_flags);

        ImGui::TreePop();
      }
      it = it.Next(&shape_list);
      nodeIndex++;
    }

    // Update selection state
    // (process outside of tree loop to avoid visual inconsistencies during the clicking frame)
    if (node_clicked != -1) {
      if (ImGui::GetIO().KeyCtrl)
        selection_mask ^= (1 << node_clicked); // CTRL+click to toggle
      else
        selection_mask = (1 << node_clicked);  // Click to single-select
    }
  }

  void RenderChildNodes(int& node_clicked, std::vector<fan::graphics::vfi_root_t::child_data_t>& children, int& selection_mask, ImGuiTreeNodeFlags base_flags) {
    int child_index = 0;
    for (auto& child : children) {
      ImGuiTreeNodeFlags node_flags = base_flags;
      const bool is_selected = (selection_mask & (1 << (intptr_t)child.NRI)) != 0;
      if (is_selected)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      if (child_index + 1 >= children.size())
        node_flags |= ImGuiTreeNodeFlags_Leaf;

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)child.NRI, node_flags, "%s %u", gloco->shape_names[child.get_shape_type()], child.NRI);

      if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        node_clicked = (intptr_t)child.NRI;

      if (node_open) {
        ImGui::TreePop();
      }
      child_index++;
    }
  }

  fan::vec2 get_mouse_position() {
    auto& style = ImGui::GetStyle();
    fan::vec2 pos = fan::vec2((ImGui::GetMousePos() - viewport_settings.start_pos + style.WindowPadding) - viewport_settings.size / 2);
    pos = fan::vec2(viewport_settings.pos) + pos / viewport_settings.zoom;
    return pos;
  }

  void render() {
    using namespace fan::graphics;

    if (viewport_settings.editor_hovered && ImGui::IsMouseClicked(0) && !fan::graphics::vfi_root_t::moving_object) {
      drag_start = get_mouse_position();
    }
    else if (viewport_settings.editor_hovered && ImGui::IsMouseDown(0) && !fan::graphics::vfi_root_t::moving_object) {
      fan::vec2 size = get_mouse_position() - drag_start;
      for (auto& i : fan::graphics::vfi_root_t::selected_objects) {
        i->disable_highlight();
      }
      fan::graphics::vfi_root_t::selected_objects.clear();
      drag_select.set_position(drag_start + size / 2);
      drag_select.set_size(size / 2);
    }
    else if (ImGui::IsMouseReleased(0)) {
      bool hit_any = false;
      auto it = shape_list.GetNodeFirst();
      while (it != shape_list.dst) {
        auto& shape = shape_list[it];
        if (fan_2d::collision::rectangle::check_collision(
          drag_select.get_position(),
          drag_select.get_size(),
          shape->children[0].get_position(),
          shape->children[0].get_size()
        )) {
          if (!fan::graphics::vfi_root_t::moving_object &&
            (drag_select.get_size().x >= 1 && drag_select.get_size().y >= 1)) {
            shape->create_highlight();
            fan::graphics::vfi_root_t::selected_objects.push_back(shape);
          }
        }
        if (fan_2d::collision::rectangle::point_inside_no_rotation(
          get_mouse_position(), shape->children[0].get_position(),
          shape->children[0].get_size()
        ))
        {
          hit_any = true;
          // change sprite image in animation viewer when clicking other element
          if (shape->children[0].get_shape_type() == loco_t::shape_type_t::sprite) {
            auto* ri = ((loco_t::sprite_t::ri_t*)shape->children[0].GetData(gloco->shaper));
            animations_application.current_animation_nr = ri->current_animation;
            animations_application.current_animation_shape_nr = ri->shape_animations;
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

    if (gloco->input_action.is_active("set_windowed_fullscreen")) {
      gloco->window.set_windowed_fullscreen();
    }

    if (gloco->input_action.is_active("toggle_content_browser")) {
      render_content_browser = !render_content_browser;
    }

    if (render_content_browser) {
      content_browser.render();
    }

    if (ImGui::Begin(editor_str, nullptr, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoBackground)) {
      fan::vec2 window_size = gloco->window.get_size();
      fan::vec2 viewport_size = ImGui::GetWindowSize();
      fan::vec2 ratio = viewport_size / viewport_size.max();
      fan::vec2 s = viewport_size;

      editor_pos = ImGui::GetWindowPos();

      f32_t zoom = viewport_settings.zoom;
      fan::vec2 ground_size = viewport_size * (1.0f / zoom);
      fan::vec2 camera_pos = gloco->camera_get_position(render_view.camera);

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

      viewport_settings.editor_hovered = ImGui::IsWindowHovered();
      fan::graphics::vfi_root_t::g_ignore_mouse = !viewport_settings.editor_hovered;
      //fan::print(viewport_settings.editor_hovered);

      auto& style = ImGui::GetStyle();
      fan::vec2 frame_padding = style.FramePadding;
      fan::vec2 viewport_pos = fan::vec2(0, frame_padding.y * 2);

      ImVec2 vMin = ImGui::GetWindowContentRegionMin();
      ImVec2 vMax = ImGui::GetWindowContentRegionMax();

      vMin.x += ImGui::GetWindowPos().x;
      vMin.y += ImGui::GetWindowPos().y;
      vMax.x += ImGui::GetWindowPos().x;
      vMax.y += ImGui::GetWindowPos().y;

      viewport_size = vMax - vMin + style.WindowPadding * 2;

      gloco->viewport_set(
        render_view.viewport,
        vMin - style.WindowPadding,
        viewport_size
      );

      gloco->camera_set_ortho(
        render_view.camera,
        fan::vec2(-viewport_size.x / 2, viewport_size.x / 2) / viewport_settings.zoom,
        fan::vec2(-viewport_size.y / 2, viewport_size.y / 2) / viewport_settings.zoom
      );


      viewport_settings.size = viewport_size;
      viewport_settings.start_pos = vMin;

      {
        std::string str = fan::to_string(viewport_settings.zoom * 100);
        str += " %";
        fan::graphics::gui::text_bottom_right(str, 1);
      }

      {
        fan::vec2 cursor_pos = ((ImGui::GetMousePos() - viewport_settings.start_pos + style.WindowPadding) - viewport_settings.size / 2);
        std::string cursor_pos_str = cursor_pos.to_string();
        std::string  str = cursor_pos_str.substr(1, cursor_pos_str.size() - 2);
        fan::graphics::gui::text_bottom_right(str.c_str(), 0);
      }

      ImGui::SetCursorPos(ImGui::GetCursorStartPos());


      content_browser.receive_drag_drop_target([&](const std::filesystem::path& fs) {

        auto file = fs.string();
        auto extension = fan::io::file::extension(file);
        if (extension == ".json") {
          fin(file);
        }
        else {
          auto image = gloco->image_load((fs).string());
          fan::vec2 initial_size = 128.f;
          fan::vec2 original_size = gloco->image_get_data(image).size;
          initial_size.x *= (original_size.x / original_size.y);
          auto nr = push_shape(loco_t::shape_type_t::sprite, get_mouse_position(), initial_size);
          shape_list[nr]->children[0].set_image(image);
        }
        });

      gui::receive_drag_drop_target("FGM_TEXTUREPACK_DROP", [&](const std::string& path) {
        loco_t::texturepack_t::ti_t ti;
        if (gloco->texture_pack.qti(path, &ti)) {
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

          auto nr = push_shape(loco_t::shape_type_t::sprite, get_mouse_position(), initial_size);
          auto& node = shape_list[nr];
          node->children[0].load_tp(&ti);
          node->children[0].get_image_data().image_path = path;
        }
        });
    }

    // keybinds
    if (fan::window::is_key_down(fan::key_left_control)) {
      if (fan::window::is_key_pressed(fan::key_d)) {
        for (auto& i : fan::graphics::vfi_root_t::selected_objects) {
          for (auto& child : i->children) {
            auto it = shape_list.NewNodeLast();
            auto& node = shape_list[it];
            node = new shapes_t::global_t{
            child.get_shape_type(),
            this, child };
          }
        }
        fan::graphics::vfi_root_t::selected_objects.clear();
      }
    }

    static fan::graphics::file_save_dialog_t save_file_dialog;
    static fan::graphics::file_open_dialog_t open_file_dialog;
    static std::string fn;

    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open..", "Ctrl+O")) {

          open_file_dialog.load("json;fmm", &fn);
        }
        if (ImGui::MenuItem("Save", "Ctrl+S")) {
          fout(previous_file_name);
        }
        if (ImGui::MenuItem("Save as", "Ctrl+Shift+S")) {
          save_file_dialog.save("json;fmm", &fn);
        }
        if (ImGui::MenuItem("Quit")) {
          auto it = shape_list.GetNodeFirst();
          while (it != shape_list.dst) {
            delete shape_list[it];
            it = it.Next(&shape_list);
          }
          shape_list.Clear();

          close_cb();
          ImGui::End();
        }
        ImGui::EndMenu();
      }
      ImGui::EndMenuBar();
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
      ImGui::End();
      return;
    }
    if (save_file_dialog.is_finished()) {
      if (fn.size() != 0) {
        fout(fn);
      }
      save_file_dialog.finished = false;
    }

    ImGui::End();

    if (ImGui::Begin("Settings")) {
      if (ImGui::ColorEdit3("background", gloco->clear_color.data())) {

      }
      if (ImGui::ColorEdit3("ambient", gloco->lighting.ambient.data())) {

      }
      if (gui::drag_float("grid snap", &fan::graphics::vfi_root_t::snap, 1, 0, FLT_MAX, "%.3f", gui::slider_flags_always_clamp)) {

      }
    }
    ImGui::End();

    if (ImGui::Begin(properties_str, nullptr)) {
      if (current_shape != nullptr) {
        open_properties(current_shape, editor_size);

        gui::new_line();
        gui::begin_child("properties_animations", 0, 1);
        gui::text("Animations");
        auto& shape = current_shape->children[0];
        loco_t::animation_shape_nr_t* shape_animation_nr = 0;
        if (shape.get_shape_type() == loco_t::shape_type_t::sprite) {
          auto* ri = ((loco_t::sprite_t::ri_t*)shape.GetData(gloco->shaper));
          shape_animation_nr = &ri->shape_animations;
          if (animations_application.current_animation_nr) {
            auto& anim = gloco->get_sprite_sheet_animation(animations_application.current_animation_nr);
          }
        }
        if (shape_animation_nr == nullptr) {
          goto g_end_animations;
        }
        {
          bool animation_changed = animations_application.render("CONTENT_BROWSER_ITEMS", *shape_animation_nr);
          if (shape.get_shape_type() == loco_t::shape_type_t::sprite && animations_application.current_animation_nr) {
            auto* ri = ((loco_t::sprite_t::ri_t*)shape.GetData(gloco->shaper));
            if (animations_application.current_animation_nr && animations_application.current_animation_shape_nr == *shape_animation_nr) {
              ri->current_animation = animations_application.current_animation_nr;
            }
          }
          if (*shape_animation_nr && animations_application.current_animation_shape_nr == *shape_animation_nr && (animations_application.toggle_play_animation || animations_application.play_animation || animation_changed)) {
            if (shape.get_shape_type() == loco_t::shape_type_t::sprite) {

              auto& anim = gloco->get_sprite_sheet_animation(animations_application.current_animation_nr);
              if (animation_changed && animations_application.current_animation_nr && animations_application.play_animation) {
                if (gloco->is_image_valid(shape.get_image()) == false) {
                  shape.set_tc_position(0);
                  shape.set_tc_size(1);
                }
              }
              else if (anim.selected_frames.size()) {
                //shape.set_image(anim.sprite_sheet);
              }

              auto& current_shape_anim = shape.get_sprite_sheet_animation();
              if (animations_application.play_animation && (animations_application.toggle_play_animation || animation_changed) &&
                current_shape_anim.selected_frames.size()) {
                shape.start_sprite_sheet_animation();
                //shape.set_sprite_sheet_frames(anim.hframes, anim.vframes);
              }

              if (animations_application.play_animation && (animations_application.toggle_play_animation || animation_changed) &&
                current_shape_anim.selected_frames.size()) {
                shape.set_sprite_sheet_fps(anim.fps);
              }
              if (animations_application.play_animation && animations_application.toggle_play_animation && !animations_application.play_animation) {
                shape.set_sprite_sheet_fps(0.0001);
              }
            }
          }
        }
      g_end_animations:
        gui::end_child();
      }
    }
    ImGui::End();


    if (ImGui::Begin(create_str, nullptr)) {
      if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("ContextMenu");
      }
      if (ImGui::BeginPopup("ContextMenu")) {
        if (ImGui::BeginMenu("Create")) {
          if (ImGui::BeginMenu("Shapes")) {
            if (ImGui::MenuItem("Sprite")) {
              push_shape(loco_t::shape_type_t::sprite, 0);
            }
            ImGui::EndMenu();
          }
          if (ImGui::BeginMenu("Lights")) {
            if (ImGui::MenuItem("Circle")) {
              push_shape(loco_t::shape_type_t::light, 0);
            }
            ImGui::EndMenu();
          }

          ImGui::EndMenu();
        }

        ImGui::EndPopup();
      }
      RenderTreeWithUnifiedSelection();
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

    ImGui::End();
  }

  void fout(std::string filename) {
    if (!filename.ends_with(".json")) {
      filename += ".json";
    }
    previous_file_name = filename;

    fan::json ostr;
    ostr["version"] = current_version;
    if (gloco->lighting.ambient != loco_t::lighting_t().ambient) {
      ostr["lighting.ambient"] = gloco->lighting.ambient;
    }
    if (gloco->clear_color != fan::colors::black) {
      ostr["clear_color"] = gloco->clear_color;
    }
    {
      auto animations_json = gloco->sprite_sheet_serialize();
      if (animations_json.empty() == false) {
        ostr.update(animations_json, true);
      }
    }
    fan::json shapes = fan::json::array();
    auto it = shape_list.GetNodeFirst();
    while (it != shape_list.dst) {
      auto& shape_instance = shape_list[it];
      auto& shape = shape_instance->children[0];

      fan::json shape_json;

      shapes_t::global_t defaults;
      switch (shape_instance->shape_type) {
      case loco_t::shape_type_t::sprite: {
        fan::graphics::shape_serialize(shape, &shape_json);
        break;
      }
      case loco_t::shape_type_t::unlit_sprite: {
        fan::graphics::shape_serialize(shape, &shape_json);
        break;
      }
      case loco_t::shape_type_t::rectangle: {
        fan::graphics::shape_serialize(shape, &shape_json);
        break;
      }
      case loco_t::shape_type_t::light: {
        fan::graphics::shape_serialize(shape, &shape_json);
        break;
      }
      default: {
        fan::print("unimplemented shape type");
        break;
      }
      }
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
    fan::io::file::write(filename, ostr.dump(2), std::ios_base::binary);
    fan::print("file saved to:" + filename);
  }

  void load_tp(fgm_t::shape_list_NodeData_t& node) {
    loco_t::texturepack_t::ti_t ti;
    if (gloco->texture_pack.qti(node->children[0].get_image_data().image_path, &ti)) {
      fan::print_no_space("non texturepack texture or failed to load texture:", node->children[0].get_image_data().image_path);
    }
    else {
      auto& data = gloco->texture_pack.get_pixel_data(ti.unique_id);
      node->children[0].load_tp(&ti);
    }
  }

  /*
  header - 4 byte
  shape_type - 2 byte
  struct size - x byte
  data{
    ...
  }
  */
  void fin(const std::string& filename, const std::source_location& callers_path = std::source_location::current()) {

    previous_file_name = filename;

    std::string in;
    fan::io::file::read(fan::io::file::find_relative_path(filename, callers_path), &in);
    fan::json json_in = fan::json::parse(in);
    auto version = json_in["version"].get<decltype(current_version)>();
    if (version != current_version) {
      fan::print("invalid file version, file:", version, "current:", current_version);
      return;
    }
    if (json_in.contains("lighting.ambient")) {
      gloco->lighting.ambient = json_in["lighting.ambient"];
    }
    if (json_in.contains("clear_color")) {
      gloco->clear_color = json_in["clear_color"];
    }
    if (json_in.contains("animations")) {
      gloco->parse_animations(json_in);
    }
    fan::graphics::shape_deserialize_t iterator;
    loco_t::shape_t shape;
    int i = 0;
    current_z = 0;
    std::string shapes = "shapes";
    if (json_in.contains("tiles")) {
      shapes = "tiles";
    }
    while (iterator.iterate(json_in[shapes], &shape)) {
      const auto& shape_json = *(iterator.data.it - 1);

      shape.set_camera(render_view.camera);
      shape.set_viewport(render_view.viewport);
      current_z = std::max(current_z, shape.get_position().z);
      auto it = shape_list.NewNodeLast();
      auto& node = shape_list[it];
      switch (shape.get_shape_type()) {
      case loco_t::shape_type_t::sprite: {
        node = new fgm_t::shapes_t::global_t(
          loco_t::shape_type_t::sprite,
          this,
          shape,
          false
        );
        load_tp(node);
        node->children[0].get_image_data().image_path = shape.get_image_data().image_path;
        break;
      }
      case loco_t::shape_type_t::unlit_sprite: {
        node = new fgm_t::shapes_t::global_t(
          loco_t::shape_type_t::unlit_sprite,
          this,
          shape,
          false
        );
        load_tp(node);
        node->children[0].get_image_data().image_path = shape.get_image_data().image_path;
        break;
      }
      case loco_t::shape_type_t::rectangle: {
        node = new fgm_t::shapes_t::global_t(
          loco_t::shape_type_t::rectangle,
          this,
          shape,
          false
        );
        break;
      }
      case loco_t::shape_type_t::light: {
        node = new fgm_t::shapes_t::global_t(
          loco_t::shape_type_t::light,
          this,
          shape,
          false
        );
        node->push_child(fan::graphics::circle_t{ {
          .render_view = &render_view,
          .position = shape.get_position(),
          .radius = shape.get_size().x,
          .color = shape.get_color(),
          .blending = true
        } });
        break;
      }
      default: {
        fan::print("unimplemented shape type");
        break;
      }
      }
      if (shape_json.contains("id")) {
        node->id = shape_json["id"].get<std::string>();
      }
      if (shape_json.contains("group_id")) {
        node->group_id = shape_json["group_id"].get<uint32_t>();
      }
    }
    ++current_z;
  }

  void invalidate_current() {
    current_shape = nullptr;
    selected_shape_type = loco_t::shape_type_t::invalid;
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

  fan::graphics::render_view_t render_view;

  fan::graphics::gui::content_browser_t content_browser{ false };

  event_type_e event_type = event_type_e::none;
  uint16_t selected_shape_type = loco_t::shape_type_t::invalid;
  shapes_t::global_t* current_shape = nullptr;
  shape_list_t shape_list;

  f32_t current_z = 1;
  uint32_t current_id = 0;

  bool render_content_browser = true;

  std::function<void()> close_cb = [] {};

  fan::vec2 drag_start;
  loco_t::shape_t drag_select;

  struct {
    f32_t zoom = 1;
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 start_pos = 0;
    fan::vec2 offset = 0;
    bool editor_hovered = false;
  }viewport_settings;


  //

  fan::graphics::gui::sprite_animations_t animations_application;

  fan::vec2 editor_pos = 0;
  fan::graphics::sprite_t background;
  std::vector<loco_t::shape_t> copy_buffer;
};
