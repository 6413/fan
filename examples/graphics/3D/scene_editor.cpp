#include <fan/utility.h>

#include <string>
#include <unordered_map>
#include <type_traits>
#include <memory>
#include <filesystem>

#include <fan/types/dme.h>

import fan;

using namespace fan::graphics;

void draw_world_orientation_axes(const fan::vec2& origin, const fan::vec2& window_pos, const fan::vec2& window_size, const fan::vec2& viewport_size, const fan::mat4& view, float size = 1.0f) {
  fan::color red = fan::colors::red;
  fan::color green = fan::colors::green;
  fan::color blue = fan::colors::blue;

  fan::vec3 x_axis(view[0][0], view[1][0], view[2][0]);
  fan::vec3 y_axis(view[0][1], view[1][1], view[2][1]);
  fan::vec3 z_axis(view[0][2], view[1][2], view[2][2]);

  x_axis = x_axis.normalize();
  y_axis = y_axis.normalize();
  z_axis = z_axis.normalize();

  auto* draw_list = gui::get_window_draw_list();

  fan::color bgColor = fan::color::rgb(10, 10, 10, 200);
  fan::vec2 bgTopLeft(window_pos - window_size / 2);
  fan::vec2 bgBottomRight(window_pos + window_size / 2);
    
  // Draw background rectangle
  draw_list->AddRectFilled(
    bgTopLeft, 
    bgBottomRight, 
    bgColor,  // Color
    5.0f     // Optional: corner rounding radius
  );

  auto ProjectAxis = [&](const fan::vec3& axis, float length) {
    return fan::vec2(
      origin.x + axis.x * length,
      origin.y - axis.y * length 
    );
  };

  fan::vec2 x_end = ProjectAxis(x_axis, size * 50);
  fan::vec2 y_end = ProjectAxis(y_axis, size * 50);
  fan::vec2 z_end = ProjectAxis(z_axis, size * 50);

  draw_list->AddLine(origin, x_end, red, 2.0f);
  draw_list->AddLine(origin, y_end, green, 2.0f);
  draw_list->AddLine(origin, z_end, blue, 2.0f);

  draw_list->AddText(
    fan::vec2(x_end.x + 5, x_end.y),
    red, "X"
  );
  draw_list->AddText(
    fan::vec2(y_end.x, y_end.y - 5),
    green, "Y"
  );
  draw_list->AddText(
    fan::vec2(z_end.x + 5, z_end.y),
    blue, "Z"
  );
}
struct pile_t {
  engine_t loco;

  static constexpr fan::vec2 initial_render_view_size{ 0.7, 1 };

  struct editor_t {
    struct flags_e {
      enum {
        hovered = 1 << 0,
      };
    };

    engine_t& get_engine() {
      return OFFSETLESS(this, pile_t, editor)->loco;
    }

    bool begin_render_common(const char* window_name, uint16_t& flags, uint16_t custom_flags = 0) {
      fan::vec2 window_size = get_engine().window.get_size();
      bool ret = gui::begin(window_name, 0, custom_flags);
      if (ret) {
        flags &= ~flags_e::hovered;
        flags |= (uint16_t)gui::is_window_hovered(gui::hovered_flags_allow_when_blocked_by_popup |
        gui::hovered_flags_allow_when_blocked_by_active_item);
      }
      return ret;
    }

    editor_t() :
      camera_nr(get_engine().open_camera_perspective()),
      viewport_nr(get_engine().open_viewport(0, get_engine().window.get_size())),
      camera(get_engine().camera_get(camera_nr)), // kinda waste, but shorter code
      viewport(get_engine().viewport_get(viewport_nr)) // kinda waste, but shorter code
    {
      camera.position = { 0.04, 0.71, 1.35 };
      camera.yaw = -180;
      camera.pitch = -5.9;
      gui::get_io().ConfigWindowsMoveFromTitleBarOnly = true;

      content_browser.init("");
      content_browser.current_view_mode = gui::content_browser_t::view_mode_large_thumbnails;

      static auto mcb = get_engine().window.add_mouse_motion_callback([&](const auto& d) {
        if (d.motion != 0 && cursor_mode == 0) {
          //fan::print(d.motion);
          camera.rotate_camera(d.motion);
        }
        });

      static auto bcb = get_engine().window.add_buttons_callback([&](const auto& d) {
        if (render_view.hovered() == false && cursor_mode == 1) {
          return;
        }
        switch (d.button) {
        case fan::mouse_right: {
          if (d.state == fan::mouse_state::press) {
            cursor_mode = 0;
          }
          else {
            cursor_mode = 1;
          }
          get_engine().window.set_cursor(cursor_mode);
          render_view.toggle_focus = true;
          break;
        }
        }
      });
    }

    struct render_view_t {
      editor_t& get_editor() {
        return *OFFSETLESS(this, editor_t, render_view);
      }
      loco_t& get_engine() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void begin_render() {
        gui::push_style_var(gui::style_var_window_padding, {0, 0});
        gui::push_style_color(gui::col_window_bg, {0, 0, 0, 0});
        if (toggle_focus) {
          gui::set_next_window_focus();
          toggle_focus = false;
        }
        if (get_editor().begin_render_common("Render view", flags)) {
          fan::graphics::gui::set_viewport(get_editor().viewport_nr);
          if (gui::is_window_focused()) {
            get_engine().camera_move(get_editor().camera, get_editor().camera_properties.speed, get_editor().camera_properties.friction);
          }

          // drag and drop
          get_editor().content_browser.receive_drag_drop_target([this](const std::filesystem::path& path) {
            get_editor().entities.add_entity(path.string());
          });
          fan::vec2 image_size = 64;
          gui::set_cursor_pos({0, image_size.y});
          {
            gui::indent(image_size.x / 2);
            gui::style_scope_t s;
            s.var(gui::style_var_frame_rounding, 20).var(gui::style_var_frame_border_size, 1);
            s.color(gui::col_button, {0, 0, 0, 0}).color(gui::col_button_hovered, {0.4, 0.4, 0.4, 0.4}).color(gui::col_button_active, {0, 0, 0, 0});
            if (fan::graphics::gui::toggle_image_button("##Camera Properties", get_editor().icon_video_camera, image_size, &toggle_camera_properties)) {
              get_editor().camera_properties.render_this = !get_editor().camera_properties.render_this;
            }
            if (fan::graphics::gui::toggle_image_button("##Lighting Properties", get_editor().icon_lightbulb, image_size, &toggle_lighting_properties)) {
              get_editor().lighting_properties.render_this = !get_editor().lighting_properties.render_this;
            }
            if (fan::graphics::gui::toggle_image_button("##Skeleton Properties", get_editor().icon_skeleton, image_size, &toggle_skeleton_properties)) {
              get_editor().skeleton_properties.render_this = !get_editor().skeleton_properties.render_this;
            }
            gui::unindent(image_size.x / 2);
          }
          viewport_position = gui::get_window_content_region_min() + gui::get_window_pos();
          viewport_size = gui::get_window_content_region_max() + gui::get_window_pos() - viewport_position;

          gui::gizmo::begin_frame();
          gui::gizmo::set_orthographic(false);
          gui::gizmo::set_drawlist();

          auto& rv = get_editor().render_view;
          gui::gizmo::set_rect(rv.viewport_position, rv.viewport_size);

          auto& cam = get_editor().camera;
          auto& ent = get_editor().entities;
          auto* transform = ent.gizmo_model && ent.property_type == entities_t::property_types_e::none ? &ent.gizmo_model->user_transform :
                            ent.selected_bone ? &ent.selected_bone->user_transform : nullptr;

          if (transform) {
            gui::gizmo::manipulate(cam.view, cam.projection, gui::gizmo::operation::translate, gui::gizmo::mode::world, *transform);
            gui::gizmo::manipulate(cam.view, cam.projection, gui::gizmo::operation::rotate, gui::gizmo::mode::world, *transform);
          }

          {
            fan::vec2 view_pos = {viewport_position.x + viewport_size.x - 100, viewport_position.y + 100};
            fan::vec2 view_size = 140;
            gui::gizmo::set_rect(view_pos, view_size);
            draw_world_orientation_axes(view_pos, view_pos, view_size, rv.viewport_size, cam.view, 1.0f);
          }
        }
      }
      void end_render() {
        gui::end();
        gui::pop_style_color();
        gui::pop_style_var();
      }
      bool hovered() const {
        return flags & flags_e::hovered;
      }
      bool toggle_camera_properties = 0;
      bool toggle_lighting_properties = 0;
      bool toggle_skeleton_properties = 0;
      uint16_t flags = 0;
      bool toggle_focus = false;
      fan::vec2 viewport_position = 0;
      fan::vec2 viewport_size = 0;
    }render_view;
    // list of entities
    struct entities_t {
      editor_t& get_editor() {
        return *OFFSETLESS(this, editor_t, entities);
      }
      engine_t& get_engine() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void add_entity(const std::string_view path) {
        entity_t entity;
        std::filesystem::path fs_path(path);
        fan::graphics::model_t::properties_t model_properties;
        model_properties.path = path;
        model_properties.texture_path = (fs_path.parent_path() / "textures").string();
        model_properties.use_cpu = 0;
        model_properties.camera = get_editor().camera_nr;
        model_properties.viewport = get_editor().viewport_nr;//////
        entity.model = std::make_unique<fan::graphics::model_t>(model_properties);
        property_type = entities_t::property_types_e::none;
        entity_list.emplace(fs_path.filename().string(), std::move(entity));
      }
      void begin_render() {
        gui::set_next_window_bg_alpha(0.5f);
        get_editor().begin_render_common("Entity list", flags, gui::window_flags_menu_bar | gui::window_flags_no_title_bar);
        if (gui::begin_menu_bar()) {
          if (gui::begin_menu("Animation")) {
            if (gui::menu_item("Open model")) {
              fan::graphics::open_files("gltf;fbx;glb;dae;vrm", [&](std::vector<std::string_view> paths) {
                for (const std::string_view path : paths) {
                  if (path.size()) {
                    add_entity(path);
                  }
                }
              }, [] {});
            }
            if (gui::menu_item("Save as")) {
              fan::graphics::save_file("gltf", [&](std::string_view p) {
                // logic
              }, [] {});
            }
            gui::end_menu();
          }
          gui::end_menu_bar();
        }

        std::size_t iterator_index = 0;
        std::size_t click_index = 0;
        std::size_t node_clicked = -1;

        //auto& [name, entity] : entity_list
        for (auto it = entity_list.begin(); it != entity_list.end(); ++it) {
          const std::string name = it->first;
          entity_t& entity = it->second;
          auto base_flags = gui::tree_node_flags_open_on_arrow | gui::tree_node_flags_open_on_double_click | gui::tree_node_flags_span_avail_width;
          bool is_selected = (tree_node_selection_mask & (1ull << click_index)) != 0;
          if (is_selected) base_flags |= gui::tree_node_flags_selected;
          bool b0 = gui::tree_node_ex((name + "##" + std::to_string(click_index)).c_str(), base_flags);
          base_flags &= ~gui::tree_node_flags_selected;
          if (gui::is_item_toggled_open()) {
            tree_node_selection_mask = 0;
            node_clicked = -1;
            property_type = entities_t::property_types_e::none;
          }
          if (gui::is_item_clicked() && !gui::is_item_toggled_open()) {
            node_clicked = click_index;
            selected_entity = entity_list.begin();
            std::advance(selected_entity, iterator_index);
          }
          if (get_engine().is_mouse_released(fan::mouse_right) && gui::is_item_hovered()) {
            gui::open_popup("RightClickPopUp");
          }
          bool did_action = 0;
          if (gui::begin_popup("RightClickPopUp")) {
            if (gui::selectable("Delete")) {
              if (it == selected_entity) selected_entity = entity_list.end();
              it = entity_list.erase(it);
              gui::close_current_popup();
              did_action = 1;
              property_type = entities_t::property_types_e::none;
            }
            gui::end_popup();
          }
          if (did_action) continue;

          ++click_index;

          if (b0) {
            property_types_e iterator;
            for (uint32_t i = 1; i < iterator.size(); ++i) {
              base_flags &= ~gui::tree_node_flags_selected;
              is_selected = (tree_node_selection_mask & (1ull << click_index)) != 0;
              if (is_selected) base_flags |= gui::tree_node_flags_selected;
              if (entity.model->bone_map.size()) {
                bool b1 = gui::tree_node_ex(iterator.NA(i)->sn, base_flags | gui::tree_node_flags_leaf);
                if (gui::is_item_toggled_open()) {
                  tree_node_selection_mask = 0;
                  node_clicked = -1;
                  property_type = entities_t::property_types_e::none;
                }
                if (gui::is_item_clicked() && !gui::is_item_toggled_open()) {
                  property_type = i;
                  node_clicked = click_index;
                  selected_entity = entity_list.begin();
                  std::advance(selected_entity, iterator_index);
                }
                if (b1) gui::tree_pop();
              }
              ++click_index;
            }
            gui::tree_pop();
          }

          {
            auto& camera = get_editor().camera;
            fan::ray3_t ray = get_engine().convert_mouse_to_ray(
             /* get_engine().get_mouse_position() - get_editor().render_view.viewport_position,
              get_editor().render_view.viewport_size,*/
              camera.position,
              camera.projection, 
              camera.view
            );
            bool hovering_on_model = fan::math::d3::is_ray_intersecting_cube(
              ray, 
              it->second.model->user_transform.get_translation() + it->second.model->aabbmin,
              it->second.model->aabbmax - it->second.model->aabbmin
            );
            if (hovering_on_model && get_engine().is_mouse_clicked()) {
              gizmo_model = it->second.model.get();
            }
          }

          // push draw
          get_engine().add_custom_single_draw([&] {
            if (entity.model->is_humanoid()) {
              entity.model->fk_calculate_poses();
              auto bts = entity.model->fk_calculate_transformations();
              entity.model->draw({ 1 }, bts);
            }
            else {
              entity.model->draw();
            }
          });
          ++iterator_index;
        }
        if (node_clicked != (std::size_t)-1) {
          if (gui::get_io().KeyCtrl)
            tree_node_selection_mask ^= (1ull << node_clicked);          // CTRL+click to toggle
          else //if (!(selection_mask & (1 << node_clicked))) // Depending on selection behavior you want, may want to preserve selection when clicking on item that is part of the selection
            tree_node_selection_mask = (1ull << node_clicked);           // Click to single-select
        }
        auto& loco = get_engine();
        if (selected_entity != entity_list.end()) {
          if (get_editor().skeleton_properties.render_bones) {
            auto& camera = get_editor().camera;
            fan::ray3_t ray = loco.convert_mouse_to_ray(
              /*loco.get_mouse_position() - get_editor().render_view.viewport_position,
              get_editor().render_view.viewport_size,*/
              camera.position,
              camera.projection, 
              camera.view
            );
            entity_t& e = selected_entity->second;
            get_editor().skeleton_properties.visual_bones.clear();
            bool picked = false;
            for (auto& [name, bone] : e.model->bone_map) {
              fan::graphics::shapes::rectangle3d_t::properties_t rp;
              rp.position = (bone->bone_transform).get_translation() + e.model->user_transform.get_translation();
              rp.size = e.model->user_transform.get_scale().max() * get_editor().skeleton_properties.bone_scale;
              bool hovering_on_bone = fan::math::d3::is_ray_intersecting_cube(ray, rp.position, rp.size);
              if (hovering_on_bone && get_engine().is_mouse_clicked() && !picked) 
              {
                if (selected_bone == bone) {
                  selected_bone = nullptr;
                }
                else {
                  selected_bone = bone;
                }
                picked = true;
              }
              rp.color = get_editor().skeleton_properties.bone_color;
              if (bone == selected_bone || hovering_on_bone) {
                rp.color.r = rp.color.r * 3;
                rp.color.g = rp.color.g * 3;
              }
              rp.camera = get_editor().camera_nr;
              rp.viewport = get_editor().viewport_nr;
              get_editor().skeleton_properties.visual_bones.push_back(rp);
            }
          }
        }
      }
      void end_render() {
        gui::end();
      }
      struct entity_t {
        std::unique_ptr<fan::graphics::model_t> model;
        bool selected = false;
      };
      // name, data
      std::unordered_multimap<std::string, entity_t> entity_list;
      uint16_t flags = 0;
      std::string file_dialog_path;
      std::vector<std::string> file_dialog_paths;
      decltype(entity_list)::iterator selected_entity = entity_list.end();
      fan::model::bone_t* selected_bone = nullptr;
      uint64_t tree_node_selection_mask = 0;
      fan::graphics::model_t* gizmo_model = nullptr;
      struct property_types_e : __dme_inherit(property_types_e, __empty_struct) {
        property_types_e() {}
#define d(name, ...) __dme(name);
        d(none);
        d(animation);
        d(bones);
#undef d
      };
      uint16_t property_type = property_types_e::none;
      //property_types_e property_type;
      //property_types_e
    }entities;

    struct entity_properties_t {
      editor_t& get_editor() {
        return *OFFSETLESS(this, editor_t, entity_properties);
      }
      loco_t& get_engine() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void begin_render() {
        gui::set_next_window_bg_alpha(0.5f);
        if (get_editor().begin_render_common("Entity properties", flags)) {
          auto& selected_entity = get_editor().entities.selected_entity;
          if (selected_entity != get_editor().entities.entity_list.end()) {
            fan::graphics::model_t& model = *selected_entity->second.model;
            entities_t& entities = get_editor().entities;
            using pte = entities_t::property_types_e;
            switch (entities.property_type) {
            case pte::none: {
              //fan_imgui_dragfloat1(model.user_position, 0.01);
              //fan_imgui_dragfloat(model.user_rotation, 0.01, -fan::math::pi, fan::math::pi);
              //fan_imgui_dragfloat1(model.user_scale, 0.01);
              break;
            }
            case pte::animation: {
              fan::graphics::model_t& model = *selected_entity->second.model;
              if (gui::checkbox("play animation", &model.play_animation)) {
              }
              std::vector<const char*> animations;
              for (auto& i : model.animation_list) {
                animations.push_back(i.first.c_str());
              }
              int current_id = model.get_active_animation_id();
              if (gui::list_box("animation list", &current_id, animations.data(), animations.size())) {
                model.active_anim = animations[current_id];
              }
              gui::drag("animation weight", &model.animation_list[model.active_anim].weight, 0.01, 0, 1);
              break;
            }
            case pte::bones: {
              break;
            }
            }
          }
        }
        int id = 0;
        for (auto& selected_entity : get_editor().entities.entity_list) {
          gui::push_id(id++);
          fan::graphics::model_t& model = *selected_entity.second.model;
          if (true) {
            model.dt += get_engine().get_delta_time() * 1000;
          }
          gui::pop_id();
        }
      }
      void end_ender() {
        gui::end();
      }
      uint16_t flags = 0;
    }entity_properties;

    struct camera_properties_t {
      editor_t& get_editor() {
        return *OFFSETLESS(this, editor_t, camera_properties);
      }
      loco_t& get_engine() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void begin_render() {
        if (render_this) {
          gui::set_next_window_bg_alpha(0.5f);
          gui::push_style_var(gui::style_var_frame_rounding, 2);
          gui::push_style_var(gui::style_var_window_padding, {10, 10});
          if (get_editor().begin_render_common("Camera properties", flags)) {
            gui::slider("zfar", &get_editor().camera.zfar, 1, 10000);
            gui::slider("znear", &get_editor().camera.znear, 0.001, 10);
            gui::slider("fov", &fov, 1, 180);
            gui::slider("sensitivity", &get_editor().camera.sensitivity, 0.01, 1);
            gui::slider("speed", &speed, 1, 10000);
            gui::slider("friction", &friction, 0, 30);
          }
          gui::pop_style_var(2);
        }
      }

      void end_render() {
        if (render_this) {
          gui::end();
        }
      }
#if defined(FAN_JSON)
      // expects camera block
      void import_settings(const fan::json& data) {
        get_editor().camera.zfar = data["zfar"];
        get_editor().camera.znear = data["znear"];
        fov = data["fov"];
        get_editor().camera.sensitivity = data["sensitivity"];
        speed = data["speed"];
        friction = data["friction"];
      }
      fan::json get_settings() {
        fan::json data;
        data["zfar"] = get_editor().camera.zfar;
        data["znear"] = get_editor().camera.znear;
        data["fov"] = fov;
        data["sensitivity"] = get_editor().camera.sensitivity;
        data["speed"] = speed;
        data["friction"] = friction;
        return data;
      }
#endif
      uint16_t flags = 0;
      f32_t fov = 90.f;
      f32_t speed = 1000;
      f32_t friction = 12;
      f32_t acceleration = 10; // todo
      bool render_this = false;
    }camera_properties;


    struct lighting_properties_t {
      editor_t& get_editor() {
        return *OFFSETLESS(this, editor_t, lighting_properties);
      }
      loco_t& get_engine() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void begin_render() {
        if (render_this) {
          gui::set_next_window_bg_alpha(0.5f);
          gui::push_style_var(gui::style_var_frame_rounding, 2);
          gui::push_style_var(gui::style_var_window_padding, {10, 10});
          if (get_editor().begin_render_common("Lighting properties", flags)) {
            if (get_editor().entities.selected_entity == get_editor().entities.entity_list.end()) {
              gui::text("Select model");
            }
            else {
              fan::graphics::model_t& model = *get_editor().entities.selected_entity->second.model;
              gui::drag("light_position", &model.light_position, 0.2);
              gui::color_edit3("model->light_color", &model.light_color);
              gui::drag("light_intensity", &model.light_intensity, 0.1);
              static f32_t specular_strength = 0.5;
              if (gui::drag("specular_strength", &specular_strength, 0.01)) {
                get_engine().shader_set_value(model.m_shader, "specular_strength", specular_strength);
              }
            }
          }
          gui::pop_style_var(2);
        }
      }

      void end_render() {
        if (render_this) {
          gui::end();
        }
      }
      uint16_t flags = 0;
      bool render_this = false;
    }lighting_properties;

    struct skeleton_properties_t {
      editor_t& get_editor() {
        return *OFFSETLESS(this, editor_t, skeleton_properties);
      }
      loco_t& get_engine() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void begin_render() {
        if (render_this) {
          gui::set_next_window_bg_alpha(0.5f);
          gui::push_style_var(gui::style_var_frame_rounding, 2);
          gui::push_style_var(gui::style_var_window_padding, {10, 10});
          if (get_editor().begin_render_common("Skeleton Properties", flags)) {
            if (gui::toggle_button("render bones", &render_bones)) {
              visual_bones.clear();
            }
            gui::slider("bone scale", &bone_scale, 0.001, 10);
            gui::color_edit4("bone color", &bone_color);
          }
          gui::pop_style_var(2);
        }
      }

      void end_render() {
        if (render_this) {
          gui::end();
        }
      }
      bool render_bones = 0;
      fan::color bone_color = fan::colors::gray;
      f32_t bone_scale = 0.5;
      uint16_t flags = 0;
      bool render_this = false;
      std::vector<fan::graphics::shape_t> visual_bones;
    }skeleton_properties;

    void begin_render() {
      gui::begin_main_menu_bar();
      if (gui::begin_menu("Settings")) {
    #if defined(FAN_JSON)
        if (gui::menu_item("Load")) {
          std::string editor_settings;
          fan::io::file::read("scene_editor.ini", &editor_settings);
          fan::json settings = fan::json::parse(editor_settings);
          camera_properties.import_settings(settings["camera"]);
        }
        if (gui::menu_item("Save")) {
          fan::json camera_settings = camera_properties.get_settings();
          fan::json output;
          output["camera"] = camera_settings;
          fan::io::file::write("scene_editor.ini", output.dump(), std::ios_base::binary);
        }
    #endif
        gui::end_menu();
      }
      gui::end_main_menu_bar();
      get_engine().camera_set_perspective(camera_nr, camera_properties.fov, viewport.size);
      gui::begin_disabled(!cursor_mode);
      render_view.begin_render();
      lighting_properties.begin_render();
      camera_properties.begin_render();
      skeleton_properties.begin_render();
      entities.begin_render();
      entity_properties.begin_render();
    }
    void end_render() {
      entity_properties.end_ender();
      entities.end_render();
      skeleton_properties.end_render();
      camera_properties.end_render();
      lighting_properties.end_render();
      render_view.end_render();
      content_browser.render();
      gui::end_disabled();
    }
    // 0 == invisible, 1 == visible
    int cursor_mode = 1;

    fan::graphics::camera_t camera_nr;
    fan::graphics::viewport_t viewport_nr;
    fan::graphics::context_camera_t& camera;
    fan::graphics::context_viewport_t& viewport;
    fan::graphics::gui::content_browser_t content_browser{false};

    fan::graphics::image_t icon_video_camera = gloco()->image_load("images/editor/video-camera.webp");
    fan::graphics::image_t icon_lightbulb = gloco()->image_load("images/editor/lightbulb.webp");
    fan::graphics::image_t icon_skeleton = gloco()->image_load("images/editor/skeleton.webp");
  }editor;
};
std::unique_ptr<pile_t> pile;

int main() {
  pile = std::make_unique<pile_t>();
  pile->loco.loop([&] {
    pile->editor.begin_render();
    pile->editor.end_render();
  });
}