#include <fan/pch.h>
#include <fan/graphics/opengl/3D/objects/model.h>

struct pile_t {
  loco_t loco;

  static constexpr fan::vec2 initial_render_view_size{0.7, 1};

  struct editor_t {
    struct flags_e {
      enum {
        hovered = 1 << 0,
      };
    };

    loco_t& get_loco() {
      return OFFSETLESS(this, pile_t, editor)->loco;
    }

    // takes position 0-1 and size 0-1 scale based on window size
    bool begin_render_common(const char* window_name, uint16_t& flags, uint16_t custom_flags = 0) {
      fan::vec2 window_size = get_loco().window.get_size();
      bool ret = ImGui::Begin(window_name, 0, 
        custom_flags
      );
      if (ret) {
        flags &= ~flags_e::hovered;
        flags |= (uint16_t)ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
      }
      return ret;
    }

    editor_t() : 
      camera_nr(get_loco().open_camera_perspective()),
      viewport_nr(get_loco().open_viewport(0, get_loco().window.get_size())),
      camera(get_loco().camera_get(camera_nr)), // kinda waste, but shorter code
      viewport(get_loco().viewport_get(viewport_nr)) // kinda waste, but shorter code
    {
      init_editor_theme();
      camera.position = {0.04, 0.71, 1.35};
      camera.m_yaw = -180;
      camera.m_pitch = -5.9;
      ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;

      content_browser.current_directory /= "models";
      content_browser.update_directory_cache();

      get_loco().window.add_mouse_motion([&](const auto& d) {
        if (d.motion != 0 && cursor_mode == 0) {
          camera.rotate_camera(d.motion);
        }
      });

      get_loco().window.add_buttons_callback([&](const auto& d) { 
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
            get_loco().window.set_cursor(cursor_mode);
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
      loco_t& get_loco() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void begin_render() {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, 0);
        if (toggle_focus) {
          ImGui::SetNextWindowFocus();
          toggle_focus = false;
        }
        if (get_editor().begin_render_common("Render view", flags)) {
         get_loco().set_imgui_viewport(get_editor().viewport_nr);
          if (ImGui::IsWindowFocused()) {
            get_editor().camera.move(get_editor().camera_properties.speed, get_editor().camera_properties.friction);
          }

          // drag and drop
          get_editor().content_browser.receive_drag_drop_target([this](const std::filesystem::path& path) {
            get_editor().entities.add_entity(path.string());
          });
          fan::vec2 image_size = 48;
          ImGui::SetCursorPos(fan::vec2(image_size.x / 2, image_size.y));
          ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 20);
          ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1);
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2, 0.2, 0.2, 0.2));
          ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
          if (ImGui::ToggleImageButton("##Camera Properties", get_editor().icon_video_camera, image_size, &toggle_camera_properties)) {
            get_editor().camera_properties.render_this = !get_editor().camera_properties.render_this;
          }
          ImGui::PopStyleColor(3);
          ImGui::PopStyleVar(2);
        }
      }
      void end_render() {
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
      }
      bool hovered() const {
        return flags & flags_e::hovered;
      }
      bool toggle_camera_properties = 0;
      uint16_t flags = 0;
      bool toggle_focus = false;
    }render_view;
    // list of entities
    struct entities_t {
      editor_t& get_editor() {
        return *OFFSETLESS(this, editor_t, entities);
      }
      loco_t& get_loco() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void add_entity(const std::string& path) {
        entity_t entity;
        std::filesystem::path fs_path(path);
        fan::graphics::model_t::properties_t model_properties;
        model_properties.path = path;
        model_properties.texture_path = (fs_path.parent_path() / "/textures").string();
        model_properties.use_cpu = 0;
        model_properties.camera = get_editor().camera_nr;
        model_properties.viewport = get_editor().viewport_nr;
        entity.model = std::make_unique<fan::graphics::model_t>(model_properties);
        entity.model->upload_modified_vertices();
        entity_list.emplace(fs_path.filename().string(), std::move(entity));
      }
      void begin_render() {
        ImGui::SetNextWindowBgAlpha(0.5f);
        get_editor().begin_render_common("Entity list", flags, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar);
          if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Animation")) {
              if (ImGui::MenuItem("Open model")) {
                open_file_dialog.load("gltf,fbx,glb,dae,vrm", &file_dialog_paths);
              }
              if (ImGui::MenuItem("Save as")) {
                save_file_dialog.save("gltf", &file_dialog_path);
              }
              ImGui::EndMenu();
            }
          }
          ImGui::EndMenuBar();

          std::size_t iterator_index = 0;
          std::size_t click_index = 0;
          std::size_t node_clicked = -1;
          
          //auto& [name, entity] : entity_list
          for (auto it = entity_list.begin(); it != entity_list.end(); ++it) {
            const std::string name = it->first;
            entity_t& entity = it->second;
            ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |ImGuiTreeNodeFlags_SpanAvailWidth;
            bool is_selected = (tree_node_selection_mask & (1ull << click_index)) != 0;
            if (is_selected)
              base_flags |= ImGuiTreeNodeFlags_Selected;
            bool b0 = ImGui::TreeNodeEx((name + "##" + std::to_string(click_index)).c_str(), base_flags);
            base_flags &= ~ImGuiTreeNodeFlags_Selected;
            if (ImGui::IsItemToggledOpen()) {
              tree_node_selection_mask = 0;
              node_clicked = -1;
            }
            // assigned selected element
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
              node_clicked = click_index;
              selected_entity = entity_list.begin();
              std::advance(selected_entity, iterator_index);
              property_type = entities_t::property_types_e::none;
            }
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Right) && ImGui::IsItemHovered()) {
              ImGui::OpenPopup("RightClickPopUp");
            }
            bool did_action = 0;
            if (ImGui::BeginPopup("RightClickPopUp")) {
              if (ImGui::Selectable("Delete")) {
                if (it == selected_entity) {
                  selected_entity = entity_list.end();
                }
                it = entity_list.erase(it);
                ImGui::CloseCurrentPopup();
                did_action = 1;
              }
              ImGui::EndPopup();
            }
            if (did_action) {
              continue;
            }

            ++click_index;

            if (b0) {              
              property_types_e iterator;
              // skip none
              for (uint32_t i = 1; i < iterator.GetMemberAmount(); ++i) {
                base_flags &= ~ImGuiTreeNodeFlags_Selected;
                //child options
                is_selected = (tree_node_selection_mask & (1ull << click_index)) != 0;
                if (is_selected)
                    base_flags |= ImGuiTreeNodeFlags_Selected;
                if (entity.model->bone_map.size()) {
                  bool b1 = ImGui::TreeNodeEx(iterator.NA(i)->sn, (base_flags | ImGuiTreeNodeFlags_Leaf));
                  if (ImGui::IsItemToggledOpen()) {
                    tree_node_selection_mask = 0;
                    node_clicked = -1;
                  }
                    
                  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    property_type = i;
                    node_clicked = click_index;
                    selected_entity = entity_list.begin();
                    std::advance(selected_entity, iterator_index);
                  }
                  if (b1) {
                    ImGui::TreePop();
                  }
                }
                ++click_index;
              }
              ImGui::TreePop();
            }

            // push draw
            get_loco().single_queue.push_back([&] {
              if (entity.model->is_humanoid()) {
                entity.model->fk_calculate_poses();
                auto bts = entity.model->fk_calculate_transformations();
                entity.model->draw({1}, bts);
              }
              else {
                entity.model->draw();
              }
            });
            ++iterator_index;
          }
          if (node_clicked != (std::size_t)-1) {
            if (ImGui::GetIO().KeyCtrl)
                tree_node_selection_mask ^= (1ull << node_clicked);          // CTRL+click to toggle
            else //if (!(selection_mask & (1 << node_clicked))) // Depending on selection behavior you want, may want to preserve selection when clicking on item that is part of the selection
                tree_node_selection_mask = (1ull << node_clicked);           // Click to single-select
          }
          { // filesystem
            if (open_file_dialog.is_finished()) {
              for (const std::string path : file_dialog_paths) {
                if (path.size()) {
                  add_entity(path);
                }
              }
              open_file_dialog.finished = false;
              file_dialog_paths.clear();
            }

            //if (save_file_dialog.is_finished()) {
            //  if (file_dialog_path.size() != 0) {
            //    auto ext = std::filesystem::path(file_dialog_path).extension();
            //    if (ext != ".gltf") {
            //      file_dialog_path += ".gltf";
            //    }
            //    // exporter will not export custom animations made, yet
            //    model->export_animation(model->get_active_animation().name, file_dialog_path);
            //  }
            //  save_file_dialog.finished = false;
            //}
          //} // filesystem
        }
      }
      void end_render() {
        ImGui::End();
      }
      struct entity_t {
        std::unique_ptr<fan::graphics::model_t> model;
        bool selected = false;
      };
      // name, data
      std::unordered_multimap<std::string, entity_t> entity_list;
      uint16_t flags = 0;
      fan::graphics::file_open_dialog_t open_file_dialog;
      fan::graphics::file_save_dialog_t save_file_dialog;
      std::string file_dialog_path;
      std::vector<std::string> file_dialog_paths;
      decltype(entity_list)::iterator selected_entity = entity_list.end();
      uint64_t tree_node_selection_mask = 0;
      struct property_types_e : __dme_inherit(property_types_e, __empty_struct){
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
      loco_t& get_loco() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void begin_render() {
        ImGui::SetNextWindowBgAlpha(0.5f);
        if (get_editor().begin_render_common("Entity properties", flags)) {
          auto& selected_entity = get_editor().entities.selected_entity;
          if (selected_entity != get_editor().entities.entity_list.end()) {
            fan::graphics::model_t& model = *selected_entity->second.model;
            entities_t& entities = get_editor().entities;
            using pte = entities_t::property_types_e;
            switch(entities.property_type) {
            case pte::none: {
              fan_imgui_dragfloat1(model.user_position, 0.01);
              fan_imgui_dragfloat(model.user_rotation, 0.01, -fan::math::pi, fan::math::pi);
              fan_imgui_dragfloat1(model.user_scale, 0.01);
              break;
            }
            case pte::animation: {

              break;
            }
            case pte::bones: {
              for (auto& [name, bone] : model.bone_map) {
                if (bone->name.find("Neck") != std::string::npos) {
                  fan::vec3 camera_position = get_editor().camera.position;

                  fan::vec3 model_position = selected_entity->second.model->user_position +
                    bone->bone_transform.get_translation() * selected_entity->second.model->user_scale;
                  fan::vec3 forward = (camera_position - model_position).normalize();
                  float pitch = std::atan2(-forward.y, std::sqrt(forward.x * forward.x + forward.z * forward.z));
                  float yaw = std::atan2(forward.x, forward.z);

                  bone->user_rotation.x = -pitch;
                  bone->user_rotation.y = -yaw;
                }
                if (ImGui::TreeNodeEx(bone->name.c_str())) {
                  {
                    fan_imgui_dragfloat1(bone->user_position, 0.1);
                    fan_imgui_dragfloat(bone->user_rotation, 0.01, -fan::math::pi, fan::math::pi);
                    fan_imgui_dragfloat1(bone->user_scale, 0.01);
                  }
                  ImGui::TreePop();
                }
              }
              break;
            }
            }
          }
        }
      }
      void end_ender() {
        ImGui::End();
      }
      uint16_t flags = 0;
    }entity_properties;

    struct camera_properties_t {
      editor_t& get_editor() {
        return *OFFSETLESS(this, editor_t, camera_properties);
      }
      loco_t& get_loco() {
        return OFFSETLESS(&get_editor(), pile_t, editor)->loco;
      }
      void begin_render() {
        if (render_this) {
          ImGui::SetNextWindowBgAlpha(0.5f);
          ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2);
          ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
          if (get_editor().begin_render_common("Camera properties", flags)) {
            ImGui::SliderFloat("zfar", &get_editor().camera.zfar, 1, 10000);
            ImGui::SliderFloat("znear", &get_editor().camera.znear, 0.001, 10);
            ImGui::SliderFloat("fov", &fov, 1, 180);
            ImGui::SliderFloat("sensitivity", &get_editor().camera.sensitivity, 0.01, 1);
            ImGui::SliderFloat("speed", &speed, 1, 10000);
            ImGui::SliderFloat("friction", &friction, 0, 30);
          }
          ImGui::PopStyleVar(2);
        }
      }
      void end_render() {
        if (render_this) {
          ImGui::End();
        }
      }
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
      uint16_t flags = 0;
      f32_t fov = 90.f;
      f32_t speed = 1000;
      f32_t friction = 12;
      f32_t acceleration = 10; // todo
      bool render_this = false;
    }camera_properties;

    void begin_render() {
      ImGui::BeginMainMenuBar();
      if (ImGui::BeginMenu("Settings")) {
        if (ImGui::MenuItem("Load")) {
          std::string editor_settings;
          fan::io::file::read("scene_editor.ini", &editor_settings);
          fan::json settings = fan::json::parse(editor_settings);
          camera_properties.import_settings(settings["camera"]);
        }
        if (ImGui::MenuItem("Save")) {
          fan::json camera_settings = camera_properties.get_settings();
          fan::json output;
          output["camera"] = camera_settings;

          fan::io::file::write("scene_editor.ini", output.dump(), std::ios_base::binary);
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
      get_loco().camera_set_perspective(camera_nr, camera_properties.fov, viewport.viewport_size);
      ImGui::BeginDisabled(!cursor_mode);
      render_view.begin_render();
      camera_properties.begin_render();
      entities.begin_render();
      entity_properties.begin_render();
    }
    void end_render() {
      entity_properties.end_ender();
      entities.end_render();
      camera_properties.end_render();
      render_view.end_render();
      content_browser.render();      
      ImGui::EndDisabled();
    }
    // 0 == invisible, 1 == visible
    int cursor_mode = 1;

    loco_t::camera_t camera_nr;
    loco_t::viewport_t viewport_nr;
    fan::opengl::context_t::camera_t& camera;
    fan::opengl::context_t::viewport_t& viewport;
    fan::graphics::imgui_content_browser_t content_browser;

    loco_t::image_t icon_video_camera = gloco->image_load("images_editor/video-camera.webp");

    void init_editor_theme() {
	    ImGuiStyle& style = ImGui::GetStyle();
	
	    style.Alpha = 1.0f;
	    style.DisabledAlpha = 0.6000000238418579f;
	    style.WindowPadding = ImVec2(5.5f, 8.300000190734863f);
	    style.WindowRounding = 4.5f;
	    style.WindowBorderSize = 1.0f;
	    style.WindowMinSize = ImVec2(32.0f, 32.0f);
	    style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
	    style.WindowMenuButtonPosition = ImGuiDir_Left;
	    style.ChildRounding = 3.200000047683716f;
	    style.ChildBorderSize = 1.0f;
	    style.PopupRounding = 2.700000047683716f;
	    style.PopupBorderSize = 1.0f;
	    style.FramePadding = ImVec2(4.0f, 3.0f);
	    style.FrameRounding = 2.400000095367432f;
	    style.FrameBorderSize = 0.0f;
	    style.ItemSpacing = ImVec2(8.0f, 4.0f);
	    style.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
	    style.CellPadding = ImVec2(4.0f, 2.0f);
	    style.IndentSpacing = 21.0f;
	    style.ColumnsMinSpacing = 6.0f;
	    style.ScrollbarSize = 14.0f;
	    style.ScrollbarRounding = 9.0f;
	    style.GrabMinSize = 10.0f;
	    style.GrabRounding = 3.200000047683716f;
	    style.TabRounding = 3.5f;
	    style.TabBorderSize = 1.0f;
	    style.TabMinWidthForCloseButton = 0.0f;
	    style.ColorButtonPosition = ImGuiDir_Right;
	    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
	    style.SelectableTextAlign = ImVec2(0.0f, 0.0f);
	
	    style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
	    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.4980392158031464f, 0.4980392158031464f, 0.4980392158031464f, 1.0f);
	    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.05882352963089943f, 0.05882352963089943f, 0.05882352963089943f, 0.9399999976158142f);
	    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
	    style.Colors[ImGuiCol_PopupBg] = ImVec4(0.0784313753247261f, 0.0784313753247261f, 0.0784313753247261f, 0.9399999976158142f);
	    style.Colors[ImGuiCol_Border] = ImVec4(0.4274509847164154f, 0.4274509847164154f, 0.4980392158031464f, 0.5f);
	    style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
	    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.1372549086809158f, 0.1725490242242813f, 0.2274509817361832f, 0.5400000214576721f);
	    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.2117647081613541f, 0.2549019753932953f, 0.3019607961177826f, 0.4000000059604645f);
	    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.04313725605607033f, 0.0470588244497776f, 0.0470588244497776f, 0.6700000166893005f);
	    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.03921568766236305f, 0.03921568766236305f, 0.03921568766236305f, 1.0f);
	    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.0784313753247261f, 0.08235294371843338f, 0.09019608050584793f, 1.0f);
	    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.0f, 0.0f, 0.0f, 0.5099999904632568f);
	    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.1372549086809158f, 0.1372549086809158f, 0.1372549086809158f, 1.0f);
	    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.01960784383118153f, 0.01960784383118153f, 0.01960784383118153f, 0.5299999713897705f);
	    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.3098039329051971f, 0.3098039329051971f, 0.3098039329051971f, 1.0f);
	    style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.407843142747879f, 0.407843142747879f, 0.407843142747879f, 1.0f);
	    style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.5098039507865906f, 0.5098039507865906f, 0.5098039507865906f, 1.0f);
	    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.7176470756530762f, 0.7843137383460999f, 0.843137264251709f, 1.0f);
	    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.47843137383461f, 0.5254902243614197f, 0.572549045085907f, 1.0f);
	    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.2901960909366608f, 0.3176470696926117f, 0.3529411852359772f, 1.0f);
	    style.Colors[ImGuiCol_Button] = ImVec4(0.1490196138620377f, 0.1607843190431595f, 0.1764705926179886f, 0.4000000059604645f);
	    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.1372549086809158f, 0.1450980454683304f, 0.1568627506494522f, 1.0f);
	    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.09019608050584793f, 1.0f);
	    style.Colors[ImGuiCol_Header] = ImVec4(0.196078434586525f, 0.2156862765550613f, 0.239215686917305f, 0.3100000023841858f);
	    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.1647058874368668f, 0.1764705926179886f, 0.1921568661928177f, 0.800000011920929f);
	    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.07450980693101883f, 0.08235294371843338f, 0.09019608050584793f, 1.0f);
	    style.Colors[ImGuiCol_Separator] = ImVec4(0.4274509847164154f, 0.4274509847164154f, 0.4980392158031464f, 0.5f);
	    style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.239215686917305f, 0.3254902064800262f, 0.4235294163227081f, 0.7799999713897705f);
	    style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.2745098173618317f, 0.3803921639919281f, 0.4980392158031464f, 1.0f);
	    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.2901960909366608f, 0.3294117748737335f, 0.3764705955982208f, 0.2000000029802322f);
	    style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.239215686917305f, 0.2980392277240753f, 0.3686274588108063f, 0.6700000166893005f);
	    style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.1647058874368668f, 0.1764705926179886f, 0.1882352977991104f, 0.949999988079071f);
	    style.Colors[ImGuiCol_Tab] = ImVec4(0.1176470592617989f, 0.125490203499794f, 0.1333333402872086f, 0.8619999885559082f);
	    style.Colors[ImGuiCol_TabHovered] = ImVec4(0.3294117748737335f, 0.407843142747879f, 0.501960813999176f, 0.800000011920929f);
	    style.Colors[ImGuiCol_TabActive] = ImVec4(0.2431372553110123f, 0.2470588237047195f, 0.2549019753932953f, 1.0f);
	    style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.06666667014360428f, 0.1019607856869698f, 0.1450980454683304f, 0.9724000096321106f);
	    style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.1333333402872086f, 0.2588235437870026f, 0.4235294163227081f, 1.0f);
	    style.Colors[ImGuiCol_PlotLines] = ImVec4(0.6078431606292725f, 0.6078431606292725f, 0.6078431606292725f, 1.0f);
	    style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.0f, 0.4274509847164154f, 0.3490196168422699f, 1.0f);
	    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.8980392217636108f, 0.6980392336845398f, 0.0f, 1.0f);
	    style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.0f, 0.6000000238418579f, 0.0f, 1.0f);
	    style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.1882352977991104f, 0.1882352977991104f, 0.2000000029802322f, 1.0f);
	    style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.3098039329051971f, 0.3098039329051971f, 0.3490196168422699f, 1.0f);
	    style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.2274509817361832f, 0.2274509817361832f, 0.2470588237047195f, 1.0f);
	    style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
	    style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.05999999865889549f);
	    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.2588235437870026f, 0.5882353186607361f, 0.9764705896377563f, 0.3499999940395355f);
	    style.Colors[ImGuiCol_DragDropTarget] = ImVec4(1.0f, 1.0f, 0.0f, 0.8999999761581421f);
	    style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.2588235437870026f, 0.5882353186607361f, 0.9764705896377563f, 1.0f);
	    style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.0f, 1.0f, 1.0f, 0.699999988079071f);
	    style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.800000011920929f, 0.800000011920929f, 0.800000011920929f, 0.2000000029802322f);
	    style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.800000011920929f, 0.800000011920929f, 0.800000011920929f, 0.3499999940395355f);
    }
  }editor;
};
std::unique_ptr<pile_t> pile;

int main() {
  pile = std::make_unique<pile_t>();
  //pile->editor.entities.add_entity("models/Mutant.fbx");
  pile->loco.loop([&] {
    pile->editor.begin_render();
    pile->editor.end_render();
  });
}