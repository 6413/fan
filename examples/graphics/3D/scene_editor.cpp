#include <fan/pch.h>

#include <fan/graphics/opengl/3D/objects/model.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <imgui/imgui_internal.h>

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
    bool begin_render_common(const char* window_name, const fan::vec2& position, const fan::vec2& size, uint16_t& flags, uint16_t custom_flags = 0) {
      fan::vec2 window_size = get_loco().window.get_size();
      bool ret = ImGui::Begin(window_name, 0, 
        ImGuiWindowFlags_NoCollapse | custom_flags
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
        if (get_editor().begin_render_common("Render view", 0, initial_render_view_size, flags)) {
         get_loco().set_imgui_viewport(get_editor().viewport_nr);
        }

        // drag and drop
        get_editor().content_browser.receive_drag_drop_target([this](const std::filesystem::path& path) {
          get_editor().entities.add_entity(path.string());
        });
      }
      void end_render() {
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
      }
      bool hovered() const {
        return flags & flags_e::hovered;
      }
      uint16_t flags = 0;
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
        ImGuiWindowClass window_class;
        window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoWindowMenuButton;
        ImGui::SetNextWindowClass(&window_class);
        get_editor().begin_render_common("Entity list", { initial_render_view_size.x, 0 }, { 0.5, 0.5 }, flags, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar);
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
        if (get_editor().begin_render_common("Entity properties", { initial_render_view_size.x, 0.5 }, { 0.5, 0.5 }, flags)) {
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
            case pte::bones: {
              for (auto& [name, bone] : model.bone_map) {
                if (bone->name.contains("Neck")) {
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

    void begin_render() {
      camera.move(1000);
      get_loco().camera_set_perspective(camera_nr, 90.f, viewport.viewport_size);
      ImGui::BeginDisabled(!cursor_mode);
      render_view.begin_render();
      entities.begin_render();
      entity_properties.begin_render();
    }
    void end_render() {
      entity_properties.end_ender();
      entities.end_render();
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