#include <fan/pch.h>

struct editor_t {

  struct texturepack_2_imgui_t {

    void open(loco_t::texturepack_t* texturepack) {
      tp = texturepack;

      texturepack_images.reserve(tp->texture_list.size());

      // loaded texturepack
      tp->iterate_loaded_images([this](auto& image, uint32_t pack_id) {
        pack_element_info_t ii;
        ii.ti = loco_t::texturepack_t::ti_t{
          .pack_id = pack_id,
          .position = image.position,
          .size = image.size,
          .image = &tp->get_pixel_data(pack_id).image
        };

        ii.image_hash = image.hash;

        texturepack_images.push_back(ii);
        });
    }

    void render(editor_t* editor) {
      for (uint32_t i = 0; i < texturepack_images.size(); i++) {
        auto& node = texturepack_images[i];
        if (ImGui::ImageButton(
          (fan::string("##ibutton") + std::to_string(i)).c_str(),
          (void*)(intptr_t)node.ti.image->get_texture(),
          (node.ti.size / 4).max(fan::vec2(64)),
          node.ti.position / node.ti.image->size,
          node.ti.position / node.ti.image->size + node.ti.size / node.ti.image->size
        )) {
          current = &node;
          editor->visual_place = fan::graphics::sprite_t{ {
              .position = fan::vec3(fan::vec2(-999999), 0xfff),
              .size = (current->ti.size / 4).max(fan::vec2(64)) * editor->brush.size,
              .blending = true
          }};
          editor->visual_place.set_tp(&current->ti);
        }
        if (images_per_row && (i + 1) % images_per_row) {
          ImGui::SameLine();
        }
      }
    }

    struct pack_element_info_t {
      loco_t::texturepack_t::ti_t ti;
      uint64_t image_hash;
    };

    int images_per_row = 5;
    fan::vec2 sprite_size = 128;
    loco_t::texturepack_t* tp;
    std::vector<pack_element_info_t> texturepack_images;
    pack_element_info_t* current = 0;
  };

  static constexpr f32_t scroll_speed = 1.2;

  editor_t(loco_t::texturepack_t* tp) {
    t2i.open(tp);
    handle_mouse();
    handle_keyboard();
    handle_imgui();
  }

  struct global_t : fan::graphics::vfi_root_t {
    global_t() = default;
    template <typename T>
    global_t(editor_t* editor, const T& obj) {
      T temp = std::move(obj);
      loco_t::shapes_t::vfi_t::properties_t vfip;
      vfip.shape.rectangle->position = temp.get_position();
      vfip.shape.rectangle->position.z += editor->current_z++;
      vfip.shape.rectangle->size = temp.get_size();
      vfip.mouse_button_cb = [editor, this](const auto& d) -> int {
        //fgm->event_type = event_type_e::move;
        editor->current_shape = this;
        return 0;
      };
      fan::graphics::vfi_root_t::set_root(vfip);
      temp.set_position(fan::vec3(fan::vec2(temp.get_position()), editor->current_z - 1));
      fan::graphics::vfi_root_t::push_child(std::move(temp));

      editor->current_shape = this;
    }

    // global data
    fan::string id;
    uint32_t group_id = 0;
  };

  fan::vec2 get_mouse_position() {
    return gloco->translate_position(gloco->get_mouse_position()) + gloco->default_camera->camera.get_position();
  }

  void handle_mouse() {
    gloco->window.add_buttons_callback([this](const auto& d) {
      if (!editor_settings.hovered && d.state != fan::mouse_state::release) {
        return;
      }

      {// handle camera movement
        f32_t old_zoom = viewport_settings.zoom;

        switch (d.button) {
          case fan::mouse_middle: {
            viewport_settings.move = (bool)d.state;
            fan::vec2 old_pos = viewport_settings.pos;
            viewport_settings.offset = gloco->get_mouse_position();
            viewport_settings.pos = gloco->default_camera->camera.get_position();
            break;
          }
          case fan::mouse_scroll_up: {
            viewport_settings.zoom *= scroll_speed;
            return;
          }
          case fan::mouse_scroll_down: {
            viewport_settings.zoom /= scroll_speed;
            return;
          }
          default: {return; } //?
        };
      }// handle camera movement
    });
    gloco->window.add_mouse_move_callback([this](const auto& d) {
      if (viewport_settings.move) {
        fan::vec2 move_off = (d.position - viewport_settings.offset) / viewport_settings.zoom * 2;
        gloco->default_camera->camera.set_position(viewport_settings.pos - move_off);
      }
      if (t2i.current) {
        if (!visual_place.is_invalid()) {
          visual_place.set_position(get_mouse_position());
        }
      }
    });
  }

  void handle_keyboard() {
    gloco->window.add_keys_callback([this](const auto& d) {
      if (d.state != fan::keyboard_state::press) {
        return;
      }
      switch (d.key) {
        case fan::key_delete: {
          if (current_shape) {
            int idx = 0;
            for (auto& i : images) {
              if (i.get() == current_shape) {
                images.erase(images.begin() + idx);
                current_shape = nullptr;
              }
              ++idx;
            }
          }
          break;
        }
      }
  });
  }

  void handle_imgui() {
    element = [&] {
      auto& style = ImGui::GetStyle();
      ImVec4* colors = style.Colors;
      const ImVec4 bgColor = ImVec4(0.1, 0.1, 0.1, 0.1);
      colors[ImGuiCol_WindowBg].w = bgColor.w;
      colors[ImGuiCol_ChildBg].w = bgColor.w;
      colors[ImGuiCol_TitleBg].w = bgColor.w;

      ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
      ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
      ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
      ImGui::PopStyleColor(2);
      ImGui::Begin("Editor");
      fan::vec2 window_size = gloco->window.get_size();
      fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
      fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
      fan::vec2 offset = viewport_size - viewport_size / viewport_settings.zoom;
      fan::vec2 s = viewport_size;
      gloco->default_camera->camera.set_ortho(
        fan::vec2(-s.x, s.x) / viewport_settings.zoom,
        fan::vec2(-s.y, s.y) / viewport_settings.zoom
      );

      //gloco->default_camera->camera.set_camera_zoom(viewport_settings.zoom);
      gloco->default_camera->viewport.set(viewport_pos, viewport_size, window_size);
      viewport_settings.size = ImGui::GetContentRegionAvail();
      gloco->default_camera->viewport.set(viewport_pos, viewport_size, window_size);
      if (t2i.current) {
        ImGui::Image(
          (void*)(intptr_t)t2i.current->ti.image->get_texture(),
          (t2i.current->ti.size / 4).max(fan::vec2(64)),
          t2i.current->ti.position / t2i.current->ti.image->size,
          t2i.current->ti.position / t2i.current->ti.image->size + t2i.current->ti.size / t2i.current->ti.image->size
        );
      }
      if (editor_settings.hovered && 
        brush.mode == brush_t::mode_e::create && 
        t2i.current && ImGui::IsMouseClicked(0)
        ) {
        images.push_back(std::make_unique<editor_t::global_t>(this, fan::graphics::sprite_t{ {
            .position = fan::vec3(get_mouse_position(), 0),
            .size = (t2i.current->ti.size / 4).max(fan::vec2(64)) * brush.size,
            .blending = true
        } }));
        images.back().get()->children[0].set_tp(&t2i.current->ti);
      }
      editor_settings.hovered = ImGui::IsWindowHovered();
      ImGui::End();
      ImGui::Begin("Sprites");
      t2i.render(this);
      ImGui::End();
      ImGui::Begin("Settings");
      {
        int idx = (int)brush.mode;
        if (ImGui::Combo("type", &idx, brush.mode_names, std::size(brush.mode_names))) {
          brush.mode = (brush_t::mode_e)idx;
          if (brush.mode == brush_t::mode_e::create) {
            visual_place = fan::graphics::sprite_t{ {
                .position = fan::vec3(fan::vec2(-999999), 0xfff),
                .size = (t2i.current->ti.size / 4).max(fan::vec2(64)) * brush.size,
                .blending = true
            }};
            visual_place.set_tp(&t2i.current->ti);
          }
          else if (brush.mode == brush_t::mode_e::modify) {
            visual_place.erase();
          }
        }
        {
          static bool lock = false;
          ImGui::Checkbox("lock axis", &lock);
          fan::vec2 size = brush.size;
          if (ImGui::DragFloat2("size", size.data(), 0.1)) {
            fan::vec2 prev = brush.size;
            if (lock) {
              float ratio = prev.x / prev.y;
              if (size.x != prev.x) {
                size.y = size.x / ratio;
              }
              else if (size.y != prev.y) {
                size.x = size.y * ratio;
              }
            }
            brush.size = size;
            visual_place.set_size((t2i.current->ti.size / 4).max(fan::vec2(64)) * brush.size);
          }
        }
      }
      ImGui::End();
      ImGui::Begin("Image settings");
      if (current_shape) {
        static bool lock = false;
        ImGui::Checkbox("lock axis", &lock);
        {
          fan::vec3 position = current_shape->children[0].get_position();
          if (ImGui::DragFloat3("position", position.data())) {
            fan::vec3 prev_pos = current_shape->children[0].get_position();
            if (lock) {
              fan::vec2 off = position - prev_pos;
              if (off.x) {
                position.y += off.x;
              }
              if (off.y) {
                position.x += off.y;
              }
            }
            current_shape->set_position(position);
          }
        }
        {
          fan::vec2 size = current_shape->get_size();
          if (ImGui::DragFloat2("size", size.data())) {
            fan::vec2 prev = current_shape->get_size();
            if (lock) {
              float ratio = prev.x / prev.y;
              if (size.x != prev.x) {
                size.y = size.x / ratio;
              }
              else if (size.y != prev.y) {
                size.x = size.y * ratio;
              }
            }
            current_shape->set_size(size);
          }
        }
      }
      ImGui::End();
    };
  }

  struct {
    f32_t zoom = 1;
    bool move = false;
    fan::vec2 pos = 0;
    fan::vec2 size = 0;
    fan::vec2 offset = 0;
  }viewport_settings;

  struct {
    bool hovered = false;
  }editor_settings;

  struct brush_t {
    static constexpr const char* mode_names[] = { "Create", "Modify" };
    enum class mode_e : uint8_t{
      create,
      modify
    };
    mode_e mode = mode_e::create;

    fan::vec2 size = 1;
    int tile_size = 1;
    f32_t angle = 0;
    f32_t depth = 1;
    int jitter = 0;
    f32_t jitter_chance = 0.33;
    fan::string id;
    fan::color color = fan::color(1);
  }brush;

  uint32_t current_z = 0;
  std::vector<std::unique_ptr<global_t>> images;
  fan::graphics::imgui_element_t element;
  texturepack_2_imgui_t t2i;
  global_t* current_shape = 0;
  loco_t::shape_t visual_place;
};

int main() {
  loco_t loco;

  loco_t::texturepack_t tp;
  tp.open_compiled("texture_packs/TexturePack");

  editor_t editor(&tp);


  loco.loop([&] {

  });

  return 0;
}
