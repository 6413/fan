export module fan.graphics.editor:properties_ui;

import std;

import fan.math;
import fan.graphics.gui.base;
import fan.graphics.loco;
import :fgm_types;

export namespace fan::graphics::editor {
  struct properties_ui_t {

    template <typename FGM_T, typename GlobalT>
    static void open_properties(FGM_T& fgm, GlobalT* shape) {
      using namespace fan::graphics;

      std::string shape_str = std::string("Shape name:") + std::string(fan::graphics::shape_names[shape->children[0].get_shape_type()]);
      gui::text(shape_str);

      fan::vec3 pos = shape->get_position();
      if (gui::button("R##pos")) pos = 0;
      gui::same_line();
      if (gui::drag("shape position", &pos, 0.1f)) {
        pos.z = (int)pos.z;
        if (!fan::window::is_key_down(fan::key_left_shift) && fgm.snap > 0.0f) {
          pos.x = std::round(pos.x / fgm.snap) * fgm.snap;
          pos.y = std::round(pos.y / fgm.snap) * fgm.snap;
        }
        shape->set_position(pos);
      }

      fan::vec2 size = shape->get_size();
      if (gui::button("R##size")) size = 128;
      gui::same_line();
      if (gui::drag("shape size", &size, 0.1f)) {
        shape->set_size(size);
      }

      auto sti = shape->children[0].get_shape_type();
      if (sti == fan::graphics::shapes::shape_type_t::particles) {
        gui::shape_properties(shape->children[0]);
      }
      else {
        fan::color c = shape->get_color();
        if (gui::button("R##col")) c = fan::colors::white;
        gui::same_line();
        if (gui::color_edit4("color", &c)) {
          shape->set_color(c);
        }

        fan::vec3 angle = shape->children[0].get_angle();
        angle.x = fan::math::degrees(angle.x);
        angle.y = fan::math::degrees(angle.y);
        angle.z = fan::math::degrees(angle.z);
        if (gui::button("R##ang")) angle = 0;
        gui::same_line();
        if (gui::drag("shape angle", &angle)) {
          angle = fan::math::radians(angle);
          shape->children[0].set_angle(angle);
        }
      }

      fan::vec2 tc_position = shape->children[0].get_tc_position();
      if (gui::button("R##tcpos")) tc_position = 0;
      gui::same_line();
      if (gui::drag("tc position", &tc_position, 0.1f)) {
        shape->children[0].set_tc_position(tc_position);
      }

      fan::vec2 tc_size = shape->children[0].get_tc_size();
      if (gui::button("R##tcsize")) tc_size = 1;
      gui::same_line();
      if (gui::drag("tc size", &tc_size, 0.1f)) {
        shape->children[0].set_tc_size(tc_size);
      }

      std::string& id = shape->id;
      std::string str = id;
      if (gui::input_text("id", &str)) {
        if (gui::is_item_deactivated_after_edit() && !fgm.id_exists(str)) {
          id = str;
        }
      }

      std::string id_str = std::to_string(shape->group_id);
      str = id_str;
      if (gui::input_text("group id", &str)) {
        if (gui::is_item_deactivated_after_edit()) {
          shape->group_id = std::stoul(str);
        }
      }

      if (gui::tree_node("Physics Properties")) {
        gui::checkbox("Enable Physics", &shape->physics.enabled);
        if (shape->physics.enabled) {
          const char* body_types[] = {"Static", "Kinematic", "Dynamic"};
          gui::combo("Body Type", &shape->physics.body_type, body_types, 3);
          gui::drag("Mass", &shape->physics.mass, 0.1f);
          gui::drag("Friction", &shape->physics.friction, 0.05f);
          gui::drag("Restitution", &shape->physics.restitution, 0.05f);
          gui::checkbox("Is Sensor", &shape->physics.is_sensor);
        }
        gui::tree_pop();
      }

      if (gui::tree_node("Material")) {
        const char* material_names[] = {"Textured", "Solid Color"};
        int mt = shape->material_type;
        if (gui::combo("type", &mt, material_names, 2)) {
          shape->material_type = (uint8_t)mt;
          fgm.apply_material(shape);
        }
        if (shape->material_type == 1) {
          fan::color c = shape->get_color();
          if (gui::color_edit4("solid color", &c)) {
            shape->set_color(c);
          }
        }
        gui::tree_pop();
      }

      switch (shape->children[0].get_shape_type()) {
        case fan::graphics::shapes::shape_type_t::unlit_sprite:
        case fan::graphics::shapes::shape_type_t::sprite:
        {
          if (shape->material_type == 0) {
            fan::graphics::gui::render_texture_property(shape->children[0], 0, "Base texture", fgm.content_browser.asset_path);
            fan::graphics::gui::render_texture_property(shape->children[0], 1, "Normal map", fgm.content_browser.asset_path);
            fan::graphics::gui::render_texture_property(shape->children[0], 2, "Specular map", fgm.content_browser.asset_path);
            fan::graphics::gui::render_texture_property(shape->children[0], 3, "Occlusion map", fgm.content_browser.asset_path);

            auto current_settings = gloco()->image_get_settings(shape->children[0].get_image());

            int current_image_filter = current_settings.min_filter;
            static auto filter_names = fan::graphics::image_filter_e::get_names();
            if (gui::combo("image filter", &current_image_filter, filter_names.data(), (int)filter_names.size())) {
              current_settings.min_filter = current_image_filter;
              current_settings.mag_filter = current_image_filter;
              gloco()->image_set_settings(shape->children[0].get_image(), current_settings);
              if (shape->children[0].get_images()[0].iic() == false) {
                gloco()->image_set_settings(shape->children[0].get_images()[0], current_settings);
              }
            }

            int current_address_mode = current_settings.visual_output;
            static auto address_mode_names = fan::graphics::image_sampler_address_mode.get_names();
            if (gui::combo("address mode", &current_address_mode, address_mode_names.data(), address_mode_names.size())) {
              current_settings.visual_output = current_address_mode;
              gloco()->image_set_settings(shape->children[0].get_image(), current_settings);
              if (shape->children[0].get_images()[0].iic() == false) {
                gloco()->image_set_settings(shape->children[0].get_images()[0], current_settings);
              }
            }

            std::string& current = shape->children[0].get_image_data().image_path;
            str = current;
            if (gui::input_text("image path", &str)) {
              if (gui::is_item_deactivated_after_edit()) {
                fan::graphics::texture_pack::ti_t ti;
                if (gloco()->texture_pack.qti(str.c_str(), &ti)) {
                  gui::print("failed to load texture:", str);
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
          }
          break;
        }
      }
    }

    template <typename FGM_T, typename GlobalT>
    static void render(FGM_T& fgm, GlobalT*& current_shape) {
      using namespace fan::graphics;
      if (!current_shape) return;
      if (gui::begin("Properties")) {
        if (current_shape != nullptr) {
          open_properties(fgm, current_shape);

          gui::new_line();
          gui::begin_child("properties_animations", 0, 1);
          gui::text("Animations");
          auto& shape = current_shape->children[0];
          fan::graphics::sprite_sheet_shape_id_t* shape_animation_nr = 0;
          if (shape.get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
            fan::graphics::g_shapes->visit_shape_draw_data(shape.NRI, [&](auto& props) {
              if constexpr (requires { props.sprite_sheet_data.shape_sprite_sheets; }) {
                shape_animation_nr = &props.sprite_sheet_data.shape_sprite_sheets;
              }
            });
          }
          if (shape_animation_nr == nullptr) {
            goto g_end_animations;
          }
          {
            bool animation_changed = fgm.animations_application.render("CONTENT_BROWSER_ITEMS", *shape_animation_nr);
            if (shape.get_shape_type() == fan::graphics::shapes::shape_type_t::sprite && fgm.animations_application.current_animation_nr) {
              if (fgm.animations_application.current_animation_nr && fgm.animations_application.current_animation_shape_nr == *shape_animation_nr) {
                fan::graphics::g_shapes->visit_shape_draw_data(shape.NRI, [&](auto& props) {
                  if constexpr (requires { props.sprite_sheet_data.current_sprite_sheet; }) {
                    props.sprite_sheet_data.current_sprite_sheet = fgm.animations_application.current_animation_nr;
                  }
                });
              }
            }
            if (*shape_animation_nr && fgm.animations_application.current_animation_shape_nr == *shape_animation_nr && (fgm.animations_application.toggle_play_animation || fgm.animations_application.play_animation || animation_changed)) {
              if (shape.get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
                auto& anim = fan::graphics::get_sprite_sheet(fgm.animations_application.current_animation_nr);
                if (animation_changed && fgm.animations_application.current_animation_nr && fgm.animations_application.play_animation) {
                  if (gloco()->is_image_valid(shape.get_image()) == false) {
                    shape.set_tc_position(0);
                    shape.set_tc_size(1);
                  }
                }
                else if (anim.selected_frames.size()) {
                }

                auto& current_shape_anim = shape.get_sprite_sheet();
                if ((fgm.animations_application.toggle_play_animation || animation_changed) &&
                  fgm.animations_application.play_animation &&
                  current_shape_anim.selected_frames.size()) {
                  shape.play_sprite_sheet();
                }

                if (fgm.animations_application.toggle_play_animation && !fgm.animations_application.play_animation) {
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
    }
  };
}
