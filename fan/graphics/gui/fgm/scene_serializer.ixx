module;

export module fan.graphics.editor:scene_serializer;
import std;

import fan.graphics.loco;
import fan.graphics.gui.base;
import fan.graphics.gui.text_logger;
import fan.graphics;
import fan.io.file;

import :fgm_types;
import :animation_system;

export namespace fan::graphics::editor {
  struct scene_serializer_t {
    template <typename FGM_T>
    static void save(FGM_T& fgm, std::string filename) {
      if (filename.empty()) {
        gui::print_error("filename is empty. save file from 'File/Save as'");
        return;
      }
      if (!filename.ends_with(".json")) filename += ".json";

      fan::json out;
      out["version"] = 1;

      serialize_environment(out);
      serialize_animations(fgm, out);
      serialize_shapes(fgm, out);

      out.find_and_iterate("image_path", [&filename](fan::json& value) {
        value = fan::io::file::relative_path(value.get<std::string>(), filename).generic_string();
      });

      fan::io::file::write(filename, out.dump(2), std::ios_base::binary);
      fgm.previous_filename = filename;
      gui::print_success("File saved to " + std::filesystem::absolute(filename).generic_string());
    }

    template <typename FGM_T>
    static void load(FGM_T& fgm, const std::string& filename) {
      fgm.previous_filename = filename;
      std::string in;
      fan::io::file::read(filename, &in);
      fan::json json_in = fan::json::parse(in);

      deserialize_environment(json_in);
      deserialize_animations(fgm, json_in);
      deserialize_shapes(fgm, json_in, filename);
    }

  private:
    static void serialize_environment(fan::json& out) {
      if (gloco()->renderer_state.lighting.ambient != fan::graphics::lighting_t().ambient) {
        out["lighting.ambient"] = gloco()->renderer_state.lighting.ambient;
      }
      if (gloco()->renderer_state.clear_color != fan::colors::black) {
        out["clear_color"] = gloco()->renderer_state.clear_color;
      }
    }

    template <typename FGM_T>
    static void serialize_animations(FGM_T& fgm, fan::json& out) {
      auto animations_json = fan::graphics::sprite_sheet_serialize();
      if (!animations_json.empty()) out.update(animations_json, true);
      
      fan::json shape_anims = fan::json::array();
      for (const auto& [shape, anim] : fgm.shape_sprite_sheets) {
        if (anim.keyframes.empty()) continue;
        fan::json anim_entry;
        int shape_index = 0;
        for (auto& ptr : fgm.shape_list) {
          if (ptr.get() == shape) {
            anim_entry["shape_index"] = shape_index;
            anim_entry["animation"] = anim.serialize();
            shape_anims.push_back(anim_entry);
            break;
          }
          shape_index++;
        }
      }
      if (!shape_anims.empty()) out["shape_keyframe_animations"] = shape_anims;
    }

    template <typename FGM_T>
    static void serialize_shapes(FGM_T& fgm, fan::json& out) {
      fan::json shapes = fan::json::array();
      for (auto& instance : fgm.shape_list) {
        fan::json shape_json;
        fan::graphics::shape_to_json(instance->children[0], &shape_json);

        typename shapes_t::global_t defaults;
        if (instance->id != defaults.id) shape_json["id"] = instance->id;
        if (instance->group_id != defaults.group_id) shape_json["group_id"] = instance->group_id;
        if (instance->physics.enabled) {
          shape_json["physics"] = instance->physics.to_json();
        }

        if (auto found = fgm.shape_original_json.find(instance.get()); found != fgm.shape_original_json.end()) {
          shape_json.preserve_unknown(found->second);
        }
        shapes.push_back(shape_json);
      }
      out["shapes"] = shapes;
    }

    static void deserialize_environment(const fan::json& json_in) {
      if (json_in.contains("lighting.ambient")) gloco()->renderer_state.lighting.ambient = json_in["lighting.ambient"];
      if (json_in.contains("clear_color")) gloco()->renderer_state.clear_color = json_in["clear_color"];
    }

    template <typename FGM_T>
    static void deserialize_animations(FGM_T& fgm, fan::json& json_in) {
      if (json_in.contains("animations")) fan::graphics::sprite_sheets_parse(fgm.previous_filename, json_in);
      if (json_in.contains("shape_keyframe_animations")) {
        for (const auto& anim_entry : json_in["shape_keyframe_animations"]) {
          int shape_index = anim_entry["shape_index"].get<int>();
          shape_keyframe_animation_t anim;
          anim.deserialize(anim_entry["animation"]);

          if (shape_index >= 0 && shape_index < (int)fgm.shape_list.size()) {
            fgm.shape_sprite_sheets[fgm.shape_list[shape_index].get()] = anim;
          }
        }
      }
    }

    template <typename FGM_T>
    static void deserialize_shapes(FGM_T& fgm, fan::json& json_in, const std::string& filename) {
      json_in.find_and_iterate("image_path", [&filename](fan::json& value) {
        std::filesystem::path json_path = std::filesystem::absolute(std::filesystem::path(filename)).parent_path();
        value = (json_path / std::filesystem::path(value.get<std::string>())).generic_string();
      });

      fan::graphics::shape_deserialize_t iterator;
      fan::graphics::shape_t shape;
      fgm.current_z = 0;
      std::string shapes_key = json_in.contains("tiles") ? "tiles" : "shapes";
      
      auto shape_object = json_in;
      bool is_object = true;
      if (json_in.contains(shapes_key.c_str())) {
        shape_object = json_in[shapes_key];
        is_object = false;
      } else if (!json_in.contains("shape")) {
        gui::print_error("failed to parse .json file");
        return;
      }

      while (iterator.iterate(shape_object, &shape)) {
        shape.set_camera(fgm.render_view.camera);
        shape.set_viewport(fgm.render_view.viewport);
        fgm.current_z = std::max(fgm.current_z, shape.get_position().z);
        auto& node = fgm.shape_list.emplace_back();
        
        switch (shape.get_shape_type()) {
          case fan::graphics::shapes::shape_type_t::sprite:
          case fan::graphics::shapes::shape_type_t::unlit_sprite:
          case fan::graphics::shapes::shape_type_t::particles:
          case fan::graphics::shapes::shape_type_t::rectangle: {
            node = std::make_unique<shapes_t::global_t>(shape.get_shape_type(), shape, fgm.current_z, fgm.current_shape, false);
            if (shape.get_shape_type() == fan::graphics::shapes::shape_type_t::sprite || shape.get_shape_type() == fan::graphics::shapes::shape_type_t::unlit_sprite) {
              fgm.load_tp(node.get());
              node->children[0].get_image_data().image_path = shape.get_image_data().image_path;
            }
            break;
          }
          case fan::graphics::shapes::shape_type_t::light: {
            node = std::make_unique<shapes_t::global_t>(shape.get_shape_type(), shape, fgm.current_z, fgm.current_shape, false);
            node->children.push_back(fan::graphics::circle_t {{
              .render_view = &fgm.render_view,
              .position = shape.get_position(),
              .radius = shape.get_size().x,
              .color = shape.get_color(),
              .blending = true
            }});
            break;
          }
        }

        if (!is_object) {
          const auto shape_json = *(iterator.data.it - 1);
          if (shape_json.contains("id")) node->id = shape_json["id"].get<std::string>();
          if (shape_json.contains("group_id")) node->group_id = shape_json["group_id"].get<uint32_t>();
          if (shape_json.contains("physics")) {
            node->physics.from_json(shape_json["physics"]);
          }
          fgm.shape_original_json[node.get()] = shape_json;
        }
      }
      ++fgm.current_z;
    }
  };
}