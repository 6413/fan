#if defined (FAN_2D)

module;

export module fan.graphics.scene;

import std;
import fan;
import fan.graphics;
import fan.graphics.physics_shapes;

import fan.graphics.material;

export namespace fan::graphics {

  struct scene_t {
    struct shape_node_t {
      std::string id;
      std::uint16_t type;
      std::any physics_shape;
    };

    std::vector<shape_node_t> nodes;

    void load(const std::string& path, const std::source_location sl = std::source_location::current()) {
      std::string json_str;
      if (fan::io::file::read(path, &json_str, sl)) {
        fan::print_error("Failed to load scene file:", path);
        return;
      }

      fan::json map_data;
      try {
        map_data = fan::json::parse(json_str);
      } catch (const std::exception& e) {
        fan::print_error("Failed to parse scene JSON:", e.what());
        return;
      }

      std::unordered_map<int, material_t> materials;
      if (map_data.contains("materials")) {
        for (const auto& mat_json : map_data["materials"]) {
          material_t mat;
          if (mat_json.contains("id")) mat.id = mat_json["id"].get<int>();
          if (mat_json.contains("color")) mat.color = mat_json["color"];
          if (mat_json.contains("images")) mat.images = mat_json["images"];
          if (mat_json.contains("material_type")) mat.material_type = mat_json["material_type"].get<int>();
          materials[mat.id] = mat;
        }
      }

      fan::graphics::shape_deserialize_t iterator;
      fan::graphics::shape_t shape;

      if (!map_data.contains("shapes")) {
        return;
      }

      while (iterator.iterate(map_data["shapes"], &shape)) {
        auto& current_json = *iterator.current_json;
        std::string id = current_json.value("id", "");
        
        std::uint8_t body_type = fan::physics::body_type_e::static_body;
        if (current_json.contains("physics")) {
          body_type = current_json["physics"].value("body_type", (int)fan::physics::body_type_e::static_body);
        }

        std::uint16_t shape_type = shape.get_shape_type();

        fan::color final_color = shape.get_color();
        fan::json final_images;
        int mat_type = 0;

        if (current_json.contains("material_id")) {
          int m_id = current_json["material_id"].get<int>();
          if (materials.contains(m_id)) {
            const auto& mat = materials[m_id];
            final_color = mat.color;
            final_images = mat.images;
            mat_type = mat.material_type;
          }
        } else {
          // Fallback to flat JSON
          if (current_json.contains("material_type") && current_json["material_type"].get<int>() == 1) {
            mat_type = 1;
          }
          if (current_json.contains("images")) final_images = current_json["images"];
        }

        if (shape_type == fan::graphics::shapes::shape_type_t::rectangle) {
          fan::graphics::physics::rectangle_t body{
            fan::graphics::physics::rectangle_t::properties_t{
              .position = shape.get_position(),
              .size = shape.get_size(),
              .color = final_color,
              .body_type = body_type
            }
          };
          if (shape.get_angle() != 0) body.set_angle(shape.get_angle());
          nodes.push_back(shape_node_t{id, shape_type, std::make_any<fan::graphics::physics::rectangle_t>(std::move(body))});
        }
        else if (shape_type == fan::graphics::shapes::shape_type_t::circle) {
          fan::graphics::physics::circle_t body{
            fan::graphics::physics::circle_t::properties_t{
              .position = shape.get_position(),
              .radius = shape.get_size().x,
              .color = final_color,
              .body_type = body_type
            }
          };
          if (shape.get_angle() != 0) body.set_angle(shape.get_angle());
          nodes.push_back(shape_node_t{id, shape_type, std::make_any<fan::graphics::physics::circle_t>(std::move(body))});
        }
        else if (shape_type == fan::graphics::shapes::shape_type_t::capsule) {
          fan::graphics::physics::capsule_t body{
            fan::graphics::physics::capsule_t::properties_t{
              .position = shape.get_position(),
              .center0 = fan::vec2(0, -shape.get_size().y),
              .center1 = fan::vec2(0, shape.get_size().y),
              .radius = shape.get_size().x,
              .color = final_color,
              .body_type = body_type
            }
          };
          if (shape.get_angle() != 0) body.set_angle(shape.get_angle());
          nodes.push_back(shape_node_t{id, shape_type, std::make_any<fan::graphics::physics::capsule_t>(std::move(body))});
        }
        else if (shape_type == fan::graphics::shapes::shape_type_t::sprite) {
          fan::graphics::physics::sprite_t::properties_t p{
            .position = shape.get_position(),
            .size = shape.get_size(),
            .color = final_color,
            .body_type = body_type
          };
          if (mat_type == 1) {
            static fan::graphics::image_t white_texture = fan::graphics::image_create(fan::colors::white);
            p.image = white_texture;
          } else if (!final_images.empty() && final_images.is_array() && final_images.size() > 0) {
            auto first_image = final_images[0];
            if (first_image.contains("image_path")) {
              std::string path = first_image["image_path"].get<std::string>();
              if (!path.empty()) {
                p.image = fan::graphics::image_t(path);
              }
            }
          }
          fan::graphics::physics::sprite_t body{p};
          if (shape.get_angle() != 0) body.set_angle(shape.get_angle());
          nodes.push_back(shape_node_t{id, shape_type, std::make_any<fan::graphics::physics::sprite_t>(std::move(body))});
        }
      }
    }

    void clear() {
      nodes.clear();
    }

    template <typename T>
    T* get_physics_body(const std::string& id) {
      for (auto& node : nodes) {
        if (node.id == id && node.physics_shape.type() == typeid(T)) {
          return std::any_cast<T>(&node.physics_shape);
        }
      }
      return nullptr;
    }
  };

}

#endif