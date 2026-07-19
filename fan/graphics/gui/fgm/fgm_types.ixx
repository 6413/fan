#if defined(FAN_GUI)

export module fan.graphics.editor:fgm_types;

import std;

import fan.types;
import fan.types.vector;
import fan.types.color;
#if defined(FAN_JSON)
  import fan.types.json;
#endif
import fan.graphics.common_context;
import fan.graphics.shapes;
import fan.graphics.gui;

export namespace fan::graphics::editor {
  struct shapes_t {
    struct global_t : fan::graphics::gui::imgui_element_t {
      global_t() = default;

      template <typename T>
      global_t(std::uint16_t shape_type, const T& obj, f32_t& current_z, global_t*& current_shape, bool shape_add = true) {
        T temp = obj;
        this->shape_type = shape_type;
        if (shape_add) {
          temp.set_position(fan::vec3(fan::vec2(temp.get_position()), current_z++));
        }
        children.push_back(temp);
        current_shape = this;
      }

      fan::vec3 get_position() const { return children.empty() ? fan::vec3(0) : children[0].get_position(); }
      fan::vec2 get_size() const { return children.empty() ? fan::vec2(0) : children[0].get_size(); }
      fan::color get_color() const { return children.empty() ? fan::color(1) : children[0].get_color(); }
      
      void set_position(const fan::vec3& position, bool modify_depth = true) {
        if (children.empty()) return;
        fan::vec2 delta = fan::vec2(position - children[0].get_position());
        for (auto i{0uz}; i < children.size(); ++i) {
          auto& child = children[0];
          fan::vec3 cp = child.get_position();
          fan::vec3 new_pos = fan::vec3(fan::vec2(cp) + delta, i > 1 ? cp.z : position.z);
          child.set_position(new_pos);
        }
      }

      void set_size(const fan::vec2& size) {
        if (children.empty()) return;
        fan::vec2 offset = size - children[0].get_size();
        for (auto& child : children) {
          child.set_size(child.get_size() + offset);
        }
      }

      void set_color(const fan::color& c) {
        for (auto& child : children) {
          child.set_color(c);
        }
      }

      void enable_highlight() {}
      void disable_highlight() {}

      std::vector<fan::graphics::shape_t> children;
      std::string id;
      std::uint32_t group_id = 0;
      std::uint16_t shape_type = 0;
      int material_id = -1;
      std::uint8_t material_type = 0; // 0 = textured, 1 = solid color
      fan::graphics::image_t original_image;
      fan::color original_color = fan::colors::white;
      
      struct physics_properties_t {
        bool enabled = false;
        bool is_sensor = false;
        int body_type = 0;
        int shape_type = 0; // 0=box, 1=circle, 2=capsule
        int collision_shape = 0; // 0 = Auto (Bounds), 1 = Custom Segments
        fan::vec2 hitbox_size = 1;
        f32_t mass = 1.0f;
        f32_t friction = 0.2f;
        f32_t restitution = 0.0f;
        std::vector<fan::vec2> segment_points;

#if defined(FAN_JSON)
        fan::json to_json() const {
          fan::json p = fan::json::object();
          p["enabled"] = enabled;
          p["body_type"] = body_type;
          p["shape_type"] = shape_type;
          p["collision_shape"] = collision_shape;
          p["hitbox_size"] = hitbox_size;
          p["mass"] = mass;
          p["friction"] = friction;
          p["restitution"] = restitution;
          p["is_sensor"] = is_sensor;
          fan::json pts = fan::json::array();
          for (auto& pt : segment_points) { fan::json j; j = pt; pts.push_back(j); }
          p["segment_points"] = pts;
          return p;
        }

        void from_json(const fan::json& p) {
          enabled = p.value("enabled", false);
          body_type = p.value("body_type", 0);
          shape_type = p.value("shape_type", 0);
          collision_shape = p.value("collision_shape", 0);
          hitbox_size = p.value("hitbox_size", fan::vec2(1));
          mass = p.value("mass", 1.0f);
          friction = p.value("friction", 0.2f);
          restitution = p.value("restitution", 0.0f);
          is_sensor = p.value("is_sensor", false);
          segment_points.clear();
          if (p.contains("segment_points")) {
            for (auto& pt : p["segment_points"]) {
              segment_points.push_back(pt);
            }
          }
        }
#endif
      } physics;

      struct light_properties_t {
#if defined(FAN_JSON)
        fan::json to_json() const {
          return {
            {"enable_flicker", enable_flicker},
            {"flicker_speed", flicker_speed},
            {"flicker_min", flicker_min},
            {"flicker_max", flicker_max},
            {"ease_type", ease_type}
          };
        }

        void from_json(const fan::json& p) {
          enable_flicker = p.value("enable_flicker", false);
          flicker_speed = p.value("flicker_speed", 1.0f);
          flicker_min = p.value("flicker_min", 0.5f);
          flicker_max = p.value("flicker_max", 1.0f);
          ease_type = p.value("ease_type", 2);
        }
#endif

        bool enable_flicker = false;
        f32_t flicker_speed = 1.0f;
        f32_t flicker_min = 0.5f;
        f32_t flicker_max = 1.0f;
        int ease_type = 2;
      } light_props;

      struct dynamic_properties_t {
#if defined(FAN_JSON)
        fan::json to_json() const {
          fan::json j;
          j["target_color"] = target_color;
          j["variance_speed"] = variance_speed;
          j["ease_type"] = ease_type;
          return j;
        }

        void from_json(const fan::json& p) {
          target_color = p.value("target_color", fan::colors::white);
          variance_speed = p.value("variance_speed", 1.0f);
          ease_type = p.value("ease_type", 2);
        }
#endif
        fan::color target_color = fan::colors::white;
        f32_t variance_speed = 1.0f;
        int ease_type = 2;
        
        // Runtime cache
        fan::color base_color = fan::colors::white;
      } dynamic_props;
    };
  };
}

#endif