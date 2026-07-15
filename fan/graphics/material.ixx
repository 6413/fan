export module fan.graphics.material;

import std;
import fan.types.color;
#if defined(FAN_JSON)
import fan.types.json;
#endif

export namespace fan::graphics {

  struct material_t {
    int id = 0;
    fan::color color = fan::colors::white;
#if defined(FAN_JSON)
    fan::json images;
#endif
    int material_type = 0;

#if defined(FAN_JSON)
    bool operator==(const material_t& other) const {
      return color == other.color && images == other.images && material_type == other.material_type;
    }
#endif
  };

  struct material_system_t {
    std::unordered_map<int, material_t> materials;
    int next_material_id = 0;

    int add(const material_t& mat) {
      int id = next_material_id++;
      materials[id] = mat;
      materials[id].id = id;
      return id;
    }
  };

}
