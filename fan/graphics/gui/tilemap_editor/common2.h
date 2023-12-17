// can have like sensor, etc
enum struct mesh_property_t : uint8_t {
  none = 0,
  collider,
  sensor,
  light,
  size
};

struct tile_t {
  fan::vec3i position;
  fan::vec2i size;
  fan::vec3 angle;
  fan::color color;
  uint64_t image_hash;
  mesh_property_t mesh_property = mesh_property_t::none;
  fan::string id;
};