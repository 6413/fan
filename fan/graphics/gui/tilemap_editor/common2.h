// can have like sensor, etc
enum struct mesh_property_t : uint8_t {
  none = 0,
  collision_solid = 1,
  size
};

struct tile_t {
  fan::vec3i position;
  f32_t angle;
  uint64_t image_hash;
  mesh_property_t mesh_property;
};