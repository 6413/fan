// can have like sensor, etc
enum struct mesh_property_t : uint8_t {
  none = 0,
  collider,
  sensor,
  light,
  size
};

fan_enum_string(
  actions_e,
  none,
  open_model,
  callback // done using id
);

struct tile_t {
  fan::vec3i position;
  fan::vec2i size;
  fan::vec3 angle;
  fan::color color;
  uint64_t image_hash;
  mesh_property_t mesh_property = mesh_property_t::none;
  fan::string id;

  // actions
  actions_e action = actions_e::none;
  uint16_t key = fan::key_invalid;
  int key_state = (int)fan::keyboard_state::press;
  std::vector<std::string> object_names;
};