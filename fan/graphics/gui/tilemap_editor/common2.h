// can have like sensor, etc
enum struct mesh_property_t : uint8_t {
  none = 0,
  physics_shape,
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
  std::string image_name;
  mesh_property_t mesh_property = mesh_property_t::none;
  fan::string id;
  uint32_t flags = 0;

  // actions
  actions_e action = actions_e::none;
  uint16_t key = fan::key_invalid;
  int key_state = (int)fan::keyboard_state::press;
  std::vector<std::string> object_names;
};

struct physics_shapes_t {
  struct type_e {
    enum {
      box,
      circle
    };
  };
  loco_t::shape_t visual;
  uint8_t type = type_e::box;
  uint8_t body_type = fan::physics::body_type_e::static_body;
  bool draw = false;
  fan::physics::shape_properties_t shape_properties;
  std::string id;
};