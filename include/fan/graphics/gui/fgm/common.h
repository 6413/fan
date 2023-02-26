struct stage_maker_shape_format_0_1_0 {
  struct shape_button_t {
    fan::vec3 position;
    fan::vec2 size;
    f32_t font_size;
    uint32_t id;
  };
  struct shape_sprite_t {
    fan::vec3 position;
    fan::vec2 size;
    f32_t parallax_factor;
  };
  struct shape_text_t {
    fan::vec3 position;
    f32_t size;
  };
  struct shape_hitbox_t {
    fan::vec3 position;
    fan::vec2 size;
    loco_t::vfi_t::shape_type_t shape_type;
    uint32_t id;
  };
};

struct stage_maker_shape_format_0_1_1 {
  struct shape_button_t {
    fan_masterpiece_make(
      (fan::vec3) position,
      (fan::vec2) size,
      (f32_t) font_size,
      (fan::string) text,
      (fan::string) id
    )
  };
  struct shape_sprite_t {
    fan_masterpiece_make(
      (fan::vec3) position,
      (fan::vec2) size,
      (f32_t) parallax_factor,
      (fan::string) texturepack_name,
      (fan::string) id
    )
  };
  struct shape_text_t {
    fan_masterpiece_make(
      (fan::vec3) position,
      (f32_t) size,
      (fan::string) text,
      (fan::string) id
    );
  };
  struct shape_hitbox_t {
    fan_masterpiece_make(
      (fan::vec3) position,
      (fan::vec2) size,
      (loco_t::vfi_t::shape_type_t) vfi_type,
      (fan::string) id
    )
  };
  struct shape_mark_t {
    fan_masterpiece_make(
      (fan::vec3)position,
      (fan::string)id
    )
  };
};

using stage_maker_shape_format = stage_maker_shape_format_0_1_1;

static constexpr uint32_t version_010 = 10;
static constexpr uint32_t version_011 = 11;

static constexpr uint32_t stage_maker_format_version = version_011;

static fan::string shape_to_string(const auto& shape) {
  fan::string str;
  shape.iterate_masterpiece([&str](const auto& field) {
    if constexpr (std::is_same_v<std::remove_const_t<std::remove_reference_t<decltype(field)>>, fan::string>) {
      uint64_t string_length = field.size();
      str.append((char*)&string_length, sizeof(string_length));
      str.append(field);
    }
    else {
      str.append((char*)&field, sizeof(field));
    }
  });
  return str;
}

fan::string shapes_to_string(auto&&... shapes) {
  fan::string str;
  ((str += shape_to_string(shapes)), ...);
  return str;
}