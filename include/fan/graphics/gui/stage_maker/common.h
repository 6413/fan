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

using stage_maker_shape_format = stage_maker_shape_format_0_1_0;

static constexpr uint32_t version_010 = 10;

static constexpr uint32_t stage_maker_format_version = version_010;