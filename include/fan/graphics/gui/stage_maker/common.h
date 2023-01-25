struct stage_maker_shape_format {

  struct shape_button_t {
    fan::vec3 position;
    fan::vec2 size;
    f32_t font_size;
    fan_2d::graphics::gui::theme_t theme;
    uint32_t id;
  };
  struct shape_sprite_t {
    fan::vec3 position;
    fan::vec2 size;
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