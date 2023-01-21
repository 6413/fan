struct stage_maker_shape_format {

  struct shape_type_t {
    using _t = uint16_t;
    static constexpr _t button = 0;
    static constexpr _t sprite = 1;
    static constexpr _t text = 2;
  };

  struct shape_button_t {
    fan::vec3 position;
    fan::vec2 size;
    f32_t font_size;
    fan_2d::graphics::gui::theme_t theme;
  };
  struct shape_sprite_t {
    fan::vec3 position;
    fan::vec2 size;
  };
  struct shape_text_t {
    fan::vec3 position;
    f32_t size;
  };
};