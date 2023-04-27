struct stage_maker_shape_format_0_1_1 {

  struct shape_button_t {
    loco_t::button_t::properties_t get_properties(
      loco_t::viewport_t& viewport,
      loco_t::camera_t& camera,
      loco_t::theme_t& theme
      ) {
      loco_t::button_t::properties_t p;
      p.viewport = &viewport;
      p.camera = &camera;
      p.position = position;
      p.font_size = font_size;
      p.text = text;
      p.theme = &theme;
      return p;
    }

    fan_masterpiece_make(
      (fan::vec3) position,
      (fan::vec2) size,
      (f32_t) font_size,
      (fan::string) text,
      (fan::string) id
    )
  };
  struct shape_sprite_t {

    loco_t::sprite_t::properties_t get_properties(
      loco_t::viewport_t& viewport,
      loco_t::camera_t& camera,
      loco_t::texturepack_t& texturepack
      ) {
      loco_t::sprite_t::properties_t p;
      p.viewport = &viewport;
      p.camera = &camera;
      p.position = position;
      p.size = size;

      loco_t::texturepack_t::ti_t ti;
      if (texturepack.qti(texturepack_name, &ti)) {
        p.image = &gloco->default_texture;
      }
      else {
        auto& pd = texturepack.get_pixel_data(ti.pack_id);
        p.image = &pd.image;
        p.tc_position = ti.position / pd.image.size;
        p.tc_size = ti.size / pd.image.size;
      }
      return p;
    }

    #if defined(fgm_build_model_maker)
    fan_masterpiece_make(
      (fan::vec3) position,
      (fan::vec2) size,
      (f32_t) parallax_factor,
      (fan::string) texturepack_name,
      (fan::string) id,
      (uint32_t) group_id
    )
    #else
    fan_masterpiece_make(
      (fan::vec3)position,
      (fan::vec2)size,
      (f32_t)parallax_factor,
      (fan::string)texturepack_name,
      (fan::string)id
    )
    #endif
  };
  struct shape_text_t {

    loco_t::text_t::properties_t get_properties(
      loco_t::viewport_t& viewport,
      loco_t::camera_t& camera
      ) {
      loco_t::text_t::properties_t p;
      p.viewport = &viewport;
      p.camera = &camera;
      p.position = position;
      p.font_size = size;
      p.text = text;
      return p;
    }

    fan_masterpiece_make(
      (fan::vec3) position,
      (f32_t) size,
      (fan::string) text,
      (fan::string) id
    );
  };
  struct shape_hitbox_t {

    loco_t::sprite_t::properties_t get_properties(
      loco_t::viewport_t& viewport,
      loco_t::camera_t& camera,
      loco_t::image_t* image
      ) {
      loco_t::sprite_t::properties_t p;
      p.viewport = &viewport;
      p.camera = &camera;
      p.position = position;
      p.size = size;
      p.image = image;
      return p;
    }

    fan_masterpiece_make(
      (fan::vec3) position,
      (fan::vec2) size,
      (loco_t::vfi_t::shape_type_t) vfi_type,
      (fan::string) id
    )
  };
  struct shape_mark_t {

    loco_t::sprite_t::properties_t get_properties(
      loco_t::viewport_t& viewport,
      loco_t::camera_t& camera,
      loco_t::image_t* image
      ) {
      loco_t::sprite_t::properties_t p;
      p.viewport = &viewport;
      p.camera = &camera;
      p.position = position;
      p.image = image;
      return p;
    }

    #if defined(fgm_build_model_maker)
    fan_masterpiece_make(
      (fan::vec3)position,
      (fan::string)id,
      (uint32_t) group_id
    )
    #else
    fan_masterpiece_make(
      (fan::vec3)position,
      (fan::string)id
    )
    #endif
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