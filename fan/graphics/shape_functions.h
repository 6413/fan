static constexpr int get_shape_category(std::uint16_t sti) {
  switch (sti) {
  case fan::graphics::shapes::shape_type_t::light:
    return fan::graphics::shapes::kp::light;

  case fan::graphics::shapes::shape_type_t::capsule:
  case fan::graphics::shapes::shape_type_t::gradient:
  case fan::graphics::shapes::shape_type_t::grid:
  case fan::graphics::shapes::shape_type_t::circle:
  case fan::graphics::shapes::shape_type_t::rectangle:
  #if defined(FAN_3D)
  case fan::graphics::shapes::shape_type_t::rectangle3d:
  case fan::graphics::shapes::shape_type_t::line3d:
  #endif
  case fan::graphics::shapes::shape_type_t::line:
    return fan::graphics::shapes::kp::common;

  case fan::graphics::shapes::shape_type_t::particles:
  case fan::graphics::shapes::shape_type_t::universal_image_renderer:
  case fan::graphics::shapes::shape_type_t::unlit_sprite:
  case fan::graphics::shapes::shape_type_t::sprite:
  case fan::graphics::shapes::shape_type_t::shader_shape:
    return fan::graphics::shapes::kp::texture;

  default:
    return -1;
  }
  return -1;
}

template <typename modifier_t>
static void update_shape(fan::graphics::shapes::shape_t* shape, modifier_t&& modifier_fn) {
  if (shape->get_visual_id().iic()) {
    return;
  }

  auto sti = shape->get_shape_type();

  auto key_pack_size = fan::graphics::g_shapes->shaper.GetKeysSize(shape->get_visual_id());
  std::unique_ptr<std::uint8_t[]> key_pack(new std::uint8_t[key_pack_size]);
  fan::graphics::g_shapes->shaper.WriteKeys(shape->get_visual_id(), key_pack.get());

  modifier_fn(sti, key_pack.get());

  auto _vi = shape->GetRenderData(fan::graphics::g_shapes->shaper);
  auto vlen_t = fan::graphics::g_shapes->shaper.GetRenderDataSize(sti);
  std::unique_ptr<std::uint8_t[]> vi(new std::uint8_t[vlen_t]);
  std::memcpy(vi.get(), _vi, vlen_t);

  auto _ri = shape->GetData(fan::graphics::g_shapes->shaper);
  auto rlen_t = fan::graphics::g_shapes->shaper.GetDataSize(sti);
  std::unique_ptr<std::uint8_t[]> ri(new std::uint8_t[rlen_t]);
  std::memcpy(ri.get(), _ri, rlen_t);

  shape->erase_shaper();
  shape->get_visual_id() = fan::graphics::g_shapes->shaper.add(sti, key_pack.get(), key_pack_size, vi.get(), ri.get());

#if defined(debug_shape_t)
  fan::print_impl("+", shape->NRI);
#endif
}

template<typename sti_t, typename key_pack_t>
static void set_position_impl(sti_t sti, key_pack_t key_pack, const fan::vec3& position) {
#if FAN_DEBUG >= 3
  static constexpr auto max_depth = std::numeric_limits<decltype(fan::graphics::kps_t::common_t::depth)>::max();
  if (position.z > max_depth) {
    fan::throw_error_impl(
      ("z depth value exceeded. dont give me bigger depth than " +
        std::to_string(max_depth)).c_str()
    );
  }
#endif
  switch (get_shape_category(sti)) {
  case fan::graphics::shapes::kp::common:
    shaper_get_key_safe(depth_t, common_t, depth) = position.z;
    break;
  case fan::graphics::shapes::kp::texture:
    shaper_get_key_safe(depth_t, texture_t, depth) = position.z;
    break;
  case fan::graphics::shapes::kp::light:
    break;
  default:
    fan::throw_error_impl("unimplemented");
  }
}

static fan::graphics::camera_t get_camera(const fan::graphics::shapes::shape_t* shape) {

  fan::graphics::camera_t cam{};
  fan::graphics::g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.camera; }) {
      cam = props.camera;
    }
  });
  return cam;

  auto sti = shape->get_shape_type();
  if (shape->get_visual_id().iic()) {
    return {};
  }
  std::uint8_t* key_pack = fan::graphics::g_shapes->shaper.GetKeys(shape->get_visual_id());

  switch (get_shape_category(sti)) {
  case fan::graphics::shapes::kp::light:
    return shaper_get_key_safe(camera_t, light_t, camera);
  case fan::graphics::shapes::kp::common:
    return shaper_get_key_safe(camera_t, common_t, camera);
  case fan::graphics::shapes::kp::texture:
    return shaper_get_key_safe(camera_t, texture_t, camera);
  default:
    fan::throw_error_impl("get_camera: unsupported shape");
  }
}

template<typename sti_t, typename key_pack_t>
static void set_camera_impl(
  sti_t sti,
  key_pack_t key_pack,
  fan::graphics::camera_t camera
) {
  switch (get_shape_category(sti)) {
  case fan::graphics::shapes::kp::light:
    shaper_get_key_safe(camera_t, light_t, camera) = camera;
    break;
  case fan::graphics::shapes::kp::common:
    shaper_get_key_safe(camera_t, common_t, camera) = camera;
    break;
  case fan::graphics::shapes::kp::texture:
    shaper_get_key_safe(camera_t, texture_t, camera) = camera;
    break;
  default:
    fan::throw_error_impl("set_camera: unsupported shape");
  }
}

static void set_camera(fan::graphics::shapes::shape_t* shape, fan::graphics::camera_t camera) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_camera_impl(sti, key_pack, camera);
  });
}

static fan::graphics::viewport_t get_viewport(const fan::graphics::shapes::shape_t* shape) {
  fan::graphics::viewport_t vp{};
  fan::graphics::g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.viewport; }) {
      vp = props.viewport;
    }
  });
  return vp;
}

template<typename sti_t, typename key_pack_t>
static void set_viewport_impl(
  sti_t sti,
  key_pack_t key_pack,
  fan::graphics::viewport_t viewport
) {
  switch (get_shape_category(sti)) {
  case fan::graphics::shapes::kp::light:
    shaper_get_key_safe(viewport_t, light_t, viewport) = viewport;
    break;
  case fan::graphics::shapes::kp::common:
    shaper_get_key_safe(viewport_t, common_t, viewport) = viewport;
    break;
  case fan::graphics::shapes::kp::texture:
    shaper_get_key_safe(viewport_t, texture_t, viewport) = viewport;
    break;
  default:
    fan::throw_error_impl("set_viewport: unsupported shape");
  }
}

static void set_viewport(fan::graphics::shapes::shape_t* shape, fan::graphics::viewport_t viewport) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_viewport_impl(sti, key_pack, viewport);
  });
}

static fan::graphics::image_t get_image(const fan::graphics::shapes::shape_t* shape) {
  fan::graphics::image_t img{};
  fan::graphics::g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.image; }) {
      img = props.image;
    }
    else if constexpr (requires { props.images; }) {
      img = props.images[0];
    }
  });
  return img;
}

template<typename sti_t, typename key_pack_t>
static void set_image_impl(
  sti_t sti,
  key_pack_t key_pack,
  fan::graphics::image_t image
) {
  if (get_shape_category(sti) == fan::graphics::shapes::kp::texture) {
    shaper_get_key_safe(image_t, texture_t, image) = image;
  }
  else {
    fan::throw_error_impl("set_image: unsupported shape");
  }
}

static void set_image(fan::graphics::shapes::shape_t* shape, fan::graphics::image_t image) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_image_impl(sti, key_pack, image);
  });
  fan::graphics::g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.images; } && !requires{ props.image; }) {
      props.images[0] = image;
    }
  });
}

static void set_position(fan::graphics::shapes::shape_t* shape, const fan::vec3& position) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_position_impl(sti, key_pack, position);
  });
}

static fan::graphics::shader_t get_shader(const fan::graphics::shapes::shape_t* shape) {
  fan::graphics::shader_t result{};
  fan::graphics::g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.shader; }) {
      result = props.shader;
    }
  });
  if (shape->get_visual_id().iic()) {
    return result;
  }
  auto sti = shape->get_shape_type();
  std::uint8_t* key_pack = fan::graphics::g_shapes->shaper.GetKeys(shape->get_visual_id());
  if (get_shape_category(sti) == fan::graphics::shapes::kp::texture) {
    result.gint() = shaper_get_key_safe(shader_raw_t, texture_t, shader_raw);
  }
  return result;
}
template<typename sti_t, typename key_pack_t>
static void set_shader_impl(sti_t sti, key_pack_t key_pack, fan::graphics::shader_t shader) {
  switch (get_shape_category(sti)) {
  case fan::graphics::shapes::kp::texture:
    shaper_get_key_safe(shader_raw_t, texture_t, shader_raw) = shader.gint();
    break;
  default:
    fan::throw_error_impl("set_shader: unsupported shape");
  }
}

static void set_shader(fan::graphics::shapes::shape_t* shape, fan::graphics::shader_t shader) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_shader_impl(sti, key_pack, shader);
  });
}

/*
TODO REMOVE

*/

template<typename shape_type>
static fan::vec2 generic_get_grid_size(const fan::graphics::shapes::shape_t* s) {
  fan::vec2 result{};
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.grid_size; }) {
      result = props.grid_size;
    }
  });
  return result;
}

template<typename shape_type>
static void generic_set_grid_size(fan::graphics::shapes::shape_t* s, const fan::vec2& v) {
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.grid_size; }) {
      props.grid_size = v;
    }
  });
}

template<typename shape_type>
static bool generic_get_visible(const fan::graphics::shapes::shape_t* s) {
  bool result = true;
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.visible; }) {
      result = props.visible;
    }
  });
  return result;
}

template<typename shape_type>
static void generic_set_visible(fan::graphics::shapes::shape_t* s, bool v) {
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.visible; }) {
      props.visible = v;
    }
  });
}

static f32_t get_radius(const fan::graphics::shapes::shape_t* s) {
  f32_t r = 0;
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.radius; }) {
      r = props.radius;
    }
    else if constexpr (requires { props.size; }) {
      r = props.size.x;
    }
  });
  return r;
}

template<typename shape_type>
static f32_t generic_get_outline_size(const fan::graphics::shapes::shape_t* s) {
  f32_t r = 0;
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.outline_size; }) {
      r = props.outline_size;
    }
  });
  return r;
}

static std::array<fan::color, 4> get_colors(const fan::graphics::shapes::shape_t* s) {
  std::array<fan::color, 4> result {};
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.color; }) {
      using color_t = std::remove_cvref_t<decltype(props.color)>; 
      if constexpr (std::is_same_v<color_t, std::array<fan::color, 4>>) {
        result = props.color;
      }
    }
  });
  return result;
}
static void set_colors(fan::graphics::shapes::shape_t* s, const std::array<fan::color, 4>& v) {
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.color; }) {
      using color_t = std::remove_cvref_t<decltype(props.color)>; 
      if constexpr (std::is_same_v<color_t, std::array<fan::color, 4>>) {
        props.color = v;
      }
    }
  });

  if (s->get_visual_id().iic()) {
    return;
  }

  if (auto* vdata = s->get_vdata<fan::graphics::shapes::gradient_t::vi_t>()) {
    vdata->color = v;

    auto& vid = s->get_visual_id();
    auto& sldata = fan::graphics::g_shapes->shaper.ShapeList[vid];

    fan::graphics::g_shapes->shaper.ElementIsPartiallyEdited(
      sldata.sti,
      sldata.blid,
      sldata.ElementIndex,
      offsetof(fan::graphics::shapes::gradient_t::vi_t, color),
      sizeof(vdata->color)
    );
  }
}

template<typename shape_type, typename field_t, typename props_field_ptr_t, typename vi_field_ptr_t>
static void set_with_sync(fan::graphics::shapes::shape_t* s, const field_t& v, props_field_ptr_t props_field_ptr, vi_field_ptr_t vi_field_ptr, bool updates_keypack = false){
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.*props_field_ptr; }) {
      using actual_field_t = std::remove_reference_t<decltype(props.*props_field_ptr)>;

      if constexpr (std::is_assignable_v<actual_field_t&, const field_t&>) {
        props.*props_field_ptr = v;
      }
    }
  });

  auto& vid = s->get_visual_id();
  if (vid.iic()) {
    return;
  }

  if constexpr (requires { typename shape_type::vi_t; }) {
    if constexpr (!std::is_same_v<vi_field_ptr_t, std::nullptr_t>) {
      auto* vdata = s->get_vdata<typename shape_type::vi_t>();
      if (vdata) {
        if constexpr (std::is_member_object_pointer_v<vi_field_ptr_t>) {
          if constexpr (requires { vdata->*vi_field_ptr; }) {
            using actual_vi_field_t = std::remove_reference_t<decltype(vdata->*vi_field_ptr)>;

            if constexpr (std::is_assignable_v<actual_vi_field_t&, const field_t&>) {
              vdata->*vi_field_ptr = v;
              auto& sldata = fan::graphics::g_shapes->shaper.ShapeList[vid];
              using member_t = std::remove_reference_t<decltype(vdata->*vi_field_ptr)>;
              fan::graphics::g_shapes->shaper.ElementIsPartiallyEdited(
                sldata.sti,
                sldata.blid,
                sldata.ElementIndex,
                fan::member_offset(vi_field_ptr),
                sizeof(member_t)
              );
            }
          }
        }
      }
    }
  }

  if (updates_keypack) {
    auto sti = s->get_shape_type();
    std::uint8_t* key_pack = fan::graphics::g_shapes->shaper.GetKeys(vid);

    switch (get_shape_category(sti)) {
    case fan::graphics::shapes::kp::common:
      if constexpr (std::is_same_v<field_t, fan::vec3>) {
        shaper_get_key_safe(depth_t, common_t, depth) = v.z;
      }
      break;
    case fan::graphics::shapes::kp::texture:
      if constexpr (std::is_same_v<field_t, fan::vec3>) {
        shaper_get_key_safe(depth_t, texture_t, depth) = v.z;
      }
      break;
    }
  }
}

template<typename shape_type>
static void generic_set_radius(fan::graphics::shapes::shape_t* s, f32_t v) {
  using props_t = typename shape_type::properties_t;
  if constexpr (requires { std::declval<props_t>().radius; }) {
    if constexpr (requires { typename shape_type::vi_t; }) {
      using vi_t = typename shape_type::vi_t;
      if constexpr (requires { std::declval<vi_t>().radius; }) {
        set_with_sync<shape_type>(s, v, &props_t::radius, &vi_t::radius, false);
      } else {
        set_with_sync<shape_type>(s, v, &props_t::radius, nullptr, false);
      }
    }
  }
}

#define MAKE_ACCESSORS(name, value_type, updates_keys) \
  template<typename shape_type> \
  static value_type generic_get_##name(const fan::graphics::shapes::shape_t* s){ \
    value_type result{}; \
    fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props){ \
      if constexpr (requires { props.name; }) { \
        using actual_t = std::remove_cvref_t<decltype(props.name)>; \
        if constexpr (std::is_same_v<actual_t, value_type>) { \
          result = props.name; \
        } \
      } \
    }); \
    return result; \
  } \
  template<typename shape_type> \
  static void generic_set_##name( \
    fan::graphics::shapes::shape_t* s, \
    std::conditional_t<std::is_scalar_v<value_type>, value_type, const value_type&> v){ \
    using props_t = typename shape_type::properties_t; \
    if constexpr (requires { std::declval<props_t>().name; }) { \
      using actual_props_t = std::remove_cvref_t<decltype(std::declval<props_t>().name)>; \
      if constexpr (std::is_assignable_v<actual_props_t&, const value_type&>) { \
        if constexpr (requires { typename shape_type::vi_t; }) { \
          using vi_t = typename shape_type::vi_t; \
          if constexpr (requires { std::declval<vi_t>().name; }) { \
            using actual_vi_t = std::remove_cvref_t<decltype(std::declval<vi_t>().name)>; \
            if constexpr (std::is_assignable_v<actual_vi_t&, const value_type&>) { \
              set_with_sync<shape_type>(s, v, &props_t::name, &vi_t::name, updates_keys); \
            } \
          } else { \
            set_with_sync<shape_type>(s, v, &props_t::name, nullptr, updates_keys); \
          } \
        } else { \
          set_with_sync<shape_type>(s, v, &props_t::name, nullptr, updates_keys); \
        } \
      } \
    } \
  }


MAKE_ACCESSORS(camera, fan::graphics::camera_t, true)
MAKE_ACCESSORS(viewport, fan::graphics::viewport_t, true)
MAKE_ACCESSORS(image, fan::graphics::image_t, true)
MAKE_ACCESSORS(position, fan::vec3, true)
MAKE_ACCESSORS(color, fan::color, false)
MAKE_ACCESSORS(angle, fan::vec3, false)
MAKE_ACCESSORS(rotation_point, fan::vec2, false)
MAKE_ACCESSORS(tc_position, fan::vec2, false)
MAKE_ACCESSORS(tc_size, fan::vec2, false)
MAKE_ACCESSORS(parallax_factor, fan::vec2, false)
MAKE_ACCESSORS(flags, std::uint32_t, false)
MAKE_ACCESSORS(outline_color, fan::color, false)
MAKE_ACCESSORS(src, fan::vec3, true)
MAKE_ACCESSORS(dst, fan::vec2, false)
MAKE_ACCESSORS(shader, fan::graphics::shader_t, true)

template<typename shape_type>
static void generic_set_shader_kp(fan::graphics::shapes::shape_t* s, fan::graphics::shader_t sh) {
  generic_set_shader<shape_type>(s, sh);
  set_shader(s, sh);
}

template<typename shape_type>
static void generic_set_camera_kp(fan::graphics::shapes::shape_t* s, fan::graphics::camera_t cam){
  generic_set_camera<shape_type>(s, cam);
  set_camera(s, cam);
}

template<typename shape_type>
static void generic_set_viewport_kp(fan::graphics::shapes::shape_t* s, fan::graphics::viewport_t vp){
  generic_set_viewport<shape_type>(s, vp);
  set_viewport(s, vp);
}

template<typename shape_type>
static void generic_set_image_kp(fan::graphics::shapes::shape_t* s, fan::graphics::image_t img){
  generic_set_image<shape_type>(s, img);
  set_image(s, img);
}

template<typename shape_type>
static void generic_set_position2(fan::graphics::shapes::shape_t* s, const fan::vec2& v){
  fan::vec3 pos = generic_get_position<shape_type>(s);
  pos.x = v.x;
  pos.y = v.y;
  generic_set_position<shape_type>(s, pos);
}

template<typename shape_type>
static void generic_set_position3(fan::graphics::shapes::shape_t* s, const fan::vec3& v){
  set_position(s, v);
  generic_set_position<shape_type>(s, v);
}

template<typename shape_type>
static fan::vec2 generic_get_size(const fan::graphics::shapes::shape_t* s) {
  fan::vec2 r{};
  fan::graphics::g_shapes->visit_shape_draw_data(s->NRI, [&](auto& p) {
    if constexpr (requires { p.size; }) r = p.size;
    else if constexpr (requires { p.radius; }) r = fan::vec2{p.radius, p.radius};
  });
  return r;
}

template<typename shape_type>
static void generic_set_size(fan::graphics::shapes::shape_t* s, const fan::vec2& v) {
  if constexpr (requires { &shape_type::properties_t::size; }) {
    if constexpr (requires { &shape_type::vi_t::size; }) {
      set_with_sync<shape_type>(s, v, &shape_type::properties_t::size, &shape_type::vi_t::size);
    } else {
      set_with_sync<shape_type>(s, v, &shape_type::properties_t::size, nullptr);
    }
  } else {
    generic_set_radius<shape_type>(s, v.x);
  }
}

static void generic_set_line(fan::graphics::shapes::shape_t* shape, const fan::vec2& src, const fan::vec2& dst) {
  fan::vec3 current_src = generic_get_src<fan::graphics::shapes::line_t>(shape);
  generic_set_src<fan::graphics::shapes::line_t>(shape, { src.x, src.y, current_src.z });
  generic_set_dst<fan::graphics::shapes::line_t>(shape, dst);

  shape->update_dynamic();
}
#if defined(FAN_3D)
static void generic_set_line(fan::graphics::shapes::shape_t* shape, const fan::vec3& src, const fan::vec2& dst) {
  generic_set_src<fan::graphics::shapes::line_t>(shape, src);
  generic_set_dst<fan::graphics::shapes::line_t>(shape, dst);

  shape->update_dynamic();
}
#endif

using get_colors_fn_t = std::array<fan::color,4> (*)(const fan::graphics::shapes::shape_t*);
using set_colors_fn_t = void (*)(fan::graphics::shapes::shape_t*, const std::array<fan::color,4>&);

#define SHAPE_FUNCTION_LIST(X) \
X(push_back, fan::graphics::shapes::shape_t(*)(void*)) \
X(get_position, fan::vec3(*)(const fan::graphics::shapes::shape_t*)) \
X(set_position2, void(*)(fan::graphics::shapes::shape_t*, const fan::vec2&)) \
X(set_position3, void(*)(fan::graphics::shapes::shape_t*, const fan::vec3&)) \
X(set_position3_impl, void(*)(fan::graphics::shapes::shape_t*, const fan::vec3&)) \
X(get_size, fan::vec2(*)(const fan::graphics::shapes::shape_t*)) \
X(get_size3, fan::vec3(*)(const fan::graphics::shapes::shape_t*)) \
X(set_size, void(*)(fan::graphics::shapes::shape_t*, const fan::vec2&)) \
X(set_size3, void(*)(fan::graphics::shapes::shape_t*, const fan::vec3&)) \
X(get_rotation_point, fan::vec2(*)(const fan::graphics::shapes::shape_t*)) \
X(set_rotation_point, void(*)(fan::graphics::shapes::shape_t*, const fan::vec2&)) \
X(get_color, fan::color(*)(const fan::graphics::shapes::shape_t*)) \
X(set_color, void(*)(fan::graphics::shapes::shape_t*, const fan::color&)) \
X(get_angle, fan::vec3(*)(const fan::graphics::shapes::shape_t*)) \
X(set_angle, void(*)(fan::graphics::shapes::shape_t*, const fan::vec3&)) \
X(get_tc_position, fan::vec2(*)(const fan::graphics::shapes::shape_t*)) \
X(set_tc_position, void(*)(fan::graphics::shapes::shape_t*, const fan::vec2&)) \
X(get_tc_size, fan::vec2(*)(const fan::graphics::shapes::shape_t*)) \
X(set_tc_size, void(*)(fan::graphics::shapes::shape_t*, const fan::vec2&)) \
X(get_grid_size, fan::vec2(*)(const fan::graphics::shapes::shape_t*)) \
X(set_grid_size, void(*)(fan::graphics::shapes::shape_t*, const fan::vec2&)) \
X(get_camera, fan::graphics::camera_t(*)(const fan::graphics::shapes::shape_t*)) \
X(set_camera, void(*)(fan::graphics::shapes::shape_t*, fan::graphics::camera_t)) \
X(get_viewport, fan::graphics::viewport_t(*)(const fan::graphics::shapes::shape_t*)) \
X(set_viewport, void(*)(fan::graphics::shapes::shape_t*, fan::graphics::viewport_t)) \
X(get_image, fan::graphics::image_t(*)(const fan::graphics::shapes::shape_t*)) \
X(set_image, void(*)(fan::graphics::shapes::shape_t*, fan::graphics::image_t)) \
X(get_visible, bool(*)(const fan::graphics::shapes::shape_t*)) \
X(set_visible, void(*)(fan::graphics::shapes::shape_t*, bool)) \
X(get_parallax_factor, fan::vec2(*)(const fan::graphics::shapes::shape_t*)) \
X(set_parallax_factor, void(*)(fan::graphics::shapes::shape_t*, const fan::vec2&)) \
X(get_flags, std::uint32_t(*)(const fan::graphics::shapes::shape_t*)) \
X(set_flags, void(*)(fan::graphics::shapes::shape_t*, std::uint32_t)) \
X(get_radius, f32_t(*)(const fan::graphics::shapes::shape_t*)) \
X(set_radius, void(*)(fan::graphics::shapes::shape_t*, f32_t)) \
X(get_src, fan::vec3(*)(const fan::graphics::shapes::shape_t*)) \
X(get_dst, fan::vec2(*)(const fan::graphics::shapes::shape_t*)) \
X(get_outline_size, f32_t(*)(const fan::graphics::shapes::shape_t*)) \
X(get_outline_color, fan::color(*)(const fan::graphics::shapes::shape_t*)) \
X(set_outline_color, void(*)(fan::graphics::shapes::shape_t*, const fan::color&)) \
X(set_line, void(*)(fan::graphics::shapes::shape_t*, const fan::vec2&, const fan::vec2&)) \
X(set_line3, void(*)(fan::graphics::shapes::shape_t*, const fan::vec3&, const fan::vec2&)) \
X(get_colors, get_colors_fn_t) \
X(set_colors, set_colors_fn_t) \
X(get_shader, fan::graphics::shader_t(*)(const fan::graphics::shapes::shape_t*)) \
X(set_shader, void(*)(fan::graphics::shapes::shape_t*, fan::graphics::shader_t))


struct shape_functions_t {
#define MAKE_TYPEDEF(name, type) using name##_fn = type;
  SHAPE_FUNCTION_LIST(MAKE_TYPEDEF)
  #undef MAKE_TYPEDEF

  struct vtable_t {
    #define MAKE_MEMBER(name, type) name##_fn name;
    SHAPE_FUNCTION_LIST(MAKE_MEMBER)
    #undef MAKE_MEMBER
  };

#define SKIP(x) 

#define REGISTER_SHAPE_FUNCS(shape_name) \
vtables_storage[fan::graphics::shape_type_t::shape_name].push_back = [](void* p) -> fan::graphics::shapes::shape_t { \
    return fan::graphics::g_shapes->shape_name.push_back(*static_cast<fan::graphics::shapes::shape_name##_t::properties_t*>(p)); \
}; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_position = generic_get_position<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_position2 = generic_set_position2<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_position3 = generic_set_position3<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_position3_impl = generic_set_position<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_line = generic_set_line; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_size = generic_get_size<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_size = generic_set_size<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_color = generic_get_color<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_color = generic_set_color<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_angle = generic_get_angle<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_angle = generic_set_angle<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_rotation_point = generic_get_rotation_point<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_rotation_point = generic_set_rotation_point<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_tc_position = generic_get_tc_position<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_tc_position = generic_set_tc_position<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_tc_size = generic_get_tc_size<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_tc_size = generic_set_tc_size<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_parallax_factor = generic_get_parallax_factor<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_parallax_factor = generic_set_parallax_factor<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_flags = generic_get_flags<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_flags = generic_set_flags<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_outline_color = generic_get_outline_color<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_outline_color = generic_set_outline_color<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_camera = get_camera; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_camera = generic_set_camera_kp<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_viewport = get_viewport; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_viewport = generic_set_viewport_kp<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_image = get_image; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_image = generic_set_image_kp<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_grid_size = generic_get_grid_size<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_grid_size = generic_set_grid_size<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_visible = generic_get_visible<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_visible = generic_set_visible<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_radius = get_radius; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_radius = generic_set_radius<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_src = generic_get_src<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_dst = generic_get_dst<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_outline_size = generic_get_outline_size<fan::graphics::shapes::shape_name##_t>; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_colors = get_colors; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_colors = set_colors; \
vtables_storage[fan::graphics::shape_type_t::shape_name].get_shader = get_shader; \
vtables_storage[fan::graphics::shape_type_t::shape_name].set_shader = generic_set_shader_kp<fan::graphics::shapes::shape_name##_t>;

  /*vtables_storage[shape_type_t::shape_name].get_size3 = generic_get_size3<shape_name##_t>; \
  vtables_storage[shape_type_t::shape_name].set_size3 = generic_set_size3<shape_name##_t>; \*/

  shape_functions_t() {
    GEN_SHAPES(REGISTER_SHAPE_FUNCS, SKIP);
  }

  vtable_t& operator[](std::uint16_t shape){
    return vtables_storage[shape];
  }

  vtable_t vtables_storage[fan::graphics::shape_type_t::last];
};

template<typename T>
auto get_field(const T* props, auto field_ptr){
  if constexpr (requires { props->*field_ptr; }) {
    return props->*field_ptr;
  }
  else {
    using ret_t = std::remove_cvref_t<decltype(props->*field_ptr)>;
    return ret_t{};
  }
}