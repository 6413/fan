template <typename modifier_t>
static void update_shape(shape_t* shape, modifier_t&& modifier_fn) {

  if (shape->get_visual_id().iic()) {
    return;
  }

  auto sti = shape->get_shape_type();

  auto key_pack_size = g_shapes->shaper.GetKeysSize(shape->get_visual_id());
  std::unique_ptr<uint8_t[]> key_pack(new uint8_t[key_pack_size]);
  g_shapes->shaper.WriteKeys(shape->get_visual_id(), key_pack.get());

  modifier_fn(sti, key_pack.get());

  auto _vi = shape->GetRenderData(g_shapes->shaper);
  auto vlen_t = g_shapes->shaper.GetRenderDataSize(sti);
  std::unique_ptr<uint8_t[]> vi(new uint8_t[vlen_t]);
  std::memcpy(vi.get(), _vi, vlen_t);

  auto _ri = shape->GetData(g_shapes->shaper);
  auto rlen_t = g_shapes->shaper.GetDataSize(sti);
  std::unique_ptr<uint8_t[]> ri(new uint8_t[rlen_t]);
  std::memcpy(ri.get(), _ri, rlen_t);

  shape->erase_shaper();
  shape->get_visual_id() = g_shapes->shaper.add(sti, key_pack.get(), key_pack_size, vi.get(), ri.get());

#if defined(debug_shape_t)
  fan::print("+", shape->NRI);
#endif
}

template<typename sti_t, typename key_pack_t>
static void set_position_impl(sti_t sti, key_pack_t key_pack, const fan::vec3& position) {
#if FAN_DEBUG >= 3
  if (position.z > std::numeric_limits<decltype(kps_t::common_t::depth)>::max()) {
    fan::throw_error("z depth value exceeded. dont give me bigger depth than", std::numeric_limits<decltype(kps_t::common_t::depth)>::max());
  }
#endif
  switch (get_shape_category(sti)) {
  case shapes::kp::common:
    shaper_get_key_safe(depth_t, common_t, depth) = position.z;
    break;
  case shapes::kp::texture:
    shaper_get_key_safe(depth_t, texture_t, depth) = position.z;
    break;
  case shapes::kp::light:
    break;
  default:
    fan::print("unimplemented");
  }
}

static fan::graphics::camera_t get_camera(const shape_t* shape) {

  fan::graphics::camera_t cam{};
  g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.camera; }) {
      cam = props.camera;
    }
  });
  return cam;

  auto sti = shape->get_shape_type();
  if (!shape->get_visual_id()) {
    return {};
  }
  uint8_t* key_pack = g_shapes->shaper.GetKeys(shape->get_visual_id());

  switch (get_shape_category(sti)) {
  case shapes::kp::light:
    return shaper_get_key_safe(camera_t, light_t, camera);
  case shapes::kp::common:
    return shaper_get_key_safe(camera_t, common_t, camera);
  case shapes::kp::texture:
    return shaper_get_key_safe(camera_t, texture_t, camera);
  default:
    fan::throw_error("get_camera: unsupported shape");
  }
}

template<typename sti_t, typename key_pack_t>
static void set_camera_impl(
  sti_t sti,
  key_pack_t key_pack,
  fan::graphics::camera_t camera
) {
  switch (get_shape_category(sti)) {
  case shapes::kp::light:
    shaper_get_key_safe(camera_t, light_t, camera) = camera;
    break;
  case shapes::kp::common:
    shaper_get_key_safe(camera_t, common_t, camera) = camera;
    break;
  case shapes::kp::texture:
    shaper_get_key_safe(camera_t, texture_t, camera) = camera;
    break;
  default:
    fan::throw_error("set_camera: unsupported shape");
  }
}

static void set_camera(shape_t* shape, fan::graphics::camera_t camera) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_camera_impl(sti, key_pack, camera);
  });
}

static fan::graphics::viewport_t get_viewport(const shape_t* shape) {
  fan::graphics::viewport_t vp{};
  g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
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
  case shapes::kp::light:
    shaper_get_key_safe(viewport_t, light_t, viewport) = viewport;
    break;
  case shapes::kp::common:
    shaper_get_key_safe(viewport_t, common_t, viewport) = viewport;
    break;
  case shapes::kp::texture:
    shaper_get_key_safe(viewport_t, texture_t, viewport) = viewport;
    break;
  default:
    fan::throw_error("set_viewport: unsupported shape");
  }
}

static void set_viewport(shape_t* shape, fan::graphics::viewport_t viewport) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_viewport_impl(sti, key_pack, viewport);
  });
}

static fan::graphics::image_t get_image(const shape_t* shape) {
  fan::graphics::image_t img{};
  g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.image; }) {
      img = props.image;
    }
  });
  g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.images; }) {
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
  if (get_shape_category(sti) == shapes::kp::texture) {
    shaper_get_key_safe(image_t, texture_t, image) = image;
  }
  else {
    fan::throw_error("set_image: unsupported shape");
  }
}

static void set_image(shape_t* shape, fan::graphics::image_t image) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_image_impl(sti, key_pack, image);
  });
  g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.images; }) {
      props.images[0] = image;
    }
  });
}

static void set_position(shape_t* shape, const fan::vec3& position) {
  update_shape(shape, [&](auto sti, auto key_pack) {
    set_position_impl(sti, key_pack, position);
  });
}


/*
TODO REMOVE

*/

template<typename shape_type>
static fan::vec2 generic_get_grid_size(const shape_t* s) {
  fan::vec2 result{};
  g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.grid_size; }) {
      result = props.grid_size;
    }
  });
  return result;
}

template<typename shape_type>
static void generic_set_grid_size(shape_t* s, const fan::vec2& v) {
  g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.grid_size; }) {
      props.grid_size = v;
    }
  });
}

template<typename shape_type>
static bool generic_get_visible(const shape_t* s) {
  bool result = true;
  g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.visible; }) {
      result = props.visible;
    }
  });
  return result;
}

template<typename shape_type>
static void generic_set_visible(shape_t* s, bool v) {
  g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.visible; }) {
      props.visible = v;
    }
  });
}

static f32_t get_radius(const shape_t* s) {
  f32_t r = 0;
  g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.radius; }) {
      r = props.radius;
    }
  });
  return r;
}

template<typename shape_type>
static f32_t generic_get_outline_size(const shape_t* s) {
  f32_t r = 0;
  g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    if constexpr (requires { props.outline_size; }) {
      r = props.outline_size;
    }
  });
  return r;
}

static constexpr int get_shape_category(uint16_t sti) {
  switch (sti) {
  case fan::graphics::shapes::shape_type_t::light:
    return shapes::kp::light;

  case fan::graphics::shapes::shape_type_t::capsule:
  case fan::graphics::shapes::shape_type_t::gradient:
  case fan::graphics::shapes::shape_type_t::grid:
  case fan::graphics::shapes::shape_type_t::circle:
  case fan::graphics::shapes::shape_type_t::rectangle:
  #if defined(FAN_3D)
  case fan::graphics::shapes::shape_type_t::rectangle3d:
  #endif
  case fan::graphics::shapes::shape_type_t::line:
    return shapes::kp::common;

  case fan::graphics::shapes::shape_type_t::particles:
  case fan::graphics::shapes::shape_type_t::universal_image_renderer:
  case fan::graphics::shapes::shape_type_t::unlit_sprite:
  case fan::graphics::shapes::shape_type_t::sprite:
  case fan::graphics::shapes::shape_type_t::shader_shape:
    return shapes::kp::texture;

  default:
    return -1;
  }
  return -1;
}

#define SHAPE_FUNCTION_LIST(X) \
X(push_back, shape_t(*)(void*)) \
X(get_position, fan::vec3(*)(const shape_t*)) \
X(set_position2, void(*)(shape_t*, const fan::vec2&)) \
X(set_position3, void(*)(shape_t*, const fan::vec3&)) \
X(set_position3_impl, void(*)(shape_t*, const fan::vec3&)) \
X(get_size, fan::vec2(*)(const shape_t*)) \
X(get_size3, fan::vec3(*)(const shape_t*)) \
X(set_size, void(*)(shape_t*, const fan::vec2&)) \
X(set_size3, void(*)(shape_t*, const fan::vec3&)) \
X(get_rotation_point, fan::vec2(*)(const shape_t*)) \
X(set_rotation_point, void(*)(shape_t*, const fan::vec2&)) \
X(get_color, fan::color(*)(const shape_t*)) \
X(set_color, void(*)(shape_t*, const fan::color&)) \
X(get_angle, fan::vec3(*)(const shape_t*)) \
X(set_angle, void(*)(shape_t*, const fan::vec3&)) \
X(get_tc_position, fan::vec2(*)(const shape_t*)) \
X(set_tc_position, void(*)(shape_t*, const fan::vec2&)) \
X(get_tc_size, fan::vec2(*)(const shape_t*)) \
X(set_tc_size, void(*)(shape_t*, const fan::vec2&)) \
X(get_grid_size, fan::vec2(*)(const shape_t*)) \
X(set_grid_size, void(*)(shape_t*, const fan::vec2&)) \
X(get_camera, fan::graphics::camera_t(*)(const shape_t*)) \
X(set_camera, void(*)(shape_t*, fan::graphics::camera_t)) \
X(get_viewport, fan::graphics::viewport_t(*)(const shape_t*)) \
X(set_viewport, void(*)(shape_t*, fan::graphics::viewport_t)) \
X(get_image, fan::graphics::image_t(*)(const shape_t*)) \
X(set_image, void(*)(shape_t*, fan::graphics::image_t)) \
X(get_visible, bool(*)(const shape_t*)) \
X(set_visible, void(*)(shape_t*, bool)) \
X(get_parallax_factor, f32_t(*)(const shape_t*)) \
X(set_parallax_factor, void(*)(shape_t*, f32_t)) \
X(get_flags, uint32_t(*)(const shape_t*)) \
X(set_flags, void(*)(shape_t*, uint32_t)) \
X(get_radius, f32_t(*)(const shape_t*)) \
X(get_src, fan::vec3(*)(const shape_t*)) \
X(get_dst, fan::vec2(*)(const shape_t*)) \
X(get_outline_size, f32_t(*)(const shape_t*)) \
X(get_outline_color, fan::color(*)(const shape_t*)) \
X(set_outline_color, void(*)(shape_t*, const fan::color&)) \
X(set_line, void(*)(shape_t*, const fan::vec2&, const fan::vec2&)) \
X(set_line3, void(*)(shape_t*, const fan::vec3&, const fan::vec2&))

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
vtables_storage[shape_type_t::shape_name].push_back = [](void* p) -> shape_t { \
    return g_shapes->shape_name.push_back(*static_cast<shape_name##_t::properties_t*>(p)); \
}; \
vtables_storage[shape_type_t::shape_name].get_position = generic_get_position<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_position2 = generic_set_position2<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_position3 = generic_set_position3<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_position3_impl = generic_set_position<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_line = set_line; \
vtables_storage[shape_type_t::shape_name].get_size = generic_get_size<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_size = generic_set_size<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_color = generic_get_color<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_color = generic_set_color<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_angle = generic_get_angle<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_angle = generic_set_angle<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_rotation_point = generic_get_rotation_point<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_rotation_point = generic_set_rotation_point<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_tc_position = generic_get_tc_position<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_tc_position = generic_set_tc_position<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_tc_size = generic_get_tc_size<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_tc_size = generic_set_tc_size<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_parallax_factor = generic_get_parallax_factor<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_parallax_factor = generic_set_parallax_factor<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_flags = generic_get_flags<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_flags = generic_set_flags<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_outline_color = generic_get_outline_color<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_outline_color = generic_set_outline_color<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_camera = get_camera; \
vtables_storage[shape_type_t::shape_name].set_camera = generic_set_camera_kp<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_viewport = get_viewport; \
vtables_storage[shape_type_t::shape_name].set_viewport = generic_set_viewport_kp<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_image = get_image; \
vtables_storage[shape_type_t::shape_name].set_image = generic_set_image_kp<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_grid_size = generic_get_grid_size<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_grid_size = generic_set_grid_size<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_visible = generic_get_visible<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].set_visible = generic_set_visible<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_radius = get_radius; \
vtables_storage[shape_type_t::shape_name].get_src = generic_get_src<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_dst = generic_get_dst<shape_name##_t>; \
vtables_storage[shape_type_t::shape_name].get_outline_size = generic_get_outline_size<shape_name##_t>;

  /*vtables_storage[shape_type_t::shape_name].get_size3 = generic_get_size3<shape_name##_t>; \
  vtables_storage[shape_type_t::shape_name].set_size3 = generic_set_size3<shape_name##_t>; \*/

  shape_functions_t() {
    GEN_SHAPES(REGISTER_SHAPE_FUNCS, SKIP)
  }

  vtable_t& operator[](uint16_t shape){
    return vtables_storage[shape];
  }

  vtable_t vtables_storage[shape_type_t::last];
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

template<typename T>
fan::vec2 get_size_from_radius(const T* props){
  if constexpr (requires { props->radius; }) {
    return fan::vec2{props->radius, props->radius};
  }
  else if constexpr (requires { props->size; }) {
    return props->size;
  }
  return fan::vec2{};
}

template<typename shape_type, typename field_t, typename props_field_ptr_t, typename vi_field_ptr_t>
static void set_with_sync(shape_t* s, const field_t& v, props_field_ptr_t props_field_ptr, vi_field_ptr_t vi_field_ptr, bool updates_keypack = false){
  g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props) {
    using props_decayed_t = std::decay_t<decltype(props)>;
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
              auto& sldata = g_shapes->shaper.ShapeList[vid];
              using member_t = std::remove_reference_t<decltype(vdata->*vi_field_ptr)>;
              g_shapes->shaper.ElementIsPartiallyEdited(
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
    uint8_t* key_pack = g_shapes->shaper.GetKeys(vid);

    switch (get_shape_category(sti)) {
    case shapes::kp::common:
      if constexpr (std::is_same_v<field_t, fan::vec3>) {
        shaper_get_key_safe(depth_t, common_t, depth) = v.z;
      }
      break;
    case shapes::kp::texture:
      if constexpr (std::is_same_v<field_t, fan::vec3>) {
        shaper_get_key_safe(depth_t, texture_t, depth) = v.z;
      }
      break;
    }
  }
}

#define MAKE_ACCESSORS(name, value_type, updates_keys) \
  template<typename shape_type> \
  static value_type generic_get_##name(const shape_t* s){ \
    value_type result{}; \
    g_shapes->visit_shape_draw_data(s->NRI, [&](auto& props){ \
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
    shape_t* s, \
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
MAKE_ACCESSORS(size, fan::vec2, false)
MAKE_ACCESSORS(color, fan::color, false)
MAKE_ACCESSORS(angle, fan::vec3, false)
MAKE_ACCESSORS(rotation_point, fan::vec2, false)
MAKE_ACCESSORS(tc_position, fan::vec2, false)
MAKE_ACCESSORS(tc_size, fan::vec2, false)
MAKE_ACCESSORS(parallax_factor, f32_t, false)
MAKE_ACCESSORS(flags, uint32_t, false)
MAKE_ACCESSORS(outline_color, fan::color, false)
MAKE_ACCESSORS(src, fan::vec3, true)
MAKE_ACCESSORS(dst, fan::vec2, false)

template<typename shape_type>
static void generic_set_camera_kp(shape_t* s, fan::graphics::camera_t cam){
  generic_set_camera<shape_type>(s, cam);
  set_camera(s, cam);
}

template<typename shape_type>
static void generic_set_viewport_kp(shape_t* s, fan::graphics::viewport_t vp){
  generic_set_viewport<shape_type>(s, vp);
  set_viewport(s, vp);
}

template<typename shape_type>
static void generic_set_image_kp(shape_t* s, fan::graphics::image_t img){
  generic_set_image<shape_type>(s, img);
  set_image(s, img);
}

template<typename shape_type>
static void generic_set_position2(shape_t* s, const fan::vec2& v){
  fan::vec3 pos = generic_get_position<shape_type>(s);
  pos.x = v.x;
  pos.y = v.y;
  generic_set_position<shape_type>(s, pos);
}

template<typename shape_type>
static void generic_set_position3(shape_t* s, const fan::vec3& v){
  set_position(s, v);
  generic_set_position<shape_type>(s, v);
}

static void set_line(shape_t* shape, const fan::vec2& src, const fan::vec2& dst) {
  fan::vec3 current_src = generic_get_src<fan::graphics::shapes::line_t>(shape);
  generic_set_src<fan::graphics::shapes::line_t>(shape, { src.x, src.y, current_src.z });
  generic_set_dst<fan::graphics::shapes::line_t>(shape, dst);

  shape->update_dynamic();
}
static void set_line(shape_t* shape, const fan::vec3& src, const fan::vec2& dst) {
  generic_set_src<fan::graphics::shapes::line_t>(shape, src);
  generic_set_dst<fan::graphics::shapes::line_t>(shape, dst);

  shape->update_dynamic();
}