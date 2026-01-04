module;

#if defined(FAN_2D)

  #if defined(FAN_PHYSICS_2D)
  #include <fan/utility.h>
  #include <box2d/box2d.h>
  #endif

  #include <vector>
  #include <utility>
  #include <functional>
  #include <string>
  #include <cstring>
  #include <source_location>
  #include <coroutine>

#endif

module fan.graphics.physics_shapes;

#if defined(FAN_2D)

#if defined(FAN_PHYSICS_2D)

import fan.types;
import fan.print;

#if defined(FAN_GUI)
  import fan.graphics.gui.base;
#endif

// higher the draw depth, less debug draws will be if maximum depth is 2^16
// so make sure your game objects do not pass this depth
static constexpr uint16_t draw_depth = 0xF000;

int z_depth = 0;

/// Draw a closed polygon provided in CCW order.
void DrawPolygon(const fan::vec2* vertices, int vertexCount, b2HexColor color, void* context) {

}

/// Draw a solid closed polygon provided in CCW order.
void DrawSolidPolygon(
  b2Transform transform,
  const b2Vec2* vertices,
  int vertexCount,
  f32_t radius,
  b2HexColor color,
  void* context)
{
  if (vertexCount == 4) {
    static std::vector<fan::vec2> fan_vertices;
    fan::physics::b2_to_fan_vertices(vertices, vertexCount, fan_vertices);
    if (fan::physics::is_rectangle(fan_vertices)) {
      fan::graphics::add_shape_to_immediate_draw(fan::graphics::rectangle_t{{
        .render_view = &fan::graphics::physics::debug_render_view,
        .position = fan::vec3(fan::physics::physics_to_render(transform.p), draw_depth + z_depth),
        .size = fan::vec2(fan_vertices[2] - fan_vertices[0]).abs() / 2.f,
        .color = fan::color::from_rgb(color).set_alpha(0.5),
        .angle = std::atan2(transform.q.s, transform.q.c),
        .enable_culling = false
      }});
      return;
    }
  }

  static std::vector<fan::graphics::vertex_t> vs;
  vs.resize(vertexCount);

  for (auto [i, v] : fan::enumerate(vs)) {
    v.position = fan::physics::physics_to_render(vertices[i]);
    v.color = fan::color::from_rgb(color).set_alpha(0.5);
  }

  fan::graphics::add_shape_to_immediate_draw(fan::graphics::polygon_t {{
    .render_view = &fan::graphics::physics::debug_render_view,
    .position = fan::vec3(fan::physics::physics_to_render(transform.p), draw_depth + z_depth),
    .vertices = vs,
    .draw_mode = fan::graphics::primitive_topology_t::triangle_fan,
    .enable_culling = false
  }});
}

/// Draw a circle.
void DrawCircle(b2Vec2 center, f32_t radius, b2HexColor color, void* context) {
  fan::graphics::add_shape_to_immediate_draw(fan::graphics::circle_t {{
    .render_view = &fan::graphics::physics::debug_render_view,
    .position = fan::vec3(fan::physics::physics_to_render(center), draw_depth + z_depth),
    .radius = (f32_t)fan::physics::physics_to_render(radius).x,
    .color = fan::color::from_rgb(color).set_alpha(0.5),
    .enable_culling = false
  }});
}

/// Draw a solid circle.
void DrawSolidCircle(b2Transform transform, f32_t radius, b2HexColor color, void* context) {
  fan::graphics::add_shape_to_immediate_draw(fan::graphics::circle_t {{
    .render_view = &fan::graphics::physics::debug_render_view,
    .position = fan::vec3(fan::physics::physics_to_render(transform.p), draw_depth + z_depth),
    .radius = (f32_t)fan::physics::physics_to_render(radius).x,
    .color = fan::color::from_rgb(color).set_alpha(0.5),
    .enable_culling = false
  }});
}

/// Draw a capsule.
void DrawCapsule(b2Vec2 p1, b2Vec2 p2, f32_t radius, b2HexColor color, void* context) {
  printf("DrawCapsule\n");
}

/// Draw a solid capsule.
void DrawSolidCapsule(b2Vec2 p1, b2Vec2 p2, f32_t radius, b2HexColor color, void* context) {
  fan::graphics::add_shape_to_immediate_draw(fan::graphics::capsule_t {{
    .render_view = &fan::graphics::physics::debug_render_view,
    .position = fan::vec3(0, 0, draw_depth + z_depth),
    .center0 = fan::physics::physics_to_render(p1),
    .center1 = fan::physics::physics_to_render(p2),
    .radius = (f32_t)fan::physics::physics_to_render(radius).x,
    .color = fan::color::from_rgb(color).set_alpha(0.5),
    .enable_culling = false
  }});
}


/// Draw a line segment.
void DrawSegment(b2Vec2 p1, b2Vec2 p2, b2HexColor color, void* context) {
  fan::graphics::add_shape_to_immediate_draw(fan::graphics::line_t {{
    .render_view = &fan::graphics::physics::debug_render_view,
    .src = fan::vec3(fan::physics::physics_to_render(p1), draw_depth + z_depth),
    .dst = fan::vec3(fan::physics::physics_to_render(p2), draw_depth + z_depth),
    .color = fan::color::from_rgb(color),
    .enable_culling = false
  }});
}

/// Draw a transform. Choose your own length scale.
void DrawTransform(b2Transform transform, void* context) {

}

/// Draw a point.
void DrawPoint(b2Vec2 p, f32_t size, b2HexColor color, void* context) {
  fan::graphics::add_shape_to_immediate_draw(fan::graphics::circle_t {{
    .render_view = &fan::graphics::physics::debug_render_view,
    .position = fan::vec3(fan::physics::physics_to_render(p), draw_depth + z_depth),
    .radius = size / 2.f,
    .color = fan::color::from_rgb(color).set_alpha(0.5),
    .enable_culling = false
  }});
}

/// Draw a string.
void DrawString(b2Vec2 p, const char* s, b2HexColor color, void* context) {
#if defined(FAN_GUI)
  f32_t base_font_size = 14.0f;
  f32_t zoomed_font_size = base_font_size * fan::graphics::camera_get_zoom(fan::graphics::physics::debug_render_view.camera);
  f32_t fs_first = fan::graphics::gui::font_sizes[0];
  f32_t fs_last = fan::graphics::gui::font_sizes[std::size(fan::graphics::gui::font_sizes) - 1];

  zoomed_font_size = std::clamp(zoomed_font_size, fs_first, fs_last);

  fan::graphics::gui::push_font(
    fan::graphics::gui::get_font(zoomed_font_size)
  );

  fan::graphics::gui::text_outlined_at(s, fan::graphics::world_to_screen(fan::physics::physics_to_render(p)), fan::color::from_rgb(color));
  fan::graphics::gui::pop_font();
#endif
}

bool g_debug_draw_enabled = false;

b2DebugDraw initialize_debug(bool enabled) {
  g_debug_draw_enabled = enabled;
  return b2DebugDraw {
    .DrawPolygonFcn = (decltype(b2DebugDraw::DrawPolygonFcn))DrawPolygon,
    .DrawSolidPolygonFcn = DrawSolidPolygon,
    .DrawCircleFcn = DrawCircle,
    .DrawSolidCircleFcn = DrawSolidCircle,
    //.DrawCapsuleFcn = DrawCapsule,
    .DrawSolidCapsuleFcn = DrawSolidCapsule,
    .DrawSegmentFcn = DrawSegment,
    .DrawTransformFcn = DrawTransform,
    .DrawPointFcn = DrawPoint,
    .DrawStringFcn = DrawString,
    //	.drawShapes = enabled,
    //	.drawJoints = enabled,
    //	.drawJointExtras = enabled,
    ////	.drawAABBs = enabled,
    //	.drawMass = enabled,
    //	.drawContacts = enabled,
    //	.drawGraphColors = enabled,
    //	.drawContactNormals = enabled,
    //	.drawContactImpulses = enabled,
    //	.drawFrictionImpulses = enabled,

    /*.drawShapes = enabled,
    .drawJoints = enabled,
    .drawAABBs = enabled,
    .drawContacts=enabled*/


    .drawShapes = enabled,
    .drawJoints = enabled,
    .drawJointExtras = enabled,
    .drawBounds = enabled,
    .drawMass = enabled,
    .drawBodyNames = enabled,
    .drawContacts = enabled,
    .drawGraphColors = enabled,
    .drawContactNormals = enabled,
    .drawContactImpulses = enabled,
    .drawContactFeatures = enabled,
    .drawFrictionImpulses = enabled,
    .drawIslands = enabled,
  };
}

namespace fan::graphics::physics {

  void init() {
    static bool init_ = true;
    if (!init_) {
      return;
    }
    if (fan::graphics::physics::debug_render_view.viewport.iic()) {
      fan::graphics::physics::debug_render_view = fan::graphics::get_orthographic_render_view();
    }
    init_ = false;
    box2d_debug_draw = [] {
      fan::physics::gphysics()->debug_draw_cb = []() {
        b2World_Draw(fan::physics::gphysics()->world_id, &box2d_debug_draw);
      };
      return initialize_debug(false);
    }();
  }

  void frame_init_lazy() {
    static fan::graphics::update_callback_nr_t uc_nr;
    if (uc_nr) {
      return;
    }
    auto& uc = *fan::graphics::ctx().update_callback;
    uc_nr = uc.NewNodeLast();
    // called every frame
    uc[uc_nr] = [] (auto* loco) {
      if (fan::window::is_input_action_active("debug_physics")) {
        
        fan::graphics::physics::debug_draw(!fan::graphics::physics::get_debug_draw());
      }
    };
  }

  void debug_draw(bool enabled) {
    init();
    fan::graphics::physics::box2d_debug_draw = initialize_debug(enabled);
  }
  bool get_debug_draw() {
    return g_debug_draw_enabled;
  }

  void shape_physics_update(const fan::physics::physics_update_data_t& data) {
    frame_init_lazy();
    if (!b2Body_IsValid(*(b2BodyId*)&data.body_id)) {
      //   fan::print("invalid body data (corruption)");
      return;
    }
    if (b2Body_GetType(*(b2BodyId*)&data.body_id) == b2_staticBody) {
      return;
    }

    fan::vec2 p = b2Body_GetWorldPoint(*(b2BodyId*)&data.body_id, fan::vec2(0));
    b2Rot rotation = b2Body_GetRotation(*(b2BodyId*)&data.body_id);
    f32_t radians = b2Rot_GetAngle(rotation);

    fan::graphics::shape_t& shape = *(fan::graphics::shape_t*)&data.shape_id;
    shape.set_position(fan::vec2((p)*fan::physics::length_units_per_meter + data.draw_offset));
    if (data.sync_visual_angle) {
      shape.set_angle(fan::vec3(0, 0, radians));
    }
    b2ShapeId id[1];
    if (b2Body_GetShapes(*(b2BodyId*)&data.body_id, id, 1)) {
      auto aabb = b2Shape_GetAABB(id[0]);
      fan::vec2 size = fan::vec2(aabb.upperBound - aabb.lowerBound) / 2;
      physics_update_cb(shape, shape.get_position(), size * fan::physics::length_units_per_meter / 2, radians);
    }
    //hitbox_visualize[(void*) & data.body_id] = fan::graphics::rectangle_t{{
    //    .position = fan::vec3(p * fan::physics::length_units_per_meter, 0xffff-100),
    //    .size = 30,
    //    .color = fan::colors::green
    //}};


    // joint debug

    /*  int joint_count = b2Body_GetJointCount(data.body_id);
    if (joint_count > 0) {
    std::vector<fan::physics::joint_id_t> joints(joint_count);
    b2Body_GetJoints(data.body_id, joints.data(), joint_count);
    joint_visualize[&shape].clear();
    for (fan::physics::joint_id_t joint_id : joints) {
    fan::vec2 anchor_a = b2Joint_GetLocalAnchorA(joint_id);
    fan::vec2 anchor_b = b2Joint_GetLocalAnchorB(joint_id);
    b2BodyId body_a = b2Joint_GetBodyA(joint_id);
    b2BodyId body_b = b2Joint_GetBodyB(joint_id);
    fan::vec2 world_anchor_a = b2Body_GetWorldPoint(body_a, anchor_a);
    fan::vec2 world_anchor_b = b2Body_GetWorldPoint(body_b, anchor_b);

    joint_visualize[&shape].emplace_back(fan::graphics::circle_t{{
    .position = fan::vec3(world_anchor_a * fan::physics::length_units_per_meter, 60002),
    .radius = 3,
    .color = fan::color(0, 0, 1, 0.5),
    .blending = true
    }});

    joint_visualize[&shape].emplace_back(fan::graphics::line_t{{
    .src = fan::vec3(p * fan::physics::length_units_per_meter, 60001),
    .dst = fan::vec3(world_anchor_b * fan::physics::length_units_per_meter, 60001),
    .color = fan::color(1, 0, 0, 0.5),
    .blending = true
    }});
    }
    }*/
  }

  mass_data_t::operator b2MassData() const {
    return b2MassData {.mass = mass, .center = center_of_mass, .rotationalInertia = rotational_inertia};
  }

  void base_shape_t::set_shape(fan::graphics::shape_t&& shape) {
    bool is_valid = iic() == false;
    /*fan::vec3 prev_pos;
    if (is_valid) {
    prev_pos = fan::graphics::shape_t::get_position();
    }*/
    if (physics_update_nr.iic() == false) {
      fan::physics::remove_physics_update(physics_update_nr);
    }
    *dynamic_cast<fan::graphics::shape_t*>(this) = std::move(shape);
    static_assert(sizeof(fan::graphics::shaper_t::ShapeID_t) < sizeof(uint64_t), "physics update shape_id too small");
    uint64_t body_id_data = *reinterpret_cast<uint64_t*>(dynamic_cast<body_id_t*>(this));
    physics_update_nr = fan::physics::add_physics_update({
      .shape_id = *(uint64_t*)this,
      .draw_offset = draw_offset,
      .body_id = body_id_data,
      .cb = (void*)shape_physics_update
      });
    /*if (is_valid) {
    set_position(prev_pos);
    }*/
  }

  base_shape_t::base_shape_t(fan::graphics::shape_t&& shape, fan::physics::entity_t&& entity, const mass_data_t& mass_data) :
    fan::graphics::shape_t(std::move(shape)),
    fan::physics::entity_t(std::move(entity)) {
    if (physics_update_nr.iic() == false) {
      fan::physics::remove_physics_update(physics_update_nr);
    }
    uint64_t body_id_data = *reinterpret_cast<uint64_t*>(dynamic_cast<body_id_t*>(this));
    physics_update_nr = fan::physics::add_physics_update({
      .shape_id = *(uint64_t*)this,
      .draw_offset = draw_offset,
      .body_id = body_id_data,
      .cb = (void*)shape_physics_update
      });
    b2MassData md = b2Body_GetMassData(*dynamic_cast<b2BodyId*>(this));
    mass_data_t md_copy = mass_data;
    if (mass_data.mass < 0.f) {
      md_copy.mass = md.mass;
    }
    if (mass_data.center_of_mass.x == 0 && mass_data.center_of_mass.y == 0) {
      md_copy.center_of_mass = md.center;
    }
    if (mass_data.rotational_inertia < 0.f) {
      md_copy.rotational_inertia = md.rotationalInertia;
    }
    b2Body_SetMassData(*dynamic_cast<b2BodyId*>(this), md_copy);
  }

  base_shape_t::base_shape_t(const base_shape_t& r) : fan::graphics::shape_t(r), fan::physics::entity_t(r) {
    //if (this != )
    fan::physics::body_id_t new_body_id = fan::physics::deep_copy_body(fan::physics::gphysics()->world_id, *dynamic_cast<const fan::physics::body_id_t*>(&r));
    if (!B2_ID_EQUALS(r, (*this))) {
      destroy();
    }
    set_body(new_body_id);
    b2Body_GetWorldPoint(*dynamic_cast<b2BodyId*>(this), fan::vec2(0));
    if (physics_update_nr.iic() == false) {
      fan::physics::remove_physics_update(physics_update_nr);
      physics_update_nr.sic();
    }
    if (!fan::physics::entity_t::is_valid()) {
      return;
    }
    uint64_t body_id_data = *reinterpret_cast<uint64_t*>(dynamic_cast<body_id_t*>(this));
    physics_update_nr = fan::physics::add_physics_update({
      .shape_id = *(uint64_t*)this,
      .draw_offset = draw_offset,
      .body_id = body_id_data,
      .cb = (void*)shape_physics_update
      });
  }

  base_shape_t::base_shape_t(base_shape_t&& r) : fan::graphics::shape_t(std::move(r)), fan::physics::entity_t(std::move(r)) {
    if (!B2_ID_EQUALS(r, (*this))) {
      destroy();
    }
    physics_update_nr = r.physics_update_nr;
    r.physics_update_nr.sic();
    r.set_body(b2_nullBodyId);
  }

  base_shape_t::~base_shape_t() {

    erase();
  }

  base_shape_t& base_shape_t::operator=(const base_shape_t& r) {
    if (physics_update_nr.iic() == false) {
      fan::physics::remove_physics_update(physics_update_nr);
      physics_update_nr.sic();
    }
    if (this != &r) {
      fan::graphics::shape_t::operator=(r);

      fan::physics::body_id_t new_body_id = fan::physics::deep_copy_body(fan::physics::gphysics()->world_id, *dynamic_cast<const fan::physics::body_id_t*>(&r));
      if (!B2_ID_EQUALS(r, (*this))) {
        destroy();
      }
      set_body(new_body_id);
      if (!fan::physics::entity_t::is_valid()) {
        return *this;
      }
      uint64_t body_id_data = *reinterpret_cast<uint64_t*>(dynamic_cast<body_id_t*>(this));
      physics_update_nr = fan::physics::add_physics_update({
        .shape_id = *(uint64_t*)this,
        .draw_offset = draw_offset,
        .body_id = body_id_data,
        .cb = (void*)shape_physics_update
        });
    }
    return *this;
  }

  base_shape_t& base_shape_t::operator=(base_shape_t&& r) {
    if (!B2_ID_EQUALS(r, (*this))) {
      destroy();
    }
    if (physics_update_nr.iic() == false) {
      fan::physics::remove_physics_update(physics_update_nr);
      physics_update_nr.sic();
    }
    if (this != &r) {
      fan::graphics::shape_t::operator=(std::move(r));
      fan::physics::entity_t::operator=(std::move(*dynamic_cast<fan::physics::entity_t*>(&r)));
      r.set_body(b2_nullBodyId);
      physics_update_nr = r.physics_update_nr;
      r.physics_update_nr.sic();
    }
    return *this;
  }

  void base_shape_t::erase() {
    fan::graphics::shape_t::erase();
    fan::physics::entity_t::destroy();
    if (physics_update_nr.iic() == false) {
      fan::physics::remove_physics_update(physics_update_nr);
    }
    physics_update_nr.sic();
  }

  fan::vec2 base_shape_t::get_draw_offset() const {
    return fan::physics::gphysics()->physics_updates[physics_update_nr].draw_offset;
  }
  void base_shape_t::set_draw_offset(fan::vec2 new_draw_offset) {
    draw_offset = new_draw_offset;
    fan::physics::gphysics()->physics_updates[physics_update_nr].draw_offset = new_draw_offset;
  }

  void base_shape_t::sync_visual_angle(bool flag) {
    fan::physics::gphysics()->physics_updates[physics_update_nr].sync_visual_angle = flag;
  }

  fan::vec3 base_shape_t::get_position() const {
    // The visual position might have not been updated (maybe) so use physics position but visual Z position
    //return fan::vec3(fan::physics::entity_t::get_position(), fan::graphics::shape_t::get_position().z);
    return fan::graphics::shape_t::get_position();
  }
  // used for camera
  fan::vec3 base_shape_t::get_physics_position() const {
    return fan::vec3(fan::physics::entity_t::get_position(), fan::graphics::shape_t::get_position().z);
  }
  rectangle_t::properties_t::operator fan::graphics::rectangle_properties_t() const {
    return fan::graphics::rectangle_properties_t {
      .render_view = render_view,
      .position = position,
      .size = size,
      .color = color,
      .outline_color = outline_color,
      .angle = angle,
      .rotation_point = rotation_point,
      .blending = blending
    };
  }
  rectangle_t::rectangle_t(const rectangle_t::properties_t& p) : base_shape_t(
    fan::graphics::shape_t(fan::graphics::rectangle_t {p}),
    fan::physics::entity_t(fan::physics::gphysics()->create_box(p.position, p.size, p.angle.z, p.body_type, p.shape_properties)),
    p.mass_data
  ) {}
  rectangle_t::rectangle_t(const rectangle_t& r) : base_shape_t(r) {}
  rectangle_t& rectangle_t::operator=(const rectangle_t& r) {
    base_shape_t::operator=(r);
    return *this;
  }
  rectangle_t& rectangle_t::operator=(rectangle_t&& r) {
    base_shape_t::operator=(std::move(r));
    return *this;
  }
  sprite_t::sprite_t(const properties_t& p) : base_shape_t(
    fan::graphics::shape_t(fan::graphics::sprite_t {p}),
    fan::physics::entity_t(fan::physics::gphysics()->create_box(p.position, p.size, p.angle.z, p.body_type, p.shape_properties)),
    p.mass_data
  ) {}

  sprite_t::sprite_t(const sprite_t& r) : base_shape_t(r) {}

  sprite_t::sprite_t(sprite_t&& r) : base_shape_t(std::move(r)) {}

  sprite_t& sprite_t::operator=(const sprite_t& r) {
    base_shape_t::operator=(r);
    return *this;
  }

  sprite_t& sprite_t::operator=(sprite_t&& r) {
    base_shape_t::operator=(std::move(r));
    return *this;
  }

  circle_t::circle_t(const properties_t& p) : base_shape_t(
    fan::graphics::shape_t(fan::graphics::circle_t {p}),
    fan::physics::entity_t(fan::physics::gphysics()->create_circle(p.position, p.radius, p.angle.z, p.body_type, p.shape_properties)),
    p.mass_data
  ) {}

  circle_t::circle_t(const circle_t& r) : base_shape_t(r) {}

  circle_t::circle_t(circle_t&& r) : base_shape_t(std::move(r)) {}

  circle_t& circle_t::operator=(const circle_t& r) {
    base_shape_t::operator=(r);
    return *this;
  }

  circle_t& circle_t::operator=(circle_t&& r) {
    base_shape_t::operator=(std::move(r));
    return *this;
  }

  circle_sprite_t::circle_sprite_t(const properties_t& p) : base_shape_t(
    fan::graphics::shape_t(fan::graphics::sprite_t {p}),
    fan::physics::entity_t(fan::physics::gphysics()->create_circle(p.position, p.radius, p.angle.z, p.body_type, p.shape_properties)),
    p.mass_data
  ) {}

  circle_sprite_t::circle_sprite_t(const circle_sprite_t& r) : base_shape_t(r) {}

  circle_sprite_t::circle_sprite_t(circle_sprite_t&& r) : base_shape_t(std::move(r)) {}

  circle_sprite_t& circle_sprite_t::operator=(const circle_sprite_t& r) {
    base_shape_t::operator=(r);
    return *this;
  }

  circle_sprite_t& circle_sprite_t::operator=(circle_sprite_t&& r) {
    base_shape_t::operator=(std::move(r));
    return *this;
  }

  capsule_t::capsule_t(const properties_t& p) : base_shape_t(
    fan::graphics::shape_t(fan::graphics::capsule_t {p}),
    fan::physics::entity_t(fan::physics::gphysics()->create_capsule(p.position, p.angle.z,
      b2Capsule {.center1 = p.center0, .center2 = p.center1, .radius = p.radius}, p.body_type, p.shape_properties)),
    p.mass_data
  ) {}

  capsule_t::capsule_t(const capsule_t& r) : base_shape_t(r) {}

  capsule_t::capsule_t(capsule_t&& r) : base_shape_t(std::move(r)) {}

  capsule_t& capsule_t::operator=(const capsule_t& r) {
    base_shape_t::operator=(r);
    return *this;
  }

  capsule_t& capsule_t::operator=(capsule_t&& r) {
    base_shape_t::operator=(std::move(r));
    return *this;
  }

  capsule_sprite_t::capsule_sprite_t(const properties_t& p) : base_shape_t(
    fan::graphics::shape_t(fan::graphics::sprite_t {p}),
    fan::physics::entity_t(fan::physics::gphysics()->create_capsule(p.position, p.angle.z, b2Capsule {
      .center1 = p.center0 * p.aabb_scale,
      .center2 = p.center1 * p.aabb_scale,
      .radius = p.size.max() / 2.f * p.aabb_scale.max()
      }, p.body_type, p.shape_properties)),
    p.mass_data
  ) {}

  capsule_sprite_t::capsule_sprite_t(const capsule_sprite_t& r) : base_shape_t(r) {}

  capsule_sprite_t::capsule_sprite_t(capsule_sprite_t&& r) : base_shape_t(std::move(r)) {}

  capsule_sprite_t& capsule_sprite_t::operator=(const capsule_sprite_t& r) {
    base_shape_t::operator=(r);
    return *this;
  }

  capsule_sprite_t& capsule_sprite_t::operator=(capsule_sprite_t&& r) {
    base_shape_t::operator=(std::move(r));
    return *this;
  }

  polygon_t::polygon_t(const properties_t& p) : base_shape_t(
    fan::graphics::shape_t(fan::graphics::polygon_t {p}),
    fan::physics::entity_t(
      [&] {
    std::vector<fan::vec2> points(p.vertices.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
      points[i] = p.vertices[i].position;
    }
    return fan::physics::gphysics()->create_polygon(
      p.position,
      p.radius,
      points.data(), points.size(), p.body_type, p.shape_properties
    );
  }()),
    p.mass_data
  ) {}

  polygon_t::polygon_t(const polygon_t& r) : base_shape_t(r) {}

  polygon_t::polygon_t(polygon_t&& r) : base_shape_t(std::move(r)) {}

  polygon_t& polygon_t::operator=(const polygon_t& r) {
    base_shape_t::operator=(r);
    return *this;
  }

  polygon_t& polygon_t::operator=(polygon_t&& r) {
    base_shape_t::operator=(std::move(r));
    return *this;
  }

  polygon_strip_t::polygon_strip_t(const properties_t& p) : base_shape_t(
    fan::graphics::shape_t(fan::graphics::polygon_t {p}),
    fan::physics::entity_t(
      [&] {
    std::vector<fan::vec2> points(p.vertices.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
      points[i] = p.vertices[i].position;
    }
    return fan::physics::gphysics()->create_segment(
      p.position,
      points, p.body_type, p.shape_properties
    );
  }()),
    p.mass_data
  ) {}

  polygon_strip_t::polygon_strip_t(const polygon_strip_t& r) : base_shape_t(r) {}

  polygon_strip_t::polygon_strip_t(polygon_strip_t&& r) : base_shape_t(std::move(r)) {}

  polygon_strip_t& polygon_strip_t::operator=(const polygon_strip_t& r) {
    base_shape_t::operator=(r);
    return *this;
  }

  polygon_strip_t& polygon_strip_t::operator=(polygon_strip_t&& r) {
    base_shape_t::operator=(std::move(r));
    return *this;
  }

  std::array<fan::graphics::physics::rectangle_t, 4> create_stroked_rectangle(
    const fan::vec2& center_position,
    const fan::vec2& half_size,
    f32_t thickness,
    const fan::color& wall_color,
    uint8_t body_type,
    std::array<fan::physics::shape_properties_t, 4> shape_properties
  ){
    std::array<fan::graphics::physics::rectangle_t, 4> walls;
    const fan::color outline = wall_color * 2;

    std::array<fan::vec2, 4> positions {{
        fan::vec2(center_position.x, center_position.y - half_size.y),
        fan::vec2(center_position.x, center_position.y + half_size.y),
        fan::vec2(center_position.x - half_size.x, center_position.y),
        fan::vec2(center_position.x + half_size.x, center_position.y)
      }};

    std::array<fan::vec2, 4> sizes {{
        fan::vec2(half_size.x, thickness),
        fan::vec2(half_size.x, thickness),
        fan::vec2(thickness, half_size.y),
        fan::vec2(thickness, half_size.y)
      }};

    for (uint32_t i = 0; i < 4; i++) {
      walls[i] = fan::graphics::physics::rectangle_t {{
        .position = positions[i],
        .size = sizes[i],
        .color = wall_color,
        .outline_color = outline,
        .body_type = body_type,
        .shape_properties = shape_properties[i]
      }};
    }

    return walls;
  }



  std::array<rectangle_t, 4> create_walls(
    const fan::vec2& bounds,
    f32_t thickness,
    const fan::color& wall_color,
    uint8_t body_type,
    std::array<fan::physics::shape_properties_t, 4> shape_properties
  ) {
    return create_stroked_rectangle(
      bounds,
      bounds,
      thickness,
      wall_color,
      body_type,
      shape_properties
    );
  }

  //void movement_state_t::move_to_direction_raw(fan::physics::body_id_t body, const fan::vec2& direction) {
  //  fan::vec2 input_dir = direction.sign() 
  //  #if defined(FAN_GUI)
  //    * (check_gui ? !fan::graphics::gui::want_io() : 1)
  //  #endif
  //  ;
  //  fan::vec2 vel = body.get_linear_velocity();
  //  f32_t dt = fan::physics::default_physics_timestep;

  //  if (input_dir.x != 0) {
  //    vel.x += input_dir.x * accelerate_force * dt * 100.f;
  //    vel.x = fan::math::clamp(vel.x, -max_speed, max_speed);
  //  }
  //  else {
  //    f32_t deceleration_factor = 0.3f * dt * 100.f;
  //    vel.x = fan::math::lerp(vel.x, 0.f, deceleration_factor);
  //  }
  //  if (input_dir.y != 0) {
  //    vel.y += input_dir.y * accelerate_force * dt * 100.f;
  //    vel.y = fan::math::clamp(vel.y, -max_speed, max_speed);
  //  }
  //  else {
  //    f32_t deceleration_factor = 0.3f * dt * 100.f;
  //    vel.y = fan::math::lerp(vel.y, 0.f, deceleration_factor);
  //  }
  //  last_direction = direction;
  //  body.set_linear_velocity({vel.x, vel.y});
  //}

  void movement_state_t::move_to_direction_raw(fan::physics::body_id_t body, const fan::vec2& direction) {
    fan::vec2 input_dir = direction.min(fan::vec2(1.f));
    fan::vec2 vel = body.get_linear_velocity();
    f32_t dt = fan::physics::default_physics_timestep;

    f32_t accel = accelerate_force * dt * 100.f;
    f32_t decel = 0.3f * dt * 100.f;

    accel = fan::math::clamp(accel, 0.f, 1.f);
    decel = fan::math::clamp(decel, 0.f, 1.f);

    if (input_dir.x != 0.f) {
      f32_t target_x = input_dir.x * max_speed;
      vel.x = fan::math::lerp(vel.x, target_x, accel);
    }
    else {
      vel.x = fan::math::lerp(vel.x, 0.f, decel);
    }

    if (input_dir.y != 0.f) {
      f32_t target_y = input_dir.y * max_speed;
      vel.y = fan::math::lerp(vel.y, target_y, accel);
    }

    last_direction = direction;
    body.set_linear_velocity(vel);
  }
  void movement_state_t::move_to_direction(fan::physics::body_id_t body, const fan::vec2& direction) {
    fan::vec2 input_dir = direction.min(fan::vec2(1.f))
    #if defined(FAN_GUI)
      * (check_gui ? !fan::graphics::gui::want_io() : 1)
    #endif
    ;

    last_direction = direction;

    fan::vec2 vel = body.get_linear_velocity();
    f32_t dt = fan::physics::default_physics_timestep;

    f32_t accel = accelerate_force * dt * 100.f;
    accel = fan::math::clamp(accel, 0.f, 1.f);

    f32_t target_x = input_dir.x * max_speed;
    vel.x = fan::math::lerp(vel.x, target_x, accel);

    body.set_linear_velocity({vel.x, vel.y});
  }

  void movement_state_t::update_ai_orientation(character2d_t& character, const fan::vec2& distance) {
    static constexpr f32_t FACING_DEAD_ZONE = 30.f;
  
    if (distance.x > FACING_DEAD_ZONE) {
      desired_facing.x = 1;
    } 
    else if (distance.x < -FACING_DEAD_ZONE) {
      desired_facing.x = -1;
    }
    last_direction = distance.normalized();
  }

  void movement_state_t::perform_jump(fan::physics::body_id_t body_id, bool jump_condition, fan::vec2* wall_jump_normal, wall_jump_t* wall_jump) {
    bool on_ground = fan::physics::is_on_ground(body_id, jump_state.jumping);
    if (on_ground) {
      jump_state.last_ground_time = fan::time::now();
      jump_state.reset();
      if (wall_jump) {
        wall_jump->consumed = false;
      }
    }
    if (!jump_condition || !jump_state.handle_jump) {
      jump_state.jumping = false;
      return;
    }

    fan::physics::shape_id_t colliding_wall_id;
    fan::vec2 wall_normal = wall_jump_normal ? *wall_jump_normal
      : fan::physics::check_wall_contact(body_id, &colliding_wall_id);
    bool touching_wall = (wall_normal.x != 0 || wall_normal.y != 0);
    bool allow_coyote = jump_state.can_coyote_jump(fan::time::now());

    if (touching_wall && !on_ground && jump_state.consumed && wall_jump && !wall_jump->consumed) {
      fan::vec2 input = fan::window::get_input_vector() 
      #if defined(FAN_GUI)
        * (check_gui ? !fan::graphics::gui::want_io() : 1)
      #endif
      ;
      bool pushing_into_wall = fan::math::sgn(input.x) == fan::math::sgn(wall_normal.x);
      if (!jump_state.jumping) {
        fan::vec2 vel = body_id.get_linear_velocity();
        body_id.set_linear_velocity(fan::vec2(vel.x, 0));
        if (1) {
          if (fan::physics::wall_jump(body_id, wall_normal, wall_jump->push_away_force, jump_state.impulse)) {
            jump_state.on_jump(2);
          }
        }
        jump_state.jumping = true;
        jump_state.on_air_after_jump = true;
        wall_jump->consumed = true;
        jump_state.double_jump_consumed = true;
        return;
      }
    }

    if ((!jump_state.consumed && !jump_state.jumping) 
    #if defined(FAN_GUI)
      * (check_gui ? !fan::graphics::gui::want_io() : 1)
    #endif
    ) {
      fan::vec2 vel = body_id.get_linear_velocity();
      body_id.set_linear_velocity(fan::vec2(vel.x, 0));
      body_id.apply_linear_impulse_center({0, -jump_state.impulse});
      jump_state.on_jump(0);
      jump_state.jumping = true;
      jump_state.consumed = true;
      jump_state.on_air_after_jump = true;
      return;
    }

    if ((jump_state.allow_double_jump && !jump_state.jumping && jump_state.consumed && !jump_state.double_jump_consumed) 
    #if defined(FAN_GUI)
      * (check_gui ? !fan::graphics::gui::want_io() : 1)
    #endif
    ) {
      fan::vec2 vel = body_id.get_linear_velocity();
      body_id.set_linear_velocity(fan::vec2(vel.x, 0));
      body_id.apply_linear_impulse_center({0, -jump_state.impulse});
      jump_state.on_jump(1);
      jump_state.jumping = true;
      jump_state.double_jump_consumed = true;
    }
  }

  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  bool attack_state_t::can_attack(const fan::vec2& target_distance) {
    return cooldown_timer &&
      std::abs(target_distance.x) <= attack_range.x &&
      std::abs(target_distance.y) <= attack_range.y;
  }
  bool attack_state_t::try_attack(character2d_t* character) {
    if (is_attacking) {
      return true;
    }
    if (cooldown_timer) {
      cooldown_timer.restart();
      is_attacking = true;
      if (on_attack_start) {
        on_attack_start();
      }
      return true;
    }
    return false;
  }
  bool attack_state_t::try_attack(character2d_t* character, const fan::vec2& target_distance) {
    if (is_attacking) {
     // update_ai_orientation(*character, target_distance);
      return true;
    }
    fan::vec2 sign = character->get_image_sign();
    int8_t facing = (int8_t)fan::math::sgn(sign.x);
    int8_t desired = (int8_t)fan::math::sgn(target_distance.x);

    if (attack_requires_facing_target && !is_attacking && desired != 0 && facing != desired) {
      return false;
    }

    if (can_attack(target_distance)) {
      character->movement_state.update_ai_orientation(*character, target_distance);
      cooldown_timer.restart();
      is_attacking = true;
      if (on_attack_start) {
        on_attack_start();
      }
      return true;
    }
    return false;
  }
  void attack_state_t::end_attack() {
    std::fill(attack_used.begin(), attack_used.end(), false);
    cooldown_timer.restart();
    if (is_attacking && on_attack_end) {
      on_attack_end();
    }
    is_attacking = false;
  }

  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  bool navigation_helper_t::detect_and_handle_obstacles(
    character2d_t* character,
    const ai_behavior_t& ai_behavior,
    const fan::vec2& distance,
    fan::vec2 tile_size
  ) {
    character->movement_state.is_wall_sliding = false;

    if (!auto_jump_obstacles) {
      return false;
    }

    fan::vec2 physics_pos = character->get_physics_position();
    fan::vec2 current_vel = character->get_linear_velocity();

    tile_size.x = std::copysign(tile_size.x / jump_lookahead_tiles, distance.x);
    bool is_cliff = !fan::physics::is_point_overlapping(physics_pos + tile_size);

    bool trying_to_move = std::abs(distance.x) > 0.1f;
    bool velocity_blocked = std::abs(current_vel.x) < 10.f;
    bool is_hitting_obstacle = trying_to_move && velocity_blocked;

    if (prev_x == distance.x && character->movement_state.jump_state.jumping) {
      was_jumping = true;
    }
    else if (!character->movement_state.jump_state.jumping) {
      was_jumping = false;
    }
    is_stuck_state = character->movement_state.jump_state.jumping && was_jumping;

    fan::physics::shape_id_t colliding_wall_id;
    fan::vec2 wall_normal = fan::physics::check_wall_contact(*character, &colliding_wall_id);
    bool touching_wall = (wall_normal.x != 0 || wall_normal.y != 0);

    f32_t diff = std::abs(std::abs(prev_x) - std::abs(distance.x));
    if (diff < 3.f && touching_wall) {
      is_stuck_state = true;
    }
    else {
      is_stuck_state = false;
      stuck_timer.restart();
    }

    if (touching_wall && !character->is_on_ground() && trying_to_move) {
      bool pushing_into_wall = fan::math::sgn(distance.x) == fan::math::sgn(wall_normal.x);
      if (pushing_into_wall && current_vel.y > 0) {
        fan::physics::apply_wall_slide(*character, wall_normal, character->wall_jump.slide_speed);
        character->movement_state.is_wall_sliding = true;
      }
    }

    bool should_jump = false;

    if (is_cliff && !is_stuck_state) {
      should_jump = true;
    }
    else if ((is_hitting_obstacle || touching_wall) && stuck_timer.finished()) {
      should_jump = true;
      stuck_timer.restart();
    }
    else if (on_check_obstacle(physics_pos + fan::vec2(tile_size.x, 0))) {
      should_jump = true;
    }

    if (should_jump && ai_behavior.type != ai_behavior_t::patrol) {
      character->movement_state.perform_jump(*character, true, &wall_normal, &character->wall_jump);
    }

    prev_x = distance.x;
    
    return is_stuck_state;
  }

  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  void animation_controller_t::add_state(const animation_state_t& state) {
    states.emplace_back(state);
  }
  void animation_controller_t::update(character2d_t* character) {
    update_animation(character);

    for (auto& state : states) {
      bool triggered = false;
      bool attack_cond = state.name == "attack0";
      triggered = state.condition(*character);
      if (state.trigger_type == animation_state_t::one_shot) {
        if (triggered && !state.is_playing) {
          for (auto& other : states) {
            other.is_playing = false;
          }
          state.is_playing = true;
          if (prev_animation_id != state.animation_id) {
            character->set_current_animation_id(state.animation_id);
            character->reset_current_sprite_sheet_animation();
            character->anim_controller.current_animation_requires_velocity_fps = state.velocity_based_fps;
            if (!state.velocity_based_fps) {
              character->set_sprite_sheet_fps(state.fps);
            }

            prev_animation_id = state.animation_id;
          }
        }
        if (character->attack_state.knockback_force == 10.f && state.is_playing && state.name == "attack0") {

        }
        if (state.is_playing && character->is_animation_finished(state.animation_id)) {
          state.is_playing = false;
          if (state.name == "attack0" && character->attack_state.is_attacking) {
            character->attack_state.end_attack();
          }
          continue;
        }
        if (state.is_playing) {
          return;
        }
      }
      else if (triggered) {
        if (prev_animation_id != state.animation_id || character->get_current_animation_id() != state.animation_id) {
          character->set_current_animation_id(state.animation_id);
          character->reset_current_sprite_sheet_animation();
          character->anim_controller.current_animation_requires_velocity_fps = state.velocity_based_fps;
          if (!state.velocity_based_fps) {
            character->set_sprite_sheet_fps(state.fps);
          }
        }
        if (state.velocity_based_fps) {
          f32_t speed = fan::math::clamp((f32_t)character->movement_state.last_direction.length() + 0.35f, 0.f, 1.f);
          character->set_sprite_sheet_fps(state.fps * speed);
        }
        prev_animation_id = state.animation_id;
        return;
      }
    }
  }
  void animation_controller_t::cancel_current() {
    for (auto& state : states) {
      state.is_playing = false;
    }
  }

  animation_controller_t::animation_state_t& animation_controller_t::get_state(const std::string& name) {
    for (auto& state : states) {
      if (name == state.name) {
        return state;
      }
    }
    fan::throw_error("state not found");
    __unreachable();
  }

  void animation_controller_t::update_animation(character2d_t* character) {
    if (fan::graphics::gui::want_io()) {
      return;
    }

    fan::vec2 sign = character->get_image_sign();
    fan::vec2 dir = character->movement_state.last_direction;

    if (dir.x > 0) {
      if (sign.x < 0) character->set_image_sign({1, sign.y});
      character->movement_state.desired_facing.x = 1;
      return;
    }
    if (dir.x < 0) {
      if (sign.x > 0) character->set_image_sign({-1, sign.y});
      character->movement_state.desired_facing.x = -1;
      return;
    }
  
    if (character->movement_state.desired_facing.x != 0) {
      int desired = (int)fan::math::sgn(character->movement_state.desired_facing.x);
      if ((int)fan::math::sgn(sign.x) != desired) {
        character->set_image_sign({(f32_t)desired, sign.y});
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  fan::vec2 ai_behavior_t::get_target_distance(const fan::vec2& current_position) const {
    if (!target) {
      return fan::vec2(0);
    }
    return target->get_physics_position() - current_position;
  }
  bool ai_behavior_t::should_move(const fan::vec2& distance) const {
    return (
      (std::abs(distance.x) < trigger_distance.x &&
        std::abs(distance.y) < trigger_distance.y) &&
      !(std::abs(distance.x) < closeup_distance.x &&
        std::abs(distance.y) < closeup_distance.y)
    );
  }
  void ai_behavior_t::enable_ai_follow(character2d_t* target, const fan::vec2& trigger_distance, const fan::vec2& closeup_distance) {
    this->type = ai_behavior_t::follow_target;
    this->target = target;
    this->trigger_distance = trigger_distance;
    this->closeup_distance = closeup_distance;
  }

  void ai_behavior_t::enable_ai_flee(character2d_t* target, const fan::vec2& trigger_distance, const fan::vec2& closeup_distance) {
    this->type = ai_behavior_t::flee_from_target;
    this->target = target;
    this->trigger_distance = trigger_distance;
    this->closeup_distance = closeup_distance;
  }

  void ai_behavior_t::enable_ai_patrol(const std::vector<fan::vec2>& points) {
    this->type = ai_behavior_t::patrol;
    this->patrol_points = points;
  }

  bool is_character_above(const character2d_t& top_char, const character2d_t& bottom_char, f32_t tolerance = 20.f) {
    fan::vec2 top_pos = top_char.get_physics_position();
    fan::vec2 bottom_pos = bottom_char.get_physics_position();
    bool is_above = top_pos.y < bottom_pos.y - tolerance;
    bool is_aligned = std::abs(top_pos.x - bottom_pos.x) < bottom_char.get_size().x;
    return is_above && is_aligned;
  }

  void ai_behavior_t::update_ai(character2d_t* character, navigation_helper_t& navigation, const fan::vec2& target_position, fan::vec2 tile_size) {
    character->raycast(*target);
    fan::vec2 movement_direction(0);
    fan::physics::shape_id_t colliding_wall_id;
    wall_jump.normal = fan::physics::check_wall_contact(*character, &colliding_wall_id);
    bool on_ground = fan::physics::is_on_ground(*character, character->movement_state.jump_state.jumping);
    if (on_ground) {
      character->movement_state.jump_state.last_ground_time = fan::time::now();
      character->movement_state.jump_state.on_air_after_jump = false;
      character->movement_state.jump_state.consumed = false;
    }
    else {
      character->movement_state.jump_state.jumping = false;
    }
    switch (type) {
    case ai_behavior_t::follow_target:
    {
      if (!target) {
        break;
      }
      fan::vec2 distance = get_target_distance(character->get_physics_position());
      character->movement_state.update_ai_orientation(*character, distance);
      character->attack_state.try_attack(character, distance);
      if (should_move(distance)) {
        movement_direction.x = distance.sign().x;
        f32_t air_control_multiplier = on_ground ? 1.0f : 0.8f;
        if (wall_jump.normal.x && movement_direction.x) {
          colliding_wall_id.set_friction(0.f);
          fan::vec2 vel = character->get_linear_velocity();
          if (!character->movement_state.jump_state.jumping && vel.y > 0) {
            fan::physics::apply_wall_slide(*character, wall_jump.normal, wall_jump.slide_speed);
          }
        }
        else if (colliding_wall_id) {
          colliding_wall_id.set_friction(fan::physics::shape_properties_t().friction);
        }
        f32_t diff = std::abs(std::abs(navigation.prev_x) - std::abs(distance.x));
        if (diff < 10.f && navigation.stuck_timer && std::abs(distance.x) < 10.f) {
          break;
          character->movement_state.perform_jump(*character, true, &wall_jump.normal, &wall_jump);
          navigation.stuck_timer.restart();
        }
        bool is_stuck = navigation.detect_and_handle_obstacles(character, *this, distance, tile_size);
        if (!is_stuck) {
          character->movement_state.move_to_direction(*character, movement_direction * air_control_multiplier);
        }
      }
      break;
    }
    case ai_behavior_t::flee_from_target:
    {
      if (!target) {
        break;
      }
      fan::vec2 distance = get_target_distance(character->get_physics_position());
      if (should_move(distance)) {
        movement_direction.x = -distance.sign().x;
        if (wall_jump.normal.x && movement_direction.x) {
          colliding_wall_id.set_friction(0.f);
          fan::vec2 vel = character->get_linear_velocity();
          if (!character->movement_state.jump_state.jumping && vel.y > 0) {
            fan::physics::apply_wall_slide(*character, wall_jump.normal, wall_jump.slide_speed);
          }
        }
        else if (colliding_wall_id) {
          colliding_wall_id.set_friction(fan::physics::shape_properties_t().friction);
        }
        bool is_stuck = navigation.detect_and_handle_obstacles(character, *this, distance, tile_size);
        if (!is_stuck) {
          f32_t air_control = on_ground ? 1.0f : 0.8f;
          character->movement_state.move_to_direction(*character, movement_direction * air_control);
        }
      }
      break;
    }
    case ai_behavior_t::patrol:
    {
      if (patrol_points.empty()) {
        break;
      }
      fan::vec2 target_pos = patrol_points[current_patrol_index];
      fan::vec2 distance = target_pos - character->get_center();
      if (std::abs(distance.x) < 10.f) {
        current_patrol_index = (current_patrol_index + 1) % patrol_points.size();
      }
      else {
        movement_direction.x = distance.sign().x;
        if (wall_jump.normal.x && movement_direction.x) {
          colliding_wall_id.set_friction(0.f);
          fan::vec2 vel = character->get_linear_velocity();
          if (!character->movement_state.jump_state.jumping && vel.y > 0) {
            fan::physics::apply_wall_slide(*character, wall_jump.normal, wall_jump.slide_speed);
          }
        }
        else if (colliding_wall_id) {
          colliding_wall_id.set_friction(fan::physics::shape_properties_t().friction);
        }
        fan::vec2 vel = character->get_linear_velocity();
        if (std::abs(vel.x) < 5.f && navigation.stuck_timer) {
          current_patrol_index = (current_patrol_index + 1) % patrol_points.size();
          navigation.stuck_timer.restart();
        }
        bool is_stuck = navigation.detect_and_handle_obstacles(character, *this, distance, tile_size);
        if (!is_stuck) {
          f32_t air_control = on_ground ? 1.0f : 0.8f;
          f32_t old_max = character->movement_state.max_speed;
          //character->movement_state.max_speed /= 2.f; // use half speed when patrolling
          character->movement_state.move_to_direction(*character, movement_direction * air_control);
          //character->movement_state.max_speed = old_max;
        }
      }
      break;
    }
    }
    if (character->anim_controller.auto_update_animations) {
      character->anim_controller.update(character);
    }
  }


  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  fan::graphics::physics::character2d_t fan::graphics::physics::character2d_t::from_json(
    const fan::graphics::physics::character2d_t::character_config_t& config,
    const std::source_location& callers_path
  ) {

    fan::json json_data = fan::graphics::read_json(config.json_path, callers_path);
    fan::graphics::resolve_json_image_paths(json_data, config.json_path, callers_path);
    fan::graphics::parse_animations(config.json_path, json_data, callers_path);
    auto shape = fan::graphics::extract_single_shape(json_data, callers_path);
    fan::graphics::physics::character2d_t character;
    character.set_shape(std::move(shape));
    character.set_physics_body(
      fan::graphics::physics::character_capsule(
        character,
        config.aabb_scale,
        config.physics_properties
      )
    );

    if (config.auto_animations) {
      character.setup_default_animations(config);
    }

    character.player_sprite_sheet();

    return character;
  }

  character2d_t::character2d_t(const character2d_t& o)
    : base_shape_t(o),
    anim_controller(o.anim_controller),
    attack_state(o.attack_state),
    movement_state(o.movement_state),
    wall_jump(o.wall_jump),
    movement_cb_handle(o.movement_cb_handle),
    feet{o.feet[0], o.feet[1]}
  {
    if (movement_state.enabled) {
      movement_cb_handle.remove();
      movement_cb_handle = add_movement_callback([this]() {
        process_keyboard_movement(movement_state.type);
      });
    }
  }

  character2d_t::character2d_t(character2d_t&& o) noexcept
    : base_shape_t(std::move(o)),
    anim_controller(std::move(o.anim_controller)),
    attack_state(std::move(o.attack_state)),
    movement_state(std::move(o.movement_state)),
    wall_jump(std::move(o.wall_jump)),
    movement_cb_handle(std::move(o.movement_cb_handle)),
    feet{std::move(o.feet[0]), std::move(o.feet[1])}
  {}

  character2d_t& character2d_t::operator=(const character2d_t& o) {
    if (this != &o) {
      base_shape_t::operator=(o);
      anim_controller = o.anim_controller;
      attack_state = o.attack_state;
      movement_state = o.movement_state;
      wall_jump = o.wall_jump;
      movement_cb_handle = o.movement_cb_handle;
      feet[0] = o.feet[0];
      feet[1] = o.feet[1];

      if (movement_state.enabled) {
        movement_cb_handle.remove();
        movement_cb_handle = add_movement_callback([this]() {
          process_keyboard_movement(movement_state.type);
        });
      }
    }
    return *this;
  }

  character2d_t& character2d_t::operator=(character2d_t&& o) noexcept {
    if (this != &o) {
      base_shape_t::operator=(std::move(o));
      anim_controller = std::move(o.anim_controller);
      attack_state = std::move(o.attack_state);
      movement_state = std::move(o.movement_state);
      wall_jump = std::move(o.wall_jump);
      movement_cb_handle = std::move(o.movement_cb_handle);
      feet[0] = std::move(o.feet[0]);
      feet[1] = std::move(o.feet[1]);
    }
    return *this;
  }

  character2d_t::~character2d_t() {
    if (!movement_cb_handle.iic()) {
      movement_cb_handle.remove();
      movement_cb_handle.sic();
    }
  }

  fan::vec3 character2d_t::get_center() const {
    auto p = get_position();
    return fan::vec3(fan::vec2(p - get_draw_offset()), p.z);
  }
  void character2d_t::set_physics_position(const fan::vec2& p) {
    fan::physics::entity_t::set_physics_position(p);
    fan::graphics::shape_t::set_position(p);
  }
  void character2d_t::set_shape(fan::graphics::shape_t&& shape) {
    physics::base_shape_t::set_shape(std::move(shape));

    if (movement_state.enabled) {
      movement_cb_handle.remove();
      movement_cb_handle = add_movement_callback([this]() {
        process_keyboard_movement(movement_state.type);
      });
    }
  }
  void character2d_t::set_physics_body(fan::physics::entity_t&& entity) {
    if (physics_update_nr.iic() == false) {
      fan::physics::remove_physics_update(physics_update_nr);
    }

    *static_cast<fan::physics::entity_t*>(this) = std::move(entity);

    uint64_t body_id_data = *reinterpret_cast<uint64_t*>(static_cast<fan::physics::body_id_t*>(this));
    physics_update_nr = fan::physics::add_physics_update({
      .shape_id = *(uint64_t*)static_cast<fan::graphics::shape_t*>(this),
      .draw_offset = draw_offset,
      .body_id = body_id_data,
      .cb = (void*)shape_physics_update
      });
  }
  character2d_t::movement_callback_handle_t character2d_t::add_movement_callback(std::function<void()> fn) {
    return fan::physics::add_physics_step_callback(fn);
  }
  void character2d_t::enable_default_movement(uint8_t movement) {
    movement_state.enabled = true;
    movement_state.type = movement;
    movement_cb_handle = add_movement_callback([this, movement]() {
      if (movement_state.ignore_input) {
        return;
      }
      process_keyboard_movement(movement);
    });
  }
  void character2d_t::setup_default_animations(const fan::graphics::physics::character2d_t::character_config_t& config) {
    auto anims = get_all_animations();
    struct anim_t {
      int fps = 0;
      fan::graphics::animation_nr_t id {};
    } attack, idle, move, hurt;
    for (auto& [name, anim_id] : anims) {
      auto& a = fan::graphics::all_animations[anim_id];
      if (name == "attack0") {
        attack = {a.fps, anim_id};
      }
      else if (name == "idle") {
        idle = {a.fps, anim_id};
      }
      else if (name == "move") {
        move = {a.fps, anim_id};
      }
      else if (name == "hurt") {
        hurt = {a.fps, anim_id};
      }
    }
    if (hurt.fps) {
      anim_controller.add_state({
        .name = "hurt",
        .animation_id = hurt.id,
        .fps = hurt.fps,
        .trigger_type = animation_controller_t::animation_state_t::one_shot,
        .condition = [](character2d_t& c) {
          if (c.attack_state.took_damage && !c.attack_state.is_attacking) {
            c.attack_state.took_damage = false;
            return true;
          }
          return false;
        }
      });
    }
    if (attack.fps) {
      set_animation_loop(attack.id, false);
      anim_controller.add_state({
        .name = "attack0",
        .animation_id = attack.id,
        .fps = attack.fps,
        .trigger_type = animation_controller_t::animation_state_t::one_shot,
        .condition = config.attack_cb ? config.attack_cb :
        [](character2d_t& c) -> bool {
          if (!fan::window::is_input_action_active(fan::actions::light_attack) && !fan::window::is_key_pressed(fan::gamepad_right_bumper)
          #if defined(FAN_GUI)
            || fan::graphics::gui::want_io()
          #endif
          ) {
            return false;
          }
          return c.attack_state.try_attack(&c);
        }
      });
    }
    if (idle.fps) {
      anim_controller.add_state({
        .name = "idle",
        .animation_id = idle.id,
        .fps = idle.fps,
        .condition = [](character2d_t& c) {
          if (c.get_health() <= 0) {
            return false;
          }
          return std::abs(c.get_linear_velocity().x) < 10.f && !c.movement_state.jump_state.on_air_after_jump;
        }
      });
      set_current_animation_id(idle.id);
    }
    if (move.fps) {
      anim_controller.add_state({
        .name = "move",
        .animation_id = move.id,
        .fps = move.fps,
        .velocity_based_fps = true,
        .condition = [](character2d_t& c) {
          if (c.get_health() <= 0) {
            return false;
          }
          if (c.movement_state.is_wall_sliding) {
            return true;
          }
          return std::abs(c.get_linear_velocity().x) >= 10.f/* || c.movement_state.jump_state.on_air_after_jump*/;
        }
      });
    }
    anim_controller.auto_update_animations = true;
  }
  void character2d_t::process_keyboard_movement(uint8_t movement, f32_t friction) {
    movement_state.is_wall_sliding = false;
    fan::vec2 velocity = get_linear_velocity();
    fan::physics::shape_id_t colliding_wall_id;
    wall_jump.normal = fan::physics::check_wall_contact(*this, &colliding_wall_id);
    fan::vec2 input_vector = fan::window::get_input_vector();

    switch (movement) {
    case movement_e::side_view:
    {
      bool on_ground = is_on_ground();
      f32_t air_control_multiplier = on_ground ? 1.0f : 0.8f;
      movement_state.move_to_direction(*this, fan::vec2(input_vector.x, 0) * air_control_multiplier);
      if (wall_jump.normal.x && input_vector.x) {
        if (!movement_state.jump_state.jumping && velocity.y > 0) {
          fan::physics::apply_wall_slide(*this, wall_jump.normal, wall_jump.slide_speed);
          movement_state.is_wall_sliding = true;
        }
      }

      movement_state.perform_jump(*this, fan::window::is_action_down(fan::actions::move_up), &wall_jump.normal, &wall_jump);
      break;
    }
    case movement_e::top_view:
    {
      movement_state.move_to_direction_raw(*this, input_vector);
      break;
    }
    }
  }
  bool character2d_t::is_on_ground() const {
    return fan::physics::is_on_ground(*this, movement_state.jump_state.jumping, (fan::physics::body_id_t*)feet);
  }
  f32_t character2d_t::get_max_health() const {
    return attack_state.max_health;
  }
  f32_t character2d_t::get_health() const {
    return attack_state.health;
  }
  void character2d_t::set_max_health(f32_t v) {
    attack_state.max_health = v;
  }
  void character2d_t::set_health(f32_t v) {
    attack_state.health = v;
  }
  bool character2d_t::is_dead() const {
    return get_health() <= 0.f;
  }
  void character2d_t::reset_health() {
    set_health(get_max_health());
  }
  f32_t character2d_t::get_jump_height() const {
    return movement_state.jump_state.impulse;
  }
  void character2d_t::set_jump_height(f32_t v) {
    movement_state.jump_state.impulse = v;
  }
  void character2d_t::enable_double_jump() {
    movement_state.jump_state.allow_double_jump = true;
  }
  void character2d_t::setup_attack_properties(attack_state_t&& attack_state) {
    this->attack_state = std::move(attack_state);
  }
  void character2d_t::take_hit(character2d_t* source, const fan::vec2& hit_direction, f32_t knockback_multiplier) {
    attack_state.health -= source->attack_state.damage;
    attack_state.health = std::max(attack_state.health, 0.f);
    apply_linear_impulse_center(fan::vec2(hit_direction.x * source->attack_state.knockback_force * knockback_multiplier, -source->attack_state.knockback_force / 5.f));
    attack_state.took_damage = true;
    if (attack_state.stun) {
      attack_state.end_attack();
      anim_controller.cancel_current();
    }
  }
  void character2d_t::update_animations() {
    anim_controller.update(this);
  }
  void character2d_t::cancel_animation() {
    anim_controller.cancel_current();
    attack_state.took_damage = false;
  }

  bool character2d_t::raycast(const character2d_t& target) {
    fan::vec2 from_pos = get_center();
    fan::vec2 to_pos = target.get_center();
    fan::physics::ray_result_t result = fan::physics::raycast(from_pos, to_pos);
    return result.hit && (result.shapeId == target.get_shape_id() || result.shapeId == get_shape_id());
  }
  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  void attack_hitbox_t::setup(const hitbox_config_t& cfg) {
    config = cfg;
    instances.resize(cfg.spawns.size());
    hitbox_spawned.resize(cfg.spawns.size(), false);
  }
  void attack_hitbox_t::update(character2d_t* character) {
    if (!character->attack_state.is_attacking) {
      cleanup(character);
      return;
    }
    for (size_t i = 0; i < config.spawns.size(); ++i) {
      if (!hitbox_spawned[i] && character->animation_on(config.attack_animation, config.spawns[i].frame)) {
        spawn_hitbox(character, i);
      }

      if (hitbox_spawned[i] && !character->animation_on(config.attack_animation, config.spawns[i].frame)) {
        instances[i].hitbox.destroy();
        hitbox_spawned[i] = false;
        instances[i].used = false;
      }
    }
  }
  void attack_hitbox_t::spawn_hitbox(character2d_t* character, int index) {
    if (instances[index].hitbox.is_valid()) {
      instances[index].hitbox.destroy();
    }
    f32_t direction = fan::math::sgn(character->get_tc_size().x);
    instances[index].hitbox = config.spawns[index].create_hitbox(character->get_center(), direction);
    hitbox_spawned[index] = true;
    instances[index].used = false;
  }
  bool attack_hitbox_t::spawned() const {
    return !hitbox_spawned.size();
  }
  bool attack_hitbox_t::check_hit(character2d_t* character, int index, character2d_t* target) {
    if (!character->attack_state.is_attacking) {
      return false;
    }
    if (!hitbox_spawned[index] || !instances[index].hitbox.is_valid()) {
      return false;
    }
    if (!instances[index].hitbox.test_overlap(*target)) {
      return false;
    }
    if (config.track_hit_targets) {
      if (hit_enemies.find(target->NRI) != hit_enemies.end()) {
        return false;
      }
      hit_enemies.insert(target->NRI);
    }
    if (instances[index].used) {
      return false;
    }
    instances[index].used = true;
    return true;
  }
  void attack_hitbox_t::cleanup(character2d_t* character) {
    if (character->attack_state.is_attacking) {
      return;
    }
    for (auto& instance : instances) {
      if (instance.hitbox.is_valid()) {
        instance.hitbox.destroy();
      }
      instance.spawned = false;
      instance.used = false;
    }
    std::fill(hitbox_spawned.begin(), hitbox_spawned.end(), false);
    if (config.track_hit_targets) {
      hit_enemies.clear();
    }
  }
  size_t attack_hitbox_t::hitbox_count() const {
    return instances.size();
  }

  //------------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------------

  std::string bone_to_string(int bone) {
    if (bone >= std::size(bone_names)) {
      return "N/A";
    }
    return bone_names[bone];
  }

  void update_reference_angle(b2WorldId world, fan::physics::joint_id_t& joint_id, f32_t new_reference_angle) {
    b2BodyId bodyIdA = b2Joint_GetBodyA(joint_id);
    b2BodyId bodyIdB = b2Joint_GetBodyB(joint_id);

    b2Vec2 localAnchorA = b2Joint_GetLocalAnchorA(joint_id);
    b2Vec2 localAnchorB = b2Joint_GetLocalAnchorB(joint_id);
    bool enableLimit = b2RevoluteJoint_IsLimitEnabled(joint_id);
    f32_t lowerAngle = b2RevoluteJoint_GetLowerLimit(joint_id);
    f32_t upperAngle = b2RevoluteJoint_GetUpperLimit(joint_id);
    bool enableMotor = b2RevoluteJoint_IsMotorEnabled(joint_id);
    f32_t motorSpeed = b2RevoluteJoint_GetMotorSpeed(joint_id);
    f32_t maxMotorTorque = b2RevoluteJoint_GetMaxMotorTorque(joint_id);
    f32_t hertz = b2RevoluteJoint_GetSpringHertz(joint_id);
    f32_t damping_ratio = b2RevoluteJoint_GetSpringDampingRatio(joint_id);

    b2DestroyJoint(joint_id);

    b2RevoluteJointDef jointDef;
    jointDef.bodyIdA = bodyIdA;
    jointDef.bodyIdB = bodyIdB;
    jointDef.localAnchorA = localAnchorA;
    jointDef.localAnchorB = localAnchorB;
    jointDef.referenceAngle = new_reference_angle;
    jointDef.enableLimit = enableLimit;
    jointDef.lowerAngle = lowerAngle;
    jointDef.upperAngle = upperAngle;
    jointDef.enableMotor = enableMotor;
    jointDef.motorSpeed = motorSpeed;
    jointDef.maxMotorTorque = maxMotorTorque;
    jointDef.enableSpring = hertz > 0.0f;
    jointDef.hertz = hertz;
    jointDef.dampingRatio = damping_ratio;

    joint_id = b2CreateRevoluteJoint(world, &jointDef);
  }

  human_t::human_t(const fan::vec2& position, const f32_t scale, const bone_images_t& images, const fan::color& color) {
    load(position, scale, images, color);
  }

  void human_t::load_bones(const fan::vec2& position, f32_t scale, std::array<fan::graphics::physics::bone_t, fan::graphics::physics::bone_e::bone_count>& bones) {
    for (int i = 0; i < fan::graphics::physics::bone_e::bone_count; ++i) {
      bones[i].joint_id = b2_nullJointId;
      bones[i].friction_scale = 1.0f;
      bones[i].parent_index = -1;
    }

    struct bone_data_t {
      int parent_index;
      fan::vec3 position;
      f32_t size;
      f32_t friction_scale;
      fan::vec2 pivot;
      f32_t lower_angle;
      f32_t upper_angle;
      f32_t reference_angle;
      fan::vec2 center0;
      fan::vec2 center1;
    };

    bone_data_t bone_data[] = {
      { // hip
        .parent_index = -1,
        .position = {0.0f, -0.95f, 55},
        .size = 0.095f,
        .friction_scale = 1.0f,
        .pivot = {0.0f, 0.0f},
        .lower_angle = 0.f,
        .upper_angle = 0.f,
        .reference_angle = 0.f,
        .center0 = {0.f, -0.02f},
        .center1 = {0.f, 0.02f}
      },
        { // torso
          .parent_index = fan::graphics::physics::bone_e::hip,
          .position = {0.0f, -1.2f, 60},
          .size = 0.09f,
          .friction_scale = 0.5f,
          .pivot = {0.0f, -1.0f},
          .lower_angle = -0.25f * fan::math::pi,
          .upper_angle = 0.f,
          .reference_angle = 0.f,
          .center0 = {0.f, -0.135f},
          .center1 = {0.f, 0.135f}
        },
        { // head
          .parent_index = fan::graphics::physics::bone_e::torso,
          .position = {0.0f, -1.475f, 44},
          .size = 0.075f,
          .friction_scale = 0.25f,
          .pivot = {0.0f, -1.4f},
          .lower_angle = -0.3f * fan::math::pi,
          .upper_angle = 0.1f * fan::math::pi,
          .reference_angle = 0.f,
          .center0 = {0.f, -0.038f},
          .center1 = {0.f, 0.039f}
        },
        { // upper left leg
          .parent_index = fan::graphics::physics::bone_e::hip,
          .position = {0.0f, -0.775f, 52},
          .size = 0.06f,
          .friction_scale = 1.0f,
          .pivot = {0.0f, -0.9f},
          .lower_angle = -0.5f * fan::math::pi,
          .upper_angle = 0.5f * fan::math::pi,
          .reference_angle = 0.f,
          .center0 = {0.f, -0.125f},
          .center1 = {0.f, 0.125f}
        },
        { // lower left leg
          .parent_index = fan::graphics::physics::bone_e::upper_left_leg,
          .position = {0.0f, -0.475f, 51},
          .size = 0.045f,
          .friction_scale = 0.5f,
          .pivot = {0.0f, -0.625f},
          .lower_angle = -0.5f * fan::math::pi,
          .upper_angle = 0.5f * fan::math::pi,
          .reference_angle = 0.f,
          .center0 = {0.f, -0.125f},
          .center1 = {0.f, 0.045f}
        },
        { // upper right leg
          .parent_index = fan::graphics::physics::bone_e::hip,
          .position = {0.0f, -0.775f, 54},
          .size = 0.06f,
          .friction_scale = 1.0f,
          .pivot = {0.0f, -0.9f},
          .lower_angle = -0.5f * fan::math::pi,
          .upper_angle = 0.5f * fan::math::pi,
          .reference_angle = 0.f,
          .center0 = {0.f, -0.125f},
          .center1 = {0.f, 0.125f}
        },
        { // lower right leg
          .parent_index = fan::graphics::physics::bone_e::upper_right_leg,
          .position = {0.0f, -0.475f, 53},
          .size = 0.045f,
          .friction_scale = 0.5f,
          .pivot = {0.0f, -0.625f},
          .lower_angle = -0.5f * fan::math::pi,
          .upper_angle = 0.5f * fan::math::pi,
          .reference_angle = 0.f,
          .center0 = {0.f, -0.155f},
          .center1 = {0.f, 0.125f}
        },
        { // upper left arm
          .parent_index = fan::graphics::physics::bone_e::torso,
          .position = {0.0f, -1.225f, 62},
          .size = 0.035f,
          .friction_scale = 0.5f,
          .pivot = {0.0f, -1.35f},
          .lower_angle = -0.5f * fan::math::pi,
          .upper_angle = 0.5f * fan::math::pi,
          .reference_angle = 0.f,
          .center0 = {0.f, -0.125f},
          .center1 = {0.f, 0.125f}
        },
        { // lower left arm
          .parent_index = fan::graphics::physics::bone_e::upper_left_arm,
          .position = {0.0f, -0.975f, 61},
          .size = 0.03f,
          .friction_scale = 0.1f,
          .pivot = {0.0f, -1.1f},
          .lower_angle = -0.5f * fan::math::pi,
          .upper_angle = 0.5f * fan::math::pi,
          .reference_angle = -0.25f * fan::math::pi,
          .center0 = {0.f, -0.125f},
          .center1 = {0.f, 0.125f}
        },
        { // upper right arm
          .parent_index = fan::graphics::physics::bone_e::torso,
          .position = {0.0f, -1.225f, 64},
          .size = 0.035f,
          .friction_scale = 0.5f,
          .pivot = {0.0f, -1.35f},
          .lower_angle = -0.5f * fan::math::pi,
          .upper_angle = 0.5f * fan::math::pi,
          .reference_angle = 0.f,
          .center0 = {0.f, -0.125f},
          .center1 = {0.f, 0.125f}
        },
        { // lower right arm
          .parent_index = fan::graphics::physics::bone_e::upper_right_arm,
          .position = {0.0f, -0.975f, 63},
          .size = 0.03f,
          .friction_scale = 0.1f,
          .pivot = {0.0f, -1.1f},
          .lower_angle = -0.5f * fan::math::pi,
          .upper_angle = 0.5f * fan::math::pi,
          .reference_angle = -0.25f * fan::math::pi,
          .center0 = {0.f, -0.125f},
          .center1 = {0.f, 0.125f}
        }
    };

    for (int i = 0; i < std::size(bone_data); ++i) {
      bones[i].parent_index = bone_data[i].parent_index;
      bones[i].position = fan::vec2(bone_data[i].position) * scale;
      bones[i].position.z = bone_data[i].position.z;
      bones[i].size = bone_data[i].size * scale;
      bones[i].friction_scale = bone_data[i].friction_scale;
      bones[i].pivot = bone_data[i].pivot * scale;
      bones[i].lower_angle = bone_data[i].lower_angle;
      bones[i].upper_angle = bone_data[i].upper_angle;
      bones[i].reference_angle = bone_data[i].reference_angle;
      bones[i].center0 = bone_data[i].center0 * scale;
      bones[i].center1 = bone_data[i].center1 * scale;
    }
  }

  human_t::bone_images_t human_t::load_character_images(const std::string& character_folder_path, const fan::graphics::image_load_properties_t& lp) {
    human_t::bone_images_t character_images;
    character_images[fan::graphics::physics::bone_e::head] = fan::graphics::image_load(character_folder_path + "/head.webp", lp);
    character_images[fan::graphics::physics::bone_e::torso] = fan::graphics::image_load(character_folder_path + "/torso.webp", lp);
    character_images[fan::graphics::physics::bone_e::hip] = fan::graphics::image_load(character_folder_path + "/hip.webp", lp);
    character_images[fan::graphics::physics::bone_e::upper_left_leg] = fan::graphics::image_load(character_folder_path + "/upper_leg.webp", lp);
    character_images[fan::graphics::physics::bone_e::lower_left_leg] = fan::graphics::image_load(character_folder_path + "/lower_leg.webp", lp);
    character_images[fan::graphics::physics::bone_e::upper_right_leg] = character_images[fan::graphics::physics::bone_e::upper_left_leg];
    character_images[fan::graphics::physics::bone_e::lower_right_leg] = character_images[fan::graphics::physics::bone_e::lower_left_leg];
    character_images[fan::graphics::physics::bone_e::upper_left_arm] = fan::graphics::image_load(character_folder_path + "/upper_arm.webp", lp);
    character_images[fan::graphics::physics::bone_e::lower_left_arm] = fan::graphics::image_load(character_folder_path + "/lower_arm.webp", lp);
    character_images[fan::graphics::physics::bone_e::upper_right_arm] = character_images[fan::graphics::physics::bone_e::upper_left_arm];
    character_images[fan::graphics::physics::bone_e::lower_right_arm] = character_images[fan::graphics::physics::bone_e::lower_left_arm];
    return character_images;
  }

  void human_t::animate_walk(f32_t force, f32_t dt) {

    fan::physics::body_id_t torso_id = bones[bone_e::torso].visual;
    b2Vec2 force_ = {force, 0};

    bone_t& blower_left_arm = bones[bone_e::lower_left_arm];
    bone_t& blower_right_arm = bones[bone_e::lower_right_arm];
    bone_t& bupper_left_leg = bones[bone_e::upper_left_leg];
    bone_t& bupper_right_leg = bones[bone_e::upper_right_leg];
    bone_t& blower_left_leg = bones[bone_e::lower_left_leg];
    bone_t& blower_right_leg = bones[bone_e::lower_right_leg];

    f32_t torso_vel_x = torso_id.get_linear_velocity().x;
    f32_t torso_vel_y = torso_id.get_linear_velocity().y;
    int vel_sgn = fan::math::sgn(torso_vel_x);

    int force_sgn = fan::math::sgn(force);
    f32_t swing_speed = torso_vel_x ? (vel_sgn * 0.f + torso_vel_x / 15.f) : 0;

    f32_t ttransform = b2Rot_GetAngle(b2Body_GetRotation(bones[bone_e::torso].visual));
    f32_t lutransform = b2Rot_GetAngle(b2Body_GetRotation(bupper_left_leg.visual));
    f32_t rutransform = b2Rot_GetAngle(b2Body_GetRotation(bupper_right_leg.visual));

    f32_t lltransform = b2Rot_GetAngle(b2Body_GetRotation(blower_left_leg.visual));
    f32_t rltransform = b2Rot_GetAngle(b2Body_GetRotation(blower_right_leg.visual));

    if (std::abs(torso_vel_x) / 130.f > 1.f && torso_vel_x) {
      for (int i = 0; i < bone_e::bone_count; ++i) {
        bones[i].visual.set_tc_size(fan::vec2(vel_sgn, 1));
        if (torso_vel_x < 0) {
          //b2Body_SetTransform(bones[i].visual,  bones[i].visual.get_physics_position() + fan::vec2(bones[bone_e::torso].visual.get_position().x - bones[i].visual.get_position().x, 0) / fan::physics::length_units_per_meter/2, b2Body_GetRotation(bones[i].visual));
          if (bones[i].joint_id.is_valid() == false) {
            continue;
          }
          static int x = 0;
          if (!x) {
            fan::vec2 pivot = fan::vec2(500, 300.f) / fan::physics::length_units_per_meter + bones[i].pivot * scale;
            //     update_position(fan::physics::gphysics()->world_id, bones[i].joint_id, pivot);
            x++;
          }

        }
      }
    }

    if (torso_vel_x) {
      if (!force) {
        //   torsoId.apply_force_center(fan::vec2(-torso_vel_x, 0));
      }

      f32_t quarter_pi = -0.25f * fan::math::pi;
      //quarter_pi *= 3; // why this is required?
      //quarter_pi += fan::math::pi;
      if (std::abs(torso_vel_x) / 130.f > 1.f && torso_vel_x) {
        //   update_reference_angle(fan::physics::gphysics()->world_id, blower_left_arm.joint_id, vel_sgn == 1 ? quarter_pi : -quarter_pi);
        //    update_reference_angle(fan::physics::gphysics()->world_id, blower_right_arm.joint_id, vel_sgn == 1 ? quarter_pi : -quarter_pi);
        look_direction = vel_sgn;
      }

      if (force || std::abs(torso_vel_x / 10.f) > 1.f) {
        f32_t leg_turn = 0.4;

        if (rutransform < (look_direction == 1 ? -leg_turn / 2 : -leg_turn)) {
          direction = 0;
        }
        if (rutransform > (look_direction == -1 ? leg_turn / 2 : leg_turn)) {
          direction = 1;
        }

        f32_t rotate_speed = 1.3 * std::abs(torso_vel_x) / 200.f;

        if (direction == 1) {
          bupper_right_leg.joint_id.revolute_joint_set_motor_speed(-rotate_speed);
          bupper_left_leg.joint_id.revolute_joint_set_motor_speed(rotate_speed);

        }
        else {
          bupper_right_leg.joint_id.revolute_joint_set_motor_speed(rotate_speed);
          bupper_left_leg.joint_id.revolute_joint_set_motor_speed(-rotate_speed);

        }
        blower_right_leg.joint_id.revolute_joint_set_motor_speed(look_direction * leg_turn / 4 - rltransform);
        blower_left_leg.joint_id.revolute_joint_set_motor_speed(look_direction * leg_turn / 4 - lltransform);
      }
      else {
        bupper_left_leg.joint_id.revolute_joint_set_motor_speed((ttransform - lutransform) * 5);
        bupper_right_leg.joint_id.revolute_joint_set_motor_speed((ttransform - rutransform) * 5);

        blower_left_leg.joint_id.revolute_joint_set_motor_speed((ttransform - lltransform) * 5);
        blower_right_leg.joint_id.revolute_joint_set_motor_speed((ttransform - rltransform) * 5);
      }
    }

  }

  void human_t::load_preset(const fan::vec2& position, const f32_t scale, const bone_images_t& images, std::array<bone_t, bone_e::bone_count>& bones, const fan::color& color) {
    this->scale = scale;
    int groupIndex = 1;
    f32_t frictionTorque = 0.03f;
    f32_t hertz = 5.0f;
    f32_t dampingRatio = 0.5f;
    b2WorldId worldId = fan::physics::gphysics()->world_id;

    b2Filter filter = b2DefaultFilter();

    filter.groupIndex = -groupIndex;
    filter.categoryBits = 2;
    filter.maskBits = (1 | 2);

    f32_t maxTorque = frictionTorque * scale * 1000;
    bool enableMotor = true;
    bool enableLimit = true;

    for (int i = 0; i < std::size(bones); ++i) {
      auto& bone = bones[i];
      bone.visual = capsule_sprite_t {{
          .position = fan::vec3(position + (fan::vec2(bone.position) * fan::physics::length_units_per_meter + bone.offset) * scale, bone.position.z),
          /*
          bone.center0 * fan::physics::length_units_per_meter * bone.scale * scale
          bone.center1 * fan::physics::length_units_per_meter * bone.scale * scale
          */
        .center0 = fan::vec2(0),
        .center1 = fan::vec2(0),
        .size = fan::physics::length_units_per_meter * bone.size.y * bone.scale * scale,
        .color = color,
        .image = images[i],
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties {
          .friction = 0.6,
          .fixed_rotation = i == bone_e::hip || i == bone_e::torso,
          .linear_damping = 0.0f,
          .filter = filter
      },
        }};

      if (bone.parent_index == -1) {
        continue;
      }
      fan::vec2 physics_position = bone.visual.get_physics_position();
      fan::vec2 pivot = (position / fan::physics::length_units_per_meter) + bone.pivot * scale;
      //  hitbox_visualize[&bones[i]] = fan::graphics::rectangle_t{{
      //  .position = fan::vec3(position + bone.pivot * scale * fan::physics::length_units_per_meter, 60001),
      //  .size=5,
      //  .color = fan::color(0, 0, 1, 0.2),
      //  .outline_color=fan::color(0, 0, 1, 0.2)*2,
      //  .blending=true
      //}};
      b2RevoluteJointDef joint_def = b2DefaultRevoluteJointDef();
      joint_def.bodyIdA = bones[bone.parent_index].visual;
      joint_def.bodyIdB = bone.visual;
      joint_def.localAnchorA = b2Body_GetLocalPoint(joint_def.bodyIdA, pivot);
      joint_def.localAnchorB = b2Body_GetLocalPoint(joint_def.bodyIdB, pivot);
      joint_def.referenceAngle = bone.reference_angle;
      joint_def.enableLimit = enableLimit;
      joint_def.lowerAngle = bone.lower_angle;
      joint_def.upperAngle = bone.upper_angle;
      joint_def.enableMotor = enableMotor;
      joint_def.maxMotorTorque = bone.friction_scale * maxTorque;
      joint_def.enableSpring = hertz > 0.0f;
      joint_def.hertz = hertz;
      joint_def.dampingRatio = dampingRatio;

      bone.joint_id = b2CreateRevoluteJoint(worldId, &joint_def);
    }
    is_spawned = true;
  }

  void human_t::load(const fan::vec2& position, const f32_t scale, const bone_images_t& images, const fan::color& color) {
    load_bones(position, scale, bones);
    load_preset(position, scale, images, bones, color);
  }

  void human_t::animate_jump(f32_t jump_impulse, f32_t dt, bool is_jumping) {
    bone_t& bupper_left_leg = bones[bone_e::upper_left_leg];
    bone_t& bupper_right_leg = bones[bone_e::upper_right_leg];
    bone_t& blower_left_leg = bones[bone_e::lower_left_leg];
    bone_t& blower_right_leg = bones[bone_e::lower_right_leg];
    if (is_jumping) {
      go_up = 0;
    }
    if (go_up == 1 && !jump_animation_timer.finished()) {
      bones[bone_e::torso].visual.apply_linear_impulse_center(fan::vec2(0, jump_impulse / 4));
    }
    else if (go_up == 1 && jump_animation_timer.finished()) {
      bones[bone_e::torso].visual.apply_linear_impulse_center(fan::vec2(0, -jump_impulse));
      go_up = 0;
    }
    if (go_up == 0 && is_jumping) {
      //f32_t torso_vel_x = b2Body_GetLinearVelocity(bones[bone_e::torso].visual).x;
      //b2RevoluteJoint_SetSpringHertz(blower_left_leg.joint_id, 1);
      //b2RevoluteJoint_SetSpringHertz(blower_right_leg.joint_id, 1);

      //b2RevoluteJoint_SetMotorSpeed(blower_left_leg.joint_id, fan::math::sgn(torso_vel_x) * 10.2 );
      //b2RevoluteJoint_SetMotorSpeed(blower_left_leg.joint_id, fan::math::sgn(torso_vel_x) * 10.2 );

      //b2RevoluteJoint_SetMotorSpeed(bupper_left_leg.joint_id,  fan::math::sgn(torso_vel_x) *  -10.2 );
      //b2RevoluteJoint_SetMotorSpeed(bupper_right_leg.joint_id, fan::math::sgn(torso_vel_x) *  -10.2);

      go_up = 1;
      jump_animation_timer.start(0.09e9);
    }
  }

  void human_t::erase() {
    if (!(is_spawned == true)) {
      fan::throw_error_impl();
    }

    for (int i = 0; i < bone_e::bone_count; ++i) {
      if (B2_IS_NULL(bones[i].joint_id)) {
        continue;
      }

      if (b2Joint_IsValid(bones[i].joint_id)) {
        b2DestroyJoint(bones[i].joint_id);
        bones[i].joint_id = b2_nullJointId;
      }
    }
  }

  mouse_joint_t::operator fan::physics::body_id_t& () { return dummy_body; }

  mouse_joint_t::operator const fan::physics::body_id_t& () const { return dummy_body; }

  bool mouse_joint_t::QueryCallback(b2ShapeId shapeId, void* context) {
    QueryContext* queryContext = static_cast<QueryContext*>(context);

    b2BodyId bodyId = b2Shape_GetBody(shapeId);
    b2BodyType bodyType = b2Body_GetType(bodyId);
    if (bodyType != b2_dynamicBody) {
      // continue query
      return true;
    }

    bool overlap = b2Shape_TestPoint(shapeId, queryContext->point);
    if (overlap) {
      // found shape
      queryContext->bodyId = bodyId;
      return false;
    }

    return true;
  }

  mouse_joint_t::mouse_joint_t() {

    auto default_body = b2DefaultBodyDef();
    dummy_body.set_body(b2CreateBody(fan::physics::gphysics()->world_id, &default_body));
    nr = fan::graphics::ctx().update_callback->NewNodeLast();
    // not copy safe
    (*fan::graphics::ctx().update_callback)[nr] = [this](void* ptr) {
      if (fan::window::is_mouse_down()) {
        fan::vec2 p = fan::window::get_mouse_position() / fan::physics::length_units_per_meter;
        if (!B2_IS_NON_NULL(mouse_joint)) {
          b2AABB box;
          b2Vec2 d = {0.001f, 0.001f};
          box.lowerBound = b2Sub(p, d);
          box.upperBound = b2Add(p, d);

          QueryContext queryContext = {p, b2_nullBodyId};
          b2World_OverlapAABB(fan::physics::gphysics()->world_id, box, b2DefaultQueryFilter(), QueryCallback, &queryContext);
          if (B2_IS_NON_NULL(queryContext.bodyId)) {

            b2MouseJointDef mouseDef = b2DefaultMouseJointDef();
            mouseDef.bodyIdA = dummy_body;
            mouseDef.bodyIdB = queryContext.bodyId;
            mouseDef.target = p;
            mouseDef.hertz = 5.0f;
            mouseDef.dampingRatio = 0.7f;
            mouseDef.maxForce = 1000.0f * b2Body_GetMass(queryContext.bodyId);
            mouse_joint = b2CreateMouseJoint(fan::physics::gphysics()->world_id, &mouseDef);
            b2Body_SetAwake(queryContext.bodyId, true);
          }
        }
        else {
          b2MouseJoint_SetTarget(mouse_joint, p);
          b2BodyId bodyIdB = b2Joint_GetBodyB(mouse_joint);
          b2Body_SetAwake(bodyIdB, true);
        }
      }
      else if (fan::window::is_mouse_released()) {
        if (B2_IS_NON_NULL(mouse_joint)) {
          b2DestroyJoint(mouse_joint);
          mouse_joint = b2_nullJointId;
        }
      }
    };
  }

  mouse_joint_t::~mouse_joint_t() {
    if (dummy_body.is_valid()) {
      dummy_body.destroy();
    }
    if (nr.iic() == false) {
      fan::graphics::ctx().update_callback->unlrec(nr);
      nr.sic();
    }
  }



  character2d_t character_circle(const fan::vec3& position, f32_t radius, const fan::physics::shape_properties_t& shape_properties) {
    character2d_t character {circle_t {{
        .position = position,
        .radius = radius,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties = shape_properties,
      }}};
    return character;
  }

  character2d_t character_circle(const circle_t::properties_t& visual_properties, const fan::physics::shape_properties_t& physics_properties) {
    circle_t::properties_t p = visual_properties;
    p.shape_properties = physics_properties;
    p.body_type = fan::physics::body_type_e::dynamic_body;
    return circle_t(p);
  }

  character2d_t character_circle_sprite(const fan::vec3& position, f32_t radius, const fan::graphics::image_t& image, const fan::physics::shape_properties_t& shape_properties) {
    character2d_t character {circle_sprite_t {{
        .position = position,
        .radius = radius,
        .image = image,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties = shape_properties,
      }}};
    return character;
  }

  character2d_t character_circle_sprite(const circle_sprite_t::properties_t& visual_properties, const fan::physics::shape_properties_t& physics_properties) {
    circle_sprite_t::properties_t p = visual_properties;
    p.shape_properties = physics_properties;
    p.body_type = fan::physics::body_type_e::dynamic_body;
    return circle_sprite_t(p);
  }

  character2d_t character_capsule(const fan::vec3& position, const fan::vec2& center0, const fan::vec2& center1, f32_t radius, const fan::physics::shape_properties_t& shape_properties) {
    character2d_t character {capsule_t {{
        .position = position,
        .center0 = center0,
        .center1 = center1,
        .radius = radius,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties = shape_properties,
      }}};
    return character;
  }

  // creates physics body for visual shape
  fan::physics::entity_t character_capsule(const fan::graphics::shape_t& shape, f32_t shape_size_multiplier, const fan::physics::shape_properties_t& physics_properties, uint8_t body_type) {
    f32_t half_height = shape.get_size().y * shape_size_multiplier;
    return fan::physics::gphysics()->create_capsule(
      shape.get_position(),
      shape.get_angle().z,
      b2Capsule {
        .center1 = fan::vec2(0, -half_height),
        .center2 = fan::vec2(0, half_height),
        .radius = half_height,
      }, body_type, physics_properties
      );
  }

  character2d_t character_capsule(const capsule_t::properties_t& visual_properties, const fan::physics::shape_properties_t& physics_properties) {
    capsule_t::properties_t p = visual_properties;
    p.shape_properties = physics_properties;
    p.body_type = fan::physics::body_type_e::dynamic_body;
    return capsule_t(p);
  }

  character2d_t character_capsule_sprite(const fan::vec3& position, const fan::vec2& center0, const fan::vec2& center1, const fan::vec2& size, const fan::graphics::image_t& image, const fan::physics::shape_properties_t& shape_properties) {
    character2d_t character {capsule_sprite_t {{
        .position = position,
        .center0 = center0,
        .center1 = center1,
        .size = size,
        .image = image,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties = shape_properties,
      }}};
    return character;
  }

  character2d_t character_capsule_sprite(const capsule_sprite_t::properties_t& visual_properties, const fan::physics::shape_properties_t& physics_properties) {
    capsule_sprite_t::properties_t p = visual_properties;
    p.shape_properties = physics_properties;
    p.body_type = fan::physics::body_type_e::dynamic_body;
    return capsule_sprite_t(p);
  }

  character2d_t character_rectangle(const fan::vec3& position, const fan::vec2& size, const fan::physics::shape_properties_t& shape_properties) {
    character2d_t character {rectangle_t {{
        .position = position,
        .size = size,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties = shape_properties,
      }}};
    return character;
  }

  character2d_t character_rectangle(const rectangle_t::properties_t& visual_properties, const fan::physics::shape_properties_t& physics_properties) {
    rectangle_t::properties_t p = visual_properties;
    p.shape_properties = physics_properties;
    p.body_type = fan::physics::body_type_e::dynamic_body;
    return rectangle_t(p);
  }

  character2d_t character_sprite(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, const fan::physics::shape_properties_t& shape_properties) {
    character2d_t character {sprite_t {{
        .position = position,
        .size = size,
        .image = image,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties = shape_properties,
      }}};
    return character;
  }

  character2d_t character_sprite(const sprite_t::properties_t& visual_properties, const fan::physics::shape_properties_t& physics_properties) {
    sprite_t::properties_t p = visual_properties;
    p.shape_properties = physics_properties;
    p.body_type = fan::physics::body_type_e::dynamic_body;
    return sprite_t(p);
  }

  character2d_t character_polygon(const fan::vec3& position, const std::vector<fan::graphics::vertex_t>& vertices, f32_t radius, const fan::physics::shape_properties_t& shape_properties) {
    character2d_t character {polygon_t {{
        .position = position,
        .radius = radius,
        .vertices = vertices,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties = shape_properties,
      }}};
    return character;
  }

  character2d_t character_polygon(const polygon_t::properties_t& visual_properties, const fan::physics::shape_properties_t& physics_properties) {
    polygon_t::properties_t p = visual_properties;
    p.shape_properties = physics_properties;
    p.body_type = fan::physics::body_type_e::dynamic_body;
    return polygon_t(p);
  }
}

namespace fan::physics {
  bool is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) {
    return fan::physics::gphysics()->is_on_sensor(test_id, sensor_id);
  }

  fan::physics::ray_result_t raycast(const fan::vec2& src, const fan::vec2& dst) {
    return fan::physics::gphysics()->raycast(src, dst);
  }
}

namespace fan::graphics {
  void camera_look_at(fan::graphics::camera_nr_t nr, const fan::graphics::physics::character2d_t& target, f32_t move_speed) {
    camera_set_target(nr, target.get_position(), move_speed);
  }
  void camera_look_at(const fan::graphics::physics::character2d_t& target, f32_t move_speed) {
    camera_look_at(fan::graphics::get_orthographic_render_view().camera, target, move_speed);
  }
}


// dynamic object helpers

namespace fan::graphics::physics {
  void elevator_t::init(const fan::graphics::sprite_t& sprite, const fan::vec2& start_pos, const fan::vec2& end_pos, f32_t dur) {
    visual = sprite;
    visual.set_dynamic();
    start_position = start_pos;
    end_position = end_pos;
    duration = dur;
    t = 0.0f;
    is_active = false;
    going_up = true;
    walls_created = false;
    waiting_for_player_exit = false;
    create_trigger_sensor();

    step_cb = fan::physics::add_physics_step_callback([this] {
      physics_step();
    });
  }

  void elevator_t::create_trigger_sensor() {
    fan::vec2 cage_size = visual.get_size();
    trigger_sensor = fan::physics::gphysics()->create_rectangle(
      start_position,
      cage_size / 15.f,
      0.0f,
      fan::physics::body_type_e::kinematic_body,
      {.is_sensor = true}
    );
  }

  void elevator_t::create_elevator_box() {
    if (walls_created) {
      return;
    }
    walls_created = true;

    fan::vec2 cage_size = visual.get_size();
    fan::vec2 p = trigger_sensor.get_position();

    auto rects = fan::physics::create_stroked_rectangle(
      p,
      cage_size,
      3.f,
      fan::physics::body_type_e::kinematic_body,
      {{
        {.friction = 1.f, .fixed_rotation = true,},
        {.friction = 1.f, .fixed_rotation = true,},
        {.friction = 0.f, .fixed_rotation = true,},
        {.friction = 0.f, .fixed_rotation = true,}
      }}
    );

    for (uint32_t i = 0; i < 4; i++) {
      if (walls[i]) {
        walls[i].destroy();
      }
      walls[i] = rects[i];
    }
  }

  void elevator_t::start() {
    if (is_active) {
      return;
    }
    is_active = true;
    t = 0.0f;
    walls_created = false;
    create_elevator_box();
  }

  void elevator_t::physics_step() {
    if (!is_active) {
      return;
    }

    f32_t dt = fan::physics::default_physics_timestep;
    t += dt / duration;
    if (t >= 1.0f) {
      t = 1.0f;
    }

    fan::vec2 from = going_up ? start_position : end_position;
    fan::vec2 to = going_up ? end_position : start_position;

    f32_t u = fan::apply_ease(fan::ease_e::sine, t);
    fan::vec2 target = from + (to - from) * u;

    fan::vec2 current = trigger_sensor.get_position();
    fan::vec2 vel = (target - current) / dt;

    if (t < 1.0f) {
      trigger_sensor.set_linear_velocity(vel);
      for (auto& w : walls) {
        if (w) {
          w.set_linear_velocity(vel);
        }
      }
      return;
    }

    trigger_sensor.set_linear_velocity(fan::vec2(0));
    for (auto& w : walls) {
      if (w) {
        w.set_linear_velocity(fan::vec2(0));
      }
    }

    walls[2].destroy();
    walls[3].destroy();

    is_active = false;
    waiting_for_player_exit = true;

    fan::vec2 final_pos = trigger_sensor.get_position();

    if (going_up) {
      end_position = final_pos;
    }
    else {
      start_position = final_pos;
    }

    going_up = !going_up;

    if (on_end_cb) {
      on_end_cb();
    }
  }

  void elevator_t::sync_visual() {
    if (trigger_sensor) {
      visual.set_position(trigger_sensor.get_position());
    }
  }

  void elevator_t::update(const fan::physics::entity_t& sensor_triggerer) {
    sync_visual();

    bool inside = fan::physics::is_on_sensor(sensor_triggerer, trigger_sensor);

    if (!is_active) {
      if (waiting_for_player_exit) {
        if (!inside) {
          waiting_for_player_exit = false;
        }
        return;
      }

      if (inside) {
        on_start_cb();
        start();
      }
    }
    else {
      if (t >= 1.0f && !inside) {
        is_active = false;
      }
    }
  }

  void elevator_t::destroy() {
    if (step_cb.iic() == false) {
      //fan::physics::remove_physics_step_callback(step_cb);
      step_cb.~raii_nr_t();
      step_cb.sic();
    }

    if (trigger_sensor) {
      trigger_sensor.destroy();
    }

    for (auto& w : walls) {
      if (w) {
        w.destroy();
      }
    }

    walls_created = false;
    is_active = false;
  }
}

#endif

#endif