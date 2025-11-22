module;

#if defined(fan_physics)
  #include <fan/utility.h>
  #include <box2d/box2d.h>
#endif

#include <vector>
#include <utility>
#include <functional>

module fan.graphics.physics_shapes;

#if defined(fan_physics)

import fan.types;

// higher the draw depth, less debug draws will be if maximum depth is 2^16
// so make sure your game objects do not pass this depth
static constexpr uint16_t draw_depth = 0xF000;

int z_depth = 0;

std::vector<fan::graphics::line_t> debug_draw_polygon;
std::vector<fan::graphics::polygon_t> debug_draw_solid_polygon;
std::vector<fan::graphics::circle_t> debug_draw_circle;
std::vector<fan::graphics::line_t> debug_draw_line;
std::vector<fan::graphics::capsule_t> debug_draw_capsule;

/// Draw a closed polygon provided in CCW order.
void DrawPolygon(const fan::vec2* vertices, int vertexCount, b2HexColor color, void* context) {
  if ( z_depth == 2 ) {
    z_depth = 0;
  }
  for ( int i = 0; i < vertexCount; i++ ) {
    int next_i = (i + 1) % vertexCount;

    //debug_draw_polygon.emplace_back(fan::graphics::line_t{ {
    //  .src = fan::vec3(fan::physics::physics_to_render(vertices[i]), draw_depth + z_depth),
    //  .dst = fan::physics::physics_to_render(vertices[next_i]),
    //  .color = fan::color::from_rgb(color)
    //} });
  }

  ++z_depth;
}

/// Draw a solid closed polygon provided in CCW order.
void DrawSolidPolygon(b2Transform transform, const b2Vec2* vertices, int vertexCount, f32_t radius, b2HexColor color, void* context) {
  std::vector<fan::graphics::vertex_t> vs(vertexCount);
  for ( auto [i, v] : fan::enumerate(vs) ) {
    v.position = fan::physics::physics_to_render(vertices[i]);
    v.color = fan::color::from_rgb(color).set_alpha(0.5);
  }
  debug_draw_solid_polygon.emplace_back(fan::graphics::polygon_t {{
      .position = fan::vec3(fan::physics::physics_to_render(transform.p), draw_depth + z_depth),
      .vertices = vs,
      .draw_mode = fan::graphics::primitive_topology_t::triangle_fan,
      //.angle = std::acos(transform.q.c)
    }});
  ++z_depth;
}

/// Draw a circle.
void DrawCircle(b2Vec2 center, f32_t radius, b2HexColor color, void* context) {
  debug_draw_circle.emplace_back(fan::graphics::circle_t {{
      .position = fan::vec3(fan::physics::physics_to_render(center), draw_depth + z_depth),
      .radius = (f32_t)fan::physics::physics_to_render(radius).x,
      .color = fan::color::from_rgb(color).set_alpha(0.5),
    }});
  ++z_depth;
}

/// Draw a solid circle.
void DrawSolidCircle(b2Transform transform, f32_t radius, b2HexColor color, void* context) {
  debug_draw_circle.emplace_back(fan::graphics::circle_t {{
      .position = fan::vec3(fan::physics::physics_to_render(transform.p), draw_depth + z_depth),
      .radius = (f32_t)fan::physics::physics_to_render(radius).x,
      .color = fan::color::from_rgb(color).set_alpha(0.5),
    }});
  ++z_depth;
}

/// Draw a capsule.
void DrawCapsule(b2Vec2 p1, b2Vec2 p2, f32_t radius, b2HexColor color, void* context) {
  printf("DrawCapsule\n");
}

/// Draw a solid capsule.
void DrawSolidCapsule(b2Vec2 p1, b2Vec2 p2, f32_t radius, b2HexColor color, void* context) {
  debug_draw_capsule.emplace_back(fan::graphics::capsule_t {{
      .position = fan::vec3(0, 0, draw_depth + z_depth),
      .center0 = fan::physics::physics_to_render(p1),
      .center1 = fan::physics::physics_to_render(p2),
      .radius = (f32_t)fan::physics::physics_to_render(radius).x,
      .color = fan::color::from_rgb(color).set_alpha(0.5),
    }});
  ++z_depth;
}


/// Draw a line segment.
void DrawSegment(b2Vec2 p1, b2Vec2 p2, b2HexColor color, void* context) {
  debug_draw_line.emplace_back(fan::graphics::line_t {{
      .src = fan::vec3(fan::physics::physics_to_render(p1), draw_depth + z_depth),
      .dst = fan::vec3(fan::physics::physics_to_render(p2), draw_depth + z_depth),
      .color = fan::color::from_rgb(color)
    }});
  ++z_depth;
}

/// Draw a transform. Choose your own length scale.
void DrawTransform(b2Transform transform, void* context) {

}

/// Draw a point.
void DrawPoint(b2Vec2 p, f32_t size, b2HexColor color, void* context) {
  //vs.back() = vs.front();
  debug_draw_circle.emplace_back(fan::graphics::circle_t {{
      .position = fan::vec3(fan::physics::physics_to_render(p), draw_depth + z_depth),
      .radius = size / 2.f,
      .color = fan::color::from_rgb(color).set_alpha(0.5)
    }});
  ++z_depth;
}

/// Draw a string.
void DrawString(b2Vec2 p, const char* s, b2HexColor color, void* context) {
#if defined(fan_gui)
  fan::vec2 pos = fan::physics::physics_to_render(p) - fan::graphics::camera_get_position(fan::graphics::get_orthographic_render_view().camera);
  pos *= fan::graphics::camera_get_zoom(fan::graphics::get_orthographic_render_view().camera, fan::graphics::get_orthographic_render_view().viewport) * 0.5f;
  pos += fan::graphics::get_window().get_size() / 2.f;

  fan::graphics::gui::text_outlined_at(s, pos, fan::color::from_rgb(color));
#endif
}

b2DebugDraw initialize_debug(bool enabled) {
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
    if ( !init_ ) {
      return;
    }
    init_ = false;
    box2d_debug_draw = [] {
      fan::physics::gphysics->debug_draw_cb = []() {
        z_depth = 0;
        debug_draw_polygon.clear();
        debug_draw_solid_polygon.clear();
        debug_draw_circle.clear();
        debug_draw_line.clear();
        debug_draw_capsule.clear();
        b2World_Draw(fan::physics::gphysics->world_id, &box2d_debug_draw);
      };
      return initialize_debug(false);
    }();
  }

  void debug_draw(bool enabled) {
    init();
    fan::graphics::physics::box2d_debug_draw = initialize_debug(enabled);
  }

  void shape_physics_update(const fan::physics::physics_update_data_t& data) {
    if ( !b2Body_IsValid(*(b2BodyId*)&data.body_id) ) {
      //   fan::print("invalid body data (corruption)");
      return;
    }
    if ( b2Body_GetType(*(b2BodyId*)&data.body_id) == b2_staticBody ) {
      return;
    }

    fan::vec2 p = b2Body_GetWorldPoint(*(b2BodyId*)&data.body_id, fan::vec2(0));
    b2Rot rotation = b2Body_GetRotation(*(b2BodyId*)&data.body_id);
    f32_t radians = b2Rot_GetAngle(rotation);

    fan::graphics::shape_t& shape = *(fan::graphics::shape_t*)&data.shape_id;
    shape.set_position(fan::vec2((p)*fan::physics::length_units_per_meter + data.draw_offset));
    shape.set_angle(fan::vec3(0, 0, radians));
    b2ShapeId id[1];
    if ( b2Body_GetShapes(*(b2BodyId*)&data.body_id, id, 1) ) {
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
    if ( physics_update_nr.iic() == false ) {
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
    if ( physics_update_nr.iic() == false ) {
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
    if ( mass_data.mass < 0.f ) {
      md_copy.mass = md.mass;
    }
    if ( mass_data.center_of_mass.x == 0 && mass_data.center_of_mass.y == 0 ) {
      md_copy.center_of_mass = md.center;
    }
    if ( mass_data.rotational_inertia < 0.f ) {
      md_copy.rotational_inertia = md.rotationalInertia;
    }
    b2Body_SetMassData(*dynamic_cast<b2BodyId*>(this), md_copy);
  }

  base_shape_t::base_shape_t(const base_shape_t& r) : fan::graphics::shape_t(r), fan::physics::entity_t(r) {
    //if (this != )
    fan::physics::body_id_t new_body_id = fan::physics::deep_copy_body(fan::physics::gphysics->world_id, *dynamic_cast<const fan::physics::body_id_t*>(&r));
    if ( !B2_ID_EQUALS(r, (*this)) ) {
      destroy();
    }
    set_body(new_body_id);
    b2Body_GetWorldPoint(*dynamic_cast<b2BodyId*>(this), fan::vec2(0));
    if ( physics_update_nr.iic() == false ) {
      fan::physics::remove_physics_update(physics_update_nr);
      physics_update_nr.sic();
    }
    if ( !fan::physics::entity_t::is_valid() ) {
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
    if ( !B2_ID_EQUALS(r, (*this)) ) {
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
    if ( physics_update_nr.iic() == false ) {
      fan::physics::remove_physics_update(physics_update_nr);
      physics_update_nr.sic();
    }
    if ( this != &r ) {
      fan::graphics::shape_t::operator=(r);

      fan::physics::body_id_t new_body_id = fan::physics::deep_copy_body(fan::physics::gphysics->world_id, *dynamic_cast<const fan::physics::body_id_t*>(&r));
      if ( !B2_ID_EQUALS(r, (*this)) ) {
        destroy();
      }
      set_body(new_body_id);
      if ( !fan::physics::entity_t::is_valid() ) {
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
    if ( !B2_ID_EQUALS(r, (*this)) ) {
      destroy();
    }
    if ( physics_update_nr.iic() == false ) {
      fan::physics::remove_physics_update(physics_update_nr);
      physics_update_nr.sic();
    }
    if ( this != &r ) {
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
    if ( physics_update_nr.iic() == false ) {
      fan::physics::remove_physics_update(physics_update_nr);
    }
    physics_update_nr.sic();
  }

  mass_data_t base_shape_t::get_mass_data() const {
    b2MassData md = b2Body_GetMassData(*this);
    mass_data_t mass_data;
    mass_data.mass = md.mass * (fan::physics::length_units_per_meter * fan::physics::length_units_per_meter);
    mass_data.center_of_mass = md.center * fan::physics::length_units_per_meter;
    mass_data.rotational_inertia = md.rotationalInertia * (fan::physics::length_units_per_meter * fan::physics::length_units_per_meter);
    return mass_data;
  }

  f32_t base_shape_t::get_mass() const {
    return get_mass_data().mass;
  }

  void base_shape_t::set_draw_offset(fan::vec2 new_draw_offset) {
    draw_offset = new_draw_offset;
    (*fan::physics::gphysics->physics_updates)[physics_update_nr].draw_offset = new_draw_offset;
  }

  fan::vec3 base_shape_t::get_position() const {
    // The visual position might have not been updated (maybe) so use physics position but visual Z position
    //return fan::vec3(fan::physics::entity_t::get_position(), fan::graphics::shape_t::get_position().z);
    return fan::graphics::shape_t::get_position();
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
    fan::graphics::shape_t(fan::graphics::rectangle_t{ p }),
    fan::physics::entity_t(fan::physics::gphysics->create_box(p.position, p.size, p.angle.z, p.body_type, p.shape_properties)),
    p.mass_data
  ) {
  }
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
    fan::graphics::shape_t(fan::graphics::sprite_t{ p }),
    fan::physics::entity_t(fan::physics::gphysics->create_box(p.position, p.size, p.angle.z, p.body_type, p.shape_properties)),
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
    fan::graphics::shape_t(fan::graphics::circle_t{ p }),
    fan::physics::entity_t(fan::physics::gphysics->create_circle(p.position, p.radius, p.angle.z, p.body_type, p.shape_properties)),
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
    fan::graphics::shape_t(fan::graphics::sprite_t{ p }),
    fan::physics::entity_t(fan::physics::gphysics->create_circle(p.position, p.radius, p.angle.z, p.body_type, p.shape_properties)),
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
    fan::graphics::shape_t(fan::graphics::capsule_t{ p }),
    fan::physics::entity_t(fan::physics::gphysics->create_capsule(p.position, p.angle.z, 
      b2Capsule{ .center1 = p.center0, .center2 = p.center1, .radius = p.radius }, p.body_type, p.shape_properties)),
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
    fan::graphics::shape_t(fan::graphics::sprite_t{ p }),
    fan::physics::entity_t(fan::physics::gphysics->create_capsule(p.position, p.angle.z, b2Capsule{ 
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
    fan::graphics::shape_t(fan::graphics::polygon_t{ p }),
    fan::physics::entity_t(
      [&] {
    std::vector<fan::vec2> points(p.vertices.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
      points[i] = p.vertices[i].position;
    }
    return fan::physics::gphysics->create_polygon(
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
    fan::graphics::shape_t(fan::graphics::polygon_t{ p }),
    fan::physics::entity_t(
      [&] {
    std::vector<fan::vec2> points(p.vertices.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
      points[i] = p.vertices[i].position;
    }
    return fan::physics::gphysics->create_segment(
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
    std::array<fan::physics::shape_properties_t, 4> shape_properties
  ) {
    std::array<fan::graphics::physics::rectangle_t, 4> walls;
    const fan::color wall_outline = wall_color * 2;
    // top
    walls[0] = fan::graphics::physics::rectangle_t{ {
        .position = fan::vec2(center_position.x, center_position.y - half_size.y),
        .size = fan::vec2(half_size.x * 2, thickness),
        .color = wall_color,
        .outline_color = wall_outline,
        .shape_properties = shape_properties[0]
      } };
    // bottom
    walls[1] = fan::graphics::physics::rectangle_t{ {
        .position = fan::vec2(center_position.x, center_position.y + half_size.y),
        .size = fan::vec2(half_size.x * 2, thickness),
        .color = wall_color,
        .outline_color = wall_color,
        .shape_properties = shape_properties[1]
      } };
    // left
    walls[2] = fan::graphics::physics::rectangle_t{ {
        .position = fan::vec2(center_position.x - half_size.x, center_position.y),
        .size = fan::vec2(thickness, half_size.y * 2),
        .color = wall_color,
        .outline_color = wall_outline,
        .shape_properties = shape_properties[2]
      } };
    // right
    walls[3] = fan::graphics::physics::rectangle_t{ {
        .position = fan::vec2(center_position.x + half_size.x, center_position.y),
        .size = fan::vec2(thickness, half_size.y * 2),
        .color = wall_color,
        .outline_color = wall_outline,
        .shape_properties = shape_properties[3]
      } };
    return walls;
  }

  std::array<rectangle_t, 4> create_walls(
    const fan::vec2& bounds,
    f32_t thickness,
    const fan::color& wall_color,
    std::array<fan::physics::shape_properties_t, 4> shape_properties
  ) {
    return create_stroked_rectangle(
      bounds,
      bounds,
      thickness,
      wall_color,
      shape_properties
    );
  }

  character2d_t::character2d_t(const character2d_t& o)
    : base_shape_t(o),
    wall_jump(o.wall_jump),
    movement_cb(o.movement_cb),
    anim_controller(o.anim_controller) 
  {
    std::memcpy(&previous_movement_sign, &o.previous_movement_sign,
      offsetof(character2d_t, movement_type) - offsetof(character2d_t, previous_movement_sign));

    movement_cb.rebind(this);
  }

  character2d_t::character2d_t(character2d_t&& o) noexcept
    : base_shape_t(std::move(o)),
    wall_jump(std::move(o.wall_jump)),
    movement_cb(std::move(o.movement_cb)),
    anim_controller(std::move(o.anim_controller)) 
  {
    std::memcpy(&previous_movement_sign, &o.previous_movement_sign,
      offsetof(character2d_t, movement_type) - offsetof(character2d_t, previous_movement_sign));

    movement_cb.rebind(this);
  }

  character2d_t& character2d_t::operator=(const character2d_t& o)
  {
    if (this != &o) {
      base_shape_t::operator=(o);
      wall_jump = o.wall_jump;
      movement_cb = o.movement_cb;
      anim_controller = o.anim_controller;

      std::memcpy(&previous_movement_sign, &o.previous_movement_sign,
        offsetof(character2d_t, movement_type) - offsetof(character2d_t, previous_movement_sign));

      movement_cb.rebind(this);
    }
    return *this;
  }

  character2d_t& character2d_t::operator=(character2d_t&& o) noexcept
  {
    if (this != &o) {
      base_shape_t::operator=(std::move(o));
      wall_jump = std::move(o.wall_jump);
      movement_cb = std::move(o.movement_cb);
      anim_controller = std::move(o.anim_controller);

      std::memcpy(&previous_movement_sign, &o.previous_movement_sign,
        offsetof(character2d_t, movement_type) - offsetof(character2d_t, previous_movement_sign));

      movement_cb.rebind(this);
    }
    return *this;
  }

  void character2d_t::set_shape(fan::graphics::shape_t&& shape) {
    bool movement_was_enabled = movement_enabled;
    uint8_t saved_movement_type = movement_type;

    movement_cb.remove();
    physics::base_shape_t::set_shape(std::move(shape));

    if (movement_was_enabled) {
      movement_enabled = true;
      movement_type = saved_movement_type;
      movement_cb.rebind(this);
      movement_cb = add_movement_callback([](character2d_t* s) {
        s->process_movement(s->movement_type);
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

  void character2d_t::update_animation() {
    f32_t animation_fps = (get_linear_velocity().x / max_speed) * 30.f;
    if (current_animation_requires_velocity_fps) {
      if (has_animation()) {
        set_sprite_sheet_fps(std::abs(animation_fps));
      }
    }

    if (previous_movement_sign.x) {
      fan::vec2 uvp = get_tc_position();
      fan::vec2 uvs = get_tc_size();
      set_tc_size(fan::vec2(std::abs(uvs.x) * previous_movement_sign.x, uvs.y));
    }
  }

  bool character2d_t::is_on_ground(fan::physics::body_id_t main, std::array<fan::physics::body_id_t, 2> feet, bool jumping) {
    for (int i = 0; i < 2; ++i) {
      fan::physics::body_id_t body_id = feet[i];
      if (body_id.is_valid() == false) {
        body_id = main;
      }
      b2Vec2 velocity = b2Body_GetLinearVelocity(body_id);
      if (jumping == false && velocity.y < 0.01f) {
        int capacity = b2Body_GetContactCapacity(body_id);
        capacity = b2MinInt(capacity, 4);
        b2ContactData contactData[4];
        int count = b2Body_GetContactData(body_id, contactData, capacity);
        for (int i = 0; i < count; ++i) {
          b2BodyId bodyIdA = b2Shape_GetBody(contactData[i].shapeIdA);
          f32_t sign = 0.0f;
          if (B2_ID_EQUALS(bodyIdA, body_id)) {
            sign = -1.0f;
          }
          else {
            sign = 1.0f;
          }
          if (sign * contactData[i].manifold.normal.y < -0.9f) {
            return true;
          }
        }
      }
    }
    return false;
  }

  void character2d_t::process_movement(uint8_t movement, f32_t friction) {
    fan::vec2 velocity = get_linear_velocity();

    fan::physics::shape_id_t colliding_wall_id;
    wall_jump.normal = fan::physics::check_wall_contact(*this, &colliding_wall_id);

    fan::vec2 input_vector = fan::window::get_input_vector();

    switch (movement) {
    case movement_e::side_view: {
      bool on_ground = is_on_ground(*this, std::to_array(feet), jumping);
      f32_t air_control_multiplier = on_ground ? 1.0f : 0.8f;
      move_to_direction(fan::vec2(input_vector.x, 0) * air_control_multiplier);

      bool can_jump = on_ground || (((fan::time::now() - last_ground_time) / 1e+9 <= coyote_time) && !on_air_after_jump);

      if (on_ground) {
        last_ground_time = fan::time::now();
        on_air_after_jump = false;
      }

      if (wall_jump.normal.x && input_vector.x) {
        colliding_wall_id.set_friction(0.f);
        if (!jumping && velocity.y > 0) {
          fan::physics::apply_wall_slide(*this, wall_jump.normal, wall_jump.slide_speed);
        }
      }
      else if (colliding_wall_id) {
        colliding_wall_id.set_friction(fan::physics::shape_properties_t().friction);
      }

      bool move_up = fan::window::is_action_down("move_up");

      if (!move_up) {
        jump_consumed = false;
        jumping = false;
      }

      if (move_up && !jump_consumed && handle_jump) {
        if (wall_jump.normal && !on_ground) {
          fan::vec2 vel = get_linear_velocity();
          fan::physics::wall_jump(*this, wall_jump.normal, wall_jump.push_away_force, jump_impulse);
          on_air_after_jump = true;
          jumping = true;
          jump_consumed = true;
        }
        else if (can_jump) {
          fan::vec2 vel = get_linear_velocity();
          set_linear_velocity(fan::vec2(vel.x, 0));
          on_air_after_jump = true;
          apply_linear_impulse_center({ 0, -jump_impulse });
          jumping = true;
          jump_consumed = true;
        }
      }
      else {
        jumping = false;
      }

      jump_delay = 0;
      break;
    }

    case movement_e::top_view: {
      move_to_direction(input_vector);
      break;
    }
    }

    if (auto_update_animations) {
      anim_controller.update(*this);
      update_animation();
    }
  }

  void character2d_t::move_to_direction(const fan::vec2& direction) {
    fan::vec2 input_dir = direction.sign();
    fan::vec2 vel = get_linear_velocity();

    if (input_dir.x != 0) {
      vel.x += input_dir.x * force;
      vel.x = fan::math::clamp(vel.x, -max_speed, max_speed);
    }
    else {
      f32_t deceleration_factor = 0.05f;
      vel.x = fan::math::lerp(vel.x, 0.f, deceleration_factor);
    }

    set_linear_velocity({ vel.x, vel.y });
    previous_movement_sign = input_dir;
  }

  void character2d_t::set_physics_position(const fan::vec2& p) {
    fan::physics::entity_t::set_physics_position(p);
    fan::graphics::shape_t::set_position(p);
  }

  void character2d_t::enable_default_movement(uint8_t movement) {
    movement_enabled = true;
    movement_type = movement;

    movement_cb = add_movement_callback(
      [movement] (character2d_t* self) {
      self->process_movement(movement);
    }
    );
  }

  void character2d_t::update_animations() {
    if (auto_update_animations) {
      anim_controller.update(*this);
      update_animation();
    }
  }

  void character2d_t::setup_default_animations() {
    auto anims = get_all_animations();

    struct anim_t {
      int fps = 0;
      fan::graphics::animation_nr_t id{};
    } attack, idle, run, hurt;

    for (auto& [name, anim_id] : anims) {
      auto& a = fan::graphics::all_animations[anim_id];

      if (name == "attack0") attack = { a.fps, anim_id };
      else if (name == "idle") idle = { a.fps, anim_id };
      else if (name == "run") run = { a.fps, anim_id };
      else if (name == "hurt") hurt = { a.fps, anim_id };
    }

    if (attack.fps) {
      anim_controller.add_state("attack0", {
        .animation_id = attack.id,
        .fps = attack.fps,
        .condition = [attack, was_mouse_clicked = false](character2d_t& c) mutable -> bool { 
        if (fan::window::is_mouse_clicked() && !was_mouse_clicked) {
          c.reset_current_sprite_sheet_animation();
          was_mouse_clicked = true;
        }
        if (was_mouse_clicked) {
          if (c.is_animation_finished(attack.id)) {
            was_mouse_clicked = false;
          }
          return true;
        }
        return false;
      }
        });
    }
    if (hurt.fps) {
      anim_controller.add_state("hurt", {
        .animation_id = hurt.id,
        .fps = hurt.fps,
        .condition = [](character2d_t& c) { return  false /*todo*/; }
        });
    }
    if (idle.fps) {
      anim_controller.add_state("idle", {
        .animation_id = idle.id,
        .fps = idle.fps,
        .condition = [](character2d_t& c) { return std::abs(c.get_linear_velocity().x) < 10.f; }
        });
      set_current_animation_id(idle.id);
    }
    if (run.fps) {
      anim_controller.add_state("run", {
        .animation_id = run.id,
        .fps = run.fps,
        .condition = [](character2d_t& c) { return std::abs(c.get_linear_velocity().x) >= 10.f; }
        });
    }
    auto_update_animations = true;
  }

  void character2d_t::animation_controller_t::add_state(const std::string& name, const animation_state_t& state) {
    states[name] = state;
  }

  void character2d_t::animation_controller_t::update(character2d_t& character) {
    for (auto& [name, state] : states) {
      if (state.condition(character) && state.animation_id) {
        character.set_current_animation_id(state.animation_id);
        character.current_animation_requires_velocity_fps = state.velocity_based_fps;
        if (!state.velocity_based_fps) {
          character.set_sprite_sheet_fps(state.fps);
        }
        return;
      }
    }
  }

  character2d_t::movement_callback_handle_t character2d_t::add_movement_callback(std::function<void(character2d_t*)> fn) {
    using handle_t = movement_callback_handle_t;
    using fn_t = typename handle_t::fn_t;
    using add_fn = typename handle_t::add_fn;
    using rem_fn = typename handle_t::remove_fn;

    add_fn add = [](character2d_t* self, fn_t cb) {
      return fan::physics::add_physics_step_callback(
        [cb, self]() { cb(self); }
      );
    };

    rem_fn remove = [](character2d_t*, fan::physics::physics_step_callback_nr_t nr) {
      fan::physics::remove_physics_step_callback(nr);
    };

    return handle_t(
      this,
      std::move(add),
      std::move(remove),
      [fn](character2d_t* self) {
      fn(self);
    }
    );
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
            //     update_position(fan::physics::gphysics->world_id, bones[i].joint_id, pivot);
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
        //   update_reference_angle(fan::physics::gphysics->world_id, blower_left_arm.joint_id, vel_sgn == 1 ? quarter_pi : -quarter_pi);
        //    update_reference_angle(fan::physics::gphysics->world_id, blower_right_arm.joint_id, vel_sgn == 1 ? quarter_pi : -quarter_pi);
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
    b2WorldId worldId = fan::physics::gphysics->world_id;

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
    dummy_body.set_body(b2CreateBody(fan::physics::gphysics->world_id, &default_body));
    nr = fan::graphics::ctx().update_callback->NewNodeLast();
    // not copy safe
    (*fan::graphics::ctx().update_callback)[nr] = [this](void* ptr) {
    #if defined(fan_gui)
      if (fan::window::is_mouse_down()) {
        fan::vec2 p = fan::window::get_mouse_position() / fan::physics::length_units_per_meter;
        if (!B2_IS_NON_NULL(mouse_joint)) {
          b2AABB box;
          b2Vec2 d = {0.001f, 0.001f};
          box.lowerBound = b2Sub(p, d);
          box.upperBound = b2Add(p, d);

          QueryContext queryContext = {p, b2_nullBodyId};
          b2World_OverlapAABB(fan::physics::gphysics->world_id, box, b2DefaultQueryFilter(), QueryCallback, &queryContext);
          if (B2_IS_NON_NULL(queryContext.bodyId)) {

            b2MouseJointDef mouseDef = b2DefaultMouseJointDef();
            mouseDef.bodyIdA = dummy_body;
            mouseDef.bodyIdB = queryContext.bodyId;
            mouseDef.target = p;
            mouseDef.hertz = 5.0f;
            mouseDef.dampingRatio = 0.7f;
            mouseDef.maxForce = 1000.0f * b2Body_GetMass(queryContext.bodyId);
            mouse_joint = b2CreateMouseJoint(fan::physics::gphysics->world_id, &mouseDef);
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
    #endif
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
    return fan::physics::gphysics->create_capsule(
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
    return fan::physics::gphysics->is_on_sensor(test_id, sensor_id);
  }

  fan::physics::ray_result_t raycast(const fan::vec2& src, const fan::vec2& dst) {
    return fan::physics::gphysics->raycast(src, dst);
  }
}

#endif