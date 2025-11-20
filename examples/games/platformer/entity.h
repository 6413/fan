struct entity_t{
  entity_t() = default;
  entity_t(const fan::vec3& pos) {
    body = fan::graphics::physics::capsule_t{ {
      .position = pos,
      .body_type = fan::physics::body_type_e::dynamic_body,
      .shape_properties{ .fixed_rotation = true }
    } };
  }

  void on_hit(fan::vec2 hit_direction) {
    body.apply_linear_impulse_center(-hit_direction * 50.f);
  }

  fan::graphics::physics::capsule_t body;
};