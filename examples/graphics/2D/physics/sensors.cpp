#include <fan/pch.h>

struct pile_t {
  pile_t() {
    b2World_SetPreSolveCallback(loco.physics_context.world_id, presolve_static, this);
  }
  static bool presolve_static(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, void* context) {
    pile_t* pile = static_cast<pile_t*>(context);
    return pile->presolve(shapeIdA, shapeIdB, manifold);
  }
  bool presolve(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold) const {
    return fan::physics::presolve_oneway_collision(shapeIdA, shapeIdB, manifold, player.character);
  }
  loco_t loco;

  fan::graphics::character2d_t player{ fan::graphics::physics_shapes::capsule_t{{
    .position = fan::vec3(400, 400, 10),
    .center0 = {0.f, -128.f},
    .center1 = {0.f, 128.0f},
    .radius = 16.f,
    .color = fan::color::hex(0x715a5eff),
    .outline_color = fan::color::hex(0x715a5eff) * 2,
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .mass_data{.mass = 0.01f},
    .shape_properties{.friction = 0.6f, .density = 0.1f, .fixed_rotation = true},
  }} };
};

int main() {
  pile_t pile;
  fan::vec2 window_size = pile.loco.window.get_size();
  f32_t wall_thickness = 50.f;
  auto walls = fan::graphics::physics_shapes::create_stroked_rectangle(window_size / 2, window_size / 2, wall_thickness);

  fan::graphics::physics_shapes::rectangle_t platforms[3];
  platforms[0] = fan::graphics::physics_shapes::rectangle_t{ {
    .position = fan::vec2(window_size.x / 5, window_size.y / 1.5),
    .size = fan::vec2(wall_thickness * 4, wall_thickness / 4),
    .color = fan::color::hex(0x30a6b6ff),
    .outline_color = fan::color::hex(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::kinematic_body,
    .shape_properties{.enable_presolve_events = true},
  } };
  platforms[1] = fan::graphics::physics_shapes::rectangle_t{ {
    .position = fan::vec2(500, 500),
    .size = wall_thickness / 4,
    .color = fan::color::hex(0x30a6b6ff),
    .outline_color = fan::color::hex(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::static_body,
    .shape_properties{},
  } };
  platforms[2] = fan::graphics::physics_shapes::rectangle_t{ {
    .position = fan::vec2(700, 500),
    .size = wall_thickness / 4,
    .color = fan::color::hex(0x30a6b6ff),
    .outline_color = fan::color::hex(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::static_body,
    .shape_properties{.is_sensor = true},
  } };

  pile.loco.loop([&] {
    pile.player.process_movement();
    pile.loco.physics_context.step(pile.loco.delta_time);

    if (pile.loco.physics_context.sensor_events.is_on_sensor(pile.player.character, platforms[2])) {
      fan::printcl("sensor colliding with platform");
    }

    if (platforms[0].get_position().x < window_size.x / 4) {
      b2Body_SetLinearVelocity(platforms[0], { 200, 0 });
    }
    else if (platforms[0].get_position().x > window_size.x / 1.5) {
      b2Body_SetLinearVelocity(platforms[0], { -200, 0 });
    }
  });
}