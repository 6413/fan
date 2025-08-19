#include <box2d/box2d.h>

import fan;

struct pile_t {
  pile_t() {
    b2World_SetPreSolveCallback(loco.physics_context.world_id, presolve_static, this);
  }
  static bool presolve_static(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, void* context) {
    pile_t* pile = static_cast<pile_t*>(context);
    return pile->presolve(shapeIdA, shapeIdB, manifold);
  }
  bool presolve(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold) const {
    return fan::physics::presolve_oneway_collision(shapeIdA, shapeIdB, manifold, player);
  }
  loco_t loco;

  fan::graphics::physics::character2d_t player{ fan::graphics::physics::capsule_t{{
    .position = fan::vec3(400, 400, 10),
    .radius = 16,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .shape_properties{.friction = 0.0f, .fixed_rotation = true},
  }} };
};

int main() {
  pile_t pile;
  fan::vec2 window_size = pile.loco.window.get_size();
  f32_t wall_thickness = 50.f;
  auto walls = fan::graphics::physics::create_stroked_rectangle(window_size / 2, window_size / 2, wall_thickness);

  fan::graphics::physics::rectangle_t platforms[2];
  platforms[0] = fan::graphics::physics::rectangle_t{ {
    .position = fan::vec2(window_size.x / 5, window_size.y / 1.5),
    .size = fan::vec2(wall_thickness * 4, wall_thickness / 4),
    .color = fan::color::from_rgba(0x30a6b6ff),
    .outline_color = fan::color::from_rgba(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::kinematic_body,
    .shape_properties{.presolve_events = true},
  } };
  platforms[1] = fan::graphics::physics::rectangle_t{ {
    .position = fan::vec2(500, 500),
    .size = wall_thickness / 4,
    .color = fan::color::from_rgba(0x30a6b6ff),
    .outline_color = fan::color::from_rgba(0x30a6b6ff) * 2,
    .body_type = fan::physics::body_type_e::static_body,
    .shape_properties{},
  } };

  pile.player.impulse = 100;
  pile.player.force = 15;
  pile.player.max_speed = 270;

  pile.loco.loop([&] {

    pile.loco.physics_context.step(pile.loco.delta_time);

    pile.player.process_movement();
    if (platforms[0].get_position().x < window_size.x / 4) {
      platforms[0].set_linear_velocity({200, 0});
    }
    else if (platforms[0].get_position().x > window_size.x / 1.5) {
      platforms[0].set_linear_velocity({ -200, 0 });
    }
  });
}