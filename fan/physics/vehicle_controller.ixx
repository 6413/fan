module;

export module fan.physics.vehicle_controller;

import std;
import fan;
import fan.graphics;

export namespace fan::physics {

  struct vehicle_controller_t {
    
    vehicle_controller_t() = default;

    // We take a pointer to the physics body, which must be a base_shape_t derivative (like capsule_t)
    void bind(fan::graphics::physics::base_shape_t* physics_body) {
      body = physics_body;
    }

    void apply_thrust(f32_t force, f32_t dt) {
      if (!body) return;
      fan::vec2 dir(std::sin(body->get_angle().z), -std::cos(body->get_angle().z));
      body->apply_linear_impulse_center(dir * force * body->get_mass() * dt);
    }

    void apply_turn(f32_t turn_speed) {
      if (!body) return;
      body->set_angular_velocity(turn_speed);
    }

    fan::vec3 get_thrust_position(f32_t offset) const {
      if (!body) return 0;
      fan::vec2 dir(std::sin(body->get_angle().z), -std::cos(body->get_angle().z));
      return body->get_position() - fan::vec3(dir * offset, 0);
    }

    fan::graphics::physics::base_shape_t* body = nullptr;
  };

}
