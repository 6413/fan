#pragma once

struct entity_t {

   fan::graphics::physics::character2d_t body{ fan::graphics::physics::capsule_sprite_t{{
    .position = fan::vec3(fan::vec2(109, 123) * 64, 10),
    // collision radius,
    .center0 = {0.f, -24.f},
    .center1 = {0.f, 24.f},
    .size = 12,
    /*.color = fan::color::from_rgba(0x715a5eff),*/
    .body_type = fan::physics::body_type_e::dynamic_body,
    //.mass_data{.mass = 0.01f},
    .shape_properties{
      .friction = 0.6f, 
      .density = 0.1f, 
      .fixed_rotation = true,
      .contact_events = true,
    },
  }}};

   std::function<void()> update_cb;
};

std::vector<entity_t> entities;