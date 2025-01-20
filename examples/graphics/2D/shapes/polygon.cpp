#include <fan/pch.h>

loco_t::polygon_t::properties_t create_hexagon(const fan::vec2& position, f32_t radius, const fan::color& color) { 
  loco_t::polygon_t::properties_t pp; 
  // for triangle strip
  for (int i = 0; i < 6; ++i) { 
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(position, 100), color });
    
    f32_t angle1 = 2 * fan::math::pi * i / 6; 
    f32_t x1 = position.x + radius * std::cos(angle1); 
    f32_t y1 = position.y + radius * std::sin(angle1); 
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x1, y1), 100), color });
    
    f32_t angle2 = 2 * fan::math::pi * ((i + 1) % 6) / 6;
    f32_t x2 = position.x + radius * std::cos(angle2); 
    f32_t y2 = position.y + radius * std::sin(angle2); 
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x2, y2), 100), color });
  }
  
  return pp;
}

loco_t::polygon_t::properties_t create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
  loco_t::polygon_t::properties_t pp;
  // for triangle strip
  for (f32_t x = 0; x < groundWidth - width; x += width) {
    f32_t y1 = position.y / 2 + amplitude * std::sin(frequency * x);
    f32_t y2 = position.y / 2 + amplitude * std::sin(frequency * (x + width));
    
    // top
    pp.vertices.push_back({ fan::vec2(position.x + x, y1), fan::colors::red });
    // bottom
    pp.vertices.push_back({ fan::vec2(position.x + x, position.y), fan::colors::white });
    // next top
    pp.vertices.push_back({ fan::vec2(position.x + x + width, y2), fan::colors::red });
    // next bottom
    pp.vertices.push_back({ fan::vec2(position.x + x + width, position.y), fan::colors::white });
  }
  
  return pp;
}

std::vector<fan::vec2> ground_points(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
  std::vector<fan::vec2> outline_points;
  for (f32_t x = 0; x <= groundWidth; x += width) {
    f32_t y = position.y / 2 + amplitude * std::sin(frequency * x);
    outline_points.push_back(fan::vec2(position.x + x, y));
  }
  outline_points.push_back(fan::vec2(position.x + groundWidth, position.y));
  outline_points.push_back(fan::vec2(position.x, position.y));
  return outline_points;
}

int main() {
  loco_t loco;

  std::vector<fan::graphics::physics_shapes::circle_t> entities;
  for (int i = 0; i < 10; ++i) {
    entities.push_back(fan::graphics::physics_shapes::circle_t{{
      .position = fan::vec3(30 + i*50, -100, 400),
      .radius = fan::random::f32(16, 86),
      .color = fan::random::color(),
      .body_type = fan::physics::body_type_e::dynamic_body,
      .shape_properties{.friction=0.001f}
    }});
  }

  static constexpr f32_t amplitude = 100.0f;
  static constexpr f32_t frequency = 0.15f;
  static constexpr f32_t width = 40.0f;
  static constexpr f32_t ground_width = 2560;

  fan::graphics::polygon_t hexagon;
  fan::physics::entity_t hexagon_entity;
  {
    auto hexagon_pp = create_hexagon(fan::vec2(600, 300), 50, fan::colors::blue);
    hexagon = {{ 
      .vertices = hexagon_pp.vertices,
    }};

    std::vector<fan::vec2> hexagon_collision_points(hexagon_pp.vertices.size()-1);
    {
      for (std::size_t i = 1; i < hexagon_pp.vertices.size(); ++i) {
        hexagon_collision_points[i-1] = hexagon_pp.vertices[i].position;
      }
    }
    hexagon_entity = loco.physics_context.create_segment(hexagon_collision_points, b2_staticBody, {});
  }
  
  
  auto pp = create_sine_ground(fan::vec2(0, 800), amplitude, frequency, width, ground_width);
  fan::graphics::polygon_t ground{{
      .vertices = pp.vertices,
  }};

  auto points = ground_points(fan::vec2(0, 800), amplitude, frequency, width, ground_width);
  loco.physics_context.create_segment(points, b2_staticBody, {});

  loco.shaper.GetShapeTypes(loco_t::shape_type_t::polygon).draw_mode = GL_TRIANGLE_STRIP;

  loco.loop([&] {
    loco.physics_context.step(loco.delta_time);
  });
}