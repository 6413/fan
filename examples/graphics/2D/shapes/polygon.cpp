#include <fan/pch.h>

loco_t::polygon_t::properties_t create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
  loco_t::polygon_t::properties_t pp;
  
  for (f32_t x = 0; x < groundWidth - width; x += width) {
    f32_t y1 = position.y / 2 + amplitude * std::sin(frequency * x);
    f32_t y2 = position.y / 2 + amplitude * std::sin(frequency * (x + width));
    pp.vertices.push_back({ fan::vec2(position.x + x, y1), fan::colors::red });
    pp.vertices.push_back({ fan::vec2(position.x + x, position.y), fan::colors::white });
    pp.vertices.push_back({ fan::vec2(position.x + x + width, position.y), fan::colors::white });
    pp.vertices.push_back({ fan::vec2(position.x + x, y1), fan::colors::red });
    pp.vertices.push_back({ fan::vec2(position.x + x + width, position.y), fan::colors::white });
    pp.vertices.push_back({ fan::vec2(position.x + x + width, y2), fan::colors::red });
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

loco_t::polygon_t::properties_t create_hexagon(const fan::vec2& position, float radius, const fan::color& color) { 
  loco_t::polygon_t::properties_t pp; 
  for (int i = 0; i < 6; ++i) { 
    float angle = 2 * fan::math::pi * i / 6; 
    float x = position.x + radius * std::cos(angle); 
    float y = position.y + radius * std::sin(angle); 
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x, y), 100), color }); 
  }
  return pp;
}

int main() {
  loco_t loco;
  
  //std::vector<fan::graphics::physics_shapes::circle_t> entities;
  //for (int i = 0; i < 10; ++i) {
  //  entities.push_back(fan::graphics::physics_shapes::circle_t{{
  //    .position = fan::vec3(500 + i, -100, 400),
  //    .radius = 40,
  //    .color = fan::random::color(),
  //    .body_type = fan::physics::body_type_e::dynamic_body,
  //    .shape_properties{.friction=0.001f}
  //  }});
  //}

  const f32_t amplitude = 100.0f;
  const f32_t frequency = 0.15f;
  const f32_t width = 40.0f;
  const f32_t groundWidth = 2560;

  auto hexagon_pp = create_hexagon(fan::vec2(300, 600), 50, fan::colors::blue);
  fan::graphics::polygon_t hexagon{{ 
    .vertices = hexagon_pp.vertices,
  }};

  auto pp = create_sine_ground(fan::vec2(0, 800), amplitude, frequency, width, groundWidth);
  
  fan::graphics::polygon_t ground{{
      .vertices = pp.vertices,
  }};

  auto points = ground_points(fan::vec2(0, 800), amplitude, frequency, width, groundWidth);
  loco.physics_context.create_segment(points, b2_staticBody, {});



  //std::vector<loco_t::shape_t> lines(pp.vertices.size());
  //for (int i = 0; i < hexagon_pp.vertices.size() - 1; ++i) {
  //  lines[i] = fan::graphics::line_t{{
  //  .src = fan::vec3(*(fan::vec2*)&hexagon_pp.vertices[i].position, 10000),
  //  .dst = hexagon_pp.vertices[i + 1].position,
  //  .color = fan::colors::green
  //}};
  //}

  loco.shaper.GetShapeTypes(loco_t::shape_type_t::polygon).draw_mode = fan::opengl::GL_TRIANGLES;

  loco.loop([&] {
    loco.physics_context.step(1.0 / 1024.f);
  });
}