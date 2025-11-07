#include <fan/pch.h>

int main() {

  loco_t loco;
  loco.set_vsync(0);

  // Properties for stars and planets
  loco_t::rectangle3d_t::properties_t star_props = {
      .position = fan::vec3(0, 0, 0),
      .size = 0.5,
      .color = fan::colors::white
  };
  loco_t::rectangle3d_t::properties_t planet_props = {
      .position = fan::vec3(0, 0, 0),
      .size = 1,
      .color = fan::colors::blue
  };


  fan::graphics::model_t::properties_t p;
  p.path = "models/cube.fbx";
  // sponza model has different coordinate system so fix it by rotating model matrix
  //p.model = fan::mat4(1).rotate(fan::math::pi / 2, fan::vec3(1, 0, 0));
  p.model = p.model.scale(0.01);
  std::vector<fan::graphics::model_t> models;
  //fan::graphics::model_t model(p);

  gloco->m_post_draw.push_back([&] {

    for (auto& m : models) {
      m.draw();
    }

//    model.draw();
  });

  std::vector<fan::graphics::shape_t> stars;
  std::vector<fan::graphics::shape_t> planets;
  int star_count = 100;
  int planet_count = 10;
  float galaxy_radius = 50.0;
  float orbit_radius = 10.0;

  // Generate stars in a spherical galaxy
  for (int i = 0; i < star_count; ++i) {
    float theta = fan::random::value_f32(0.0f, fan::math::two_pi);
    float phi = fan::random::value_f32(0.0f, fan::math::pi);
    float r = fan::random::value_f32(0.0f, galaxy_radius);
    star_props.position = fan::vec3(r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi));
   // stars.push_back(star_props);
    p.model = fan::mat4(1).translate(star_props.position);
    p.model = p.model.scale(0.01);
    models.push_back(p);
  }

  // Generate planets orbiting around a central point
  for (int i = 0; i < planet_count; ++i) {
    float angle = (i / (float)planet_count) * fan::math::two_pi;
    planet_props.position = fan::vec3(cos(angle) * orbit_radius, sin(angle) * orbit_radius, 0);
    planets.push_back(planet_props);
  }

  auto& camera = gloco->camera_get(gloco->perspective_camera.camera);
  fan::vec2 motion = 0;

  loco.window.add_mouse_motion([&](const auto& d) {
    motion = d.motion;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      camera.rotate_camera(d.motion);
    }
    });


  loco.loop([&] {
    double time = fan::time::clock::now() * 1e-9; // Time in seconds

    // Update planet positions to simulate orbits
    for (int i = 0; i < planets.size(); ++i) {
      float angle = (i / (float)planet_count) * fan::math::two_pi + time * 0.1; // Speed of orbit
      planets[i].set_position(fan::vec3(cos(angle) * orbit_radius, sin(angle) * orbit_radius, sin(angle) * 2.0));
      planets[i].set_color(fan::color::hsv(fmod(time + i * 0.1, 1.0) * 360.0, 100.0, 100.0)); // Changing colors
    }

    camera.move(100);
    });
}
