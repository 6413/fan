#include fan_pch

int main() {

  loco_t loco;
  //loco.lighting.ambient -= 0.5;

  loco.clear_color = fan::colors::black;

  fan::vec2 viewport_size = loco.window.get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, viewport_size.x),
    fan::vec2(0, viewport_size.y)
  );

  loco_t::image_t smoke_texture{"images/waterdrop.webp"};

  loco_t::shapes_t::particles_t::properties_t p;
  p.position = fan::vec3(1300.f / 2 + 100, 1300.f/2, 10);
  p.count = 8;
  p.size = 150;
  p.begin_angle = 4.133;
  p.end_angle = 1.0;
  p.alive_time = 1e+9;
  p.position_velocity = fan::vec2(757,  334);
  p.image = &smoke_texture;
  p.color = fan::color(0.4, 0.4, 1.4);
  loco_t::shape_t s = p;

  fan::graphics::imgui_element_t e([&] {
    ImGui::Begin("particle settings");
    auto& vi = loco.shapes.particles.sb_get_vi(s);
    auto& ri = loco.shapes.particles.sb_get_ri(s);
    static f32_t color_intensity = 1;
    static fan::color color = p.color;
    {

      const char* items[] = { "circle", "rectangle" };
      static int current_item = 0;

      if (ImGui::Combo("shape", &current_item, items, IM_ARRAYSIZE(items))) {
        ri.shape = current_item;
      }
    }
    {
      if (ImGui::ColorPicker4("color", (f32_t*)color.data())) {
        vi.color = color * color_intensity;
      }
    }
    {
      if (ImGui::SliderFloat("color_intensity", &color_intensity, 0, 10)) {
        vi.color = color * color_intensity;
      }
    }
    {
      static fan::vec3 position = p.position;
      if (ImGui::SliderFloat3("position", position.data(), 0, 800.0f)) {
        vi.position = position;
      }
    }
    {
      static f32_t size = p.size.x;
      if (ImGui::SliderFloat("size", &size, 0, 1000.0f)) {
        vi.size = size;
      }
    }
    {
      static f32_t alive_time = p.alive_time;
      if (ImGui::SliderFloat("alive_time", &alive_time, 0, 10e+9)) {
        ri.alive_time = alive_time;
      }
    }
    if (ri.shape == loco_t::shapes_t::particles_t::shapes_e::rectangle) {
      static fan::vec2 gap_size = p.gap_size;
      if (ImGui::SliderFloat2("gap_size", gap_size.data(), -1000, 1000)) {
        ri.gap_size = gap_size;
      }
    }
    if (ri.shape == loco_t::shapes_t::particles_t::shapes_e::rectangle) {
      static fan::vec2 max_spread_size = p.max_spread_size;
      if (ImGui::SliderFloat2("max_spread_size", max_spread_size.data(), -10000, 10000)) {
        ri.max_spread_size = max_spread_size;
      }
    }
    {
      static fan::vec2 position_velocity = p.position_velocity;
      if (ImGui::SliderFloat2("position_velocity", position_velocity.data(), -10000, 10000)) {
        ri.position_velocity = position_velocity;
      }
    }
    {
      static fan::vec2 size_velocity = p.size_velocity;
      if (ImGui::SliderFloat2("size_velocity", size_velocity.data(), -100, 100)) {
        ri.size_velocity = size_velocity;
      }
    }
    {
      static fan::vec3 angle_velocity = p.angle_velocity;
      if (ImGui::DragFloat3("angle_velocity", angle_velocity.data(), 0.01, -fan::math::pi / 2, fan::math::pi / 2)) {
        ri.angle_velocity = angle_velocity;
      }
    }
    {
      static f32_t count = p.count;
      if (ImGui::SliderFloat("count", &count, 1, 5000)) {
        ri.count = count;
      }
    }
    {
      static f32_t begin_angle = p.begin_angle;
      if (ImGui::SliderFloat("begin_angle", &begin_angle, -fan::math::pi / 2, fan::math::pi / 2)) {
        ri.begin_angle = begin_angle;
      }
    }
    {
      static f32_t end_angle = p.end_angle;
      if (ImGui::SliderFloat("end_angle", &end_angle, -fan::math::pi / 2, fan::math::pi / 2)) {
        ri.end_angle = end_angle;
      }
    }
    {
      static fan::vec3 angle = p.angle;
      if (ImGui::SliderFloat("angle", angle.data(), -fan::math::pi / 2, fan::math::pi / 2)) {
        ri.angle = angle;
      }
    }

    ImGui::End();
  });

  loco.loop([&] {
    loco.get_fps();
  });
}