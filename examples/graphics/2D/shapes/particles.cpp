#include <fan/pch.h>

int main() {

  loco_t loco;
  //loco.lighting.ambient -= 0.5;

  loco.clear_color = fan::colors::black;

  auto particle_texture = loco.image_load("images/waterdrop.webp");

  fan::vec2 window_size = loco.window.get_size();

  loco_t::particles_t::properties_t p;
  p.position = fan::vec3(270, -487, 10);
  p.count = 1191;
  p.size = 2;
  p.begin_angle = 4.133;
  p.end_angle = 1.0;
  p.alive_time = 1e+9;
  p.gap_size = fan::vec2(354.535, 1.0);
  p.max_spread_size = fan::vec2(2423.231, 100.0);
  p.shape = loco_t::particles_t::shapes_e::rectangle;
  p.position_velocity = fan::vec2(757,  334);
  p.image = particle_texture;
  p.color = fan::color(0.4, 0.4, 1.4);
  loco_t::shape_t s = p;

  loco_t::shape_t bg = fan_init_struct(
    loco_t::rectangle_t::properties_t,
    .position = fan::vec3(window_size / 2, 0),
    .size = window_size / 2,
    .color = fan::color(0.0, 0, 0, 0.5)
  );

  loco.set_vsync(0);

  fan::graphics::file_save_dialog_t save_file_dialog;
  fan::graphics::file_open_dialog_t open_file_dialog;

  std::string fn;
  fan::color c = bg.get_color();
  f32_t color_intensity = 1;
  fan::vec3 position = p.position;
  f32_t size = p.size.x;

  f32_t alive_time = p.alive_time;
  fan::vec2 gap_size = p.gap_size;
  fan::vec2 max_spread_size = p.max_spread_size;
  fan::vec2 position_velocity = p.position_velocity;
  fan::vec2 size_velocity = p.size_velocity;
  fan::vec3 angle_velocity = p.angle_velocity;
  f32_t count = p.count;
  f32_t begin_angle = p.begin_angle;
  f32_t end_angle = p.end_angle;
  fan::vec3 angle = p.angle;
  int current_shape = 0;

  loco.loop([&] {
    if (ImGui::BeginMainMenuBar()) {

      if (ImGui::BeginMenu("File"))
      {
        if (ImGui::MenuItem("Open..", "Ctrl+O")) {
          open_file_dialog.load("json;fmm", &fn);
        }
        if (ImGui::MenuItem("Save as", "Ctrl+Shift+S")) {
          save_file_dialog.save("json;fmm", &fn);
        }
        ImGui::EndMenu();
      }

      if (open_file_dialog.is_finished()) {
        if (fn.size() != 0) {
          std::string data;
          fan::io::file::read(fn, &data);
          fan::json in = fan::json::parse(data);
          fan::graphics::shape_deserialize_t it;
          int i = 0;
          position = in["position"];
          size = fan::vec2(in["size"]).x;
          c = in["color"];
          alive_time = in["alive_time"];
          count = in["count"];
          position_velocity = in["position_velocity"];
          angle_velocity = in["angle_velocity"];
          begin_angle = in["begin_angle"];
          end_angle = in["end_angle"];
          angle = in["angle"];
          gap_size = in["gap_size"];
          max_spread_size = in["max_spread_size"];
          size_velocity = in["size_velocity"];
          current_shape = in["particle_shape"];
          while (it.iterate(in, &s)) {
          }

          s.set_image(particle_texture);
        }
        open_file_dialog.finished = false;
      }

      if (save_file_dialog.is_finished()) {
        if (fn.size() != 0) {
          fan::json temp;
          fan::graphics::shape_serialize(s, &temp);
          fan::io::file::write(fn, temp.dump(2), std::ios_base::binary);
        }
        save_file_dialog.finished = false;
      }
    }
    ImGui::EndMainMenuBar();

    ImGui::Begin("particle settings");


    if (ImGui::ColorEdit4("c", c.data())) {
      bg.set_color(c);
    }


    auto& ri = *(loco_t::particles_t::ri_t*)s.GetData(loco.shaper);

    static fan::color color = p.color;
    {

      const char* items[] = { "circle", "rectangle" };


      if (ImGui::Combo("shape", &current_shape, items, IM_ARRAYSIZE(items))) {
        ri.shape = current_shape;
      }
    }
    {
      if (ImGui::ColorPicker4("color", (f32_t*)color.data())) {
        ri.color = color * color_intensity;
      }
    }
    {
      if (ImGui::SliderFloat("color_intensity", &color_intensity, 0, 10)) {
        ri.color = color * color_intensity;
      }
    }
    {
      if (ImGui::SliderFloat3("position", position.data(), 0, 800.0f)) {
        ri.position = position;
      }
    }
    {

      if (ImGui::SliderFloat("size", &size, 0, 1000.0f)) {
        ri.size = size;
      }
    }
    {

      if (ImGui::SliderFloat("alive_time", &alive_time, 0, 10e+9)) {
        ri.alive_time = alive_time;
      }
    }
    if (ri.shape == loco_t::particles_t::shapes_e::rectangle) {
      
      if (ImGui::SliderFloat2("gap_size", gap_size.data(), -1000, 1000)) {
        ri.gap_size = gap_size;
      }
    }
    if (ri.shape == loco_t::particles_t::shapes_e::rectangle) {
      
      if (ImGui::SliderFloat2("max_spread_size", max_spread_size.data(), -10000, 10000)) {
        ri.max_spread_size = max_spread_size;
      }
    }
    {
      
      if (ImGui::SliderFloat2("position_velocity", position_velocity.data(), -10000, 10000)) {
        ri.position_velocity = position_velocity;
      }
    }
    {
      
      if (ImGui::SliderFloat2("size_velocity", size_velocity.data(), -100, 100)) {
        ri.size_velocity = size_velocity;
      }
    }
    {
      
      if (ImGui::DragFloat3("angle_velocity", angle_velocity.data(), 0.01, -fan::math::pi / 2, fan::math::pi / 2)) {
        ri.angle_velocity = angle_velocity;
      }
    }
    {
      
      if (ImGui::SliderFloat("count", &count, 1, 5000)) {
        ri.count = count;
      }
    }
    {
      if (ImGui::SliderFloat("begin_angle", &begin_angle, -fan::math::pi / 2, fan::math::pi / 2)) {
        ri.begin_angle = begin_angle;
      }
    }
    {
      if (ImGui::SliderFloat("end_angle", &end_angle, -fan::math::pi / 2, fan::math::pi / 2)) {
        ri.end_angle = end_angle;
      }
    }
    {

      if (ImGui::SliderFloat("angle", angle.data(), -fan::math::pi / 2, fan::math::pi / 2)) {
        ri.angle = angle;
      }
    }

    ImGui::End();
  });
}