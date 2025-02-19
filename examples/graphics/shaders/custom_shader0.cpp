#include <fan/pch.h>

int main() {

  loco_t loco;

  fan::vec2 initial_position = fan::vec2(loco.window.get_size() / 2);
  fan::vec2 initial_size = loco.window.get_size().y / 2;

  loco_t::image_t background;
  background = loco.create_image(fan::color(1, 0, 0, 1));

  fan::graphics::sprite_t sprite{ {
    .position = fan::vec3(initial_position, 1),
    .size = initial_size,
    .image = background,
    .blending = true
  } };

  fan::string shader_code;
  if (fan::io::file::read("2.glsl", &shader_code)) {
    return 1;
  }

  shader_code.resize(4096);

  loco_t::shader_t shader = loco.create_sprite_shader(shader_code);

  fan::color input_color = fan::colors::red / 10;
  input_color.a = 0.1;
  loco_t::image_t image = loco.image_load("images/lava_seamless.webp");

  loco_t::shader_shape_t::properties_t sp;
  sp.position = fan::vec3(fan::vec2(sprite.get_position()), 3);
  sp.size = sprite.get_size();
  sp.shader = shader;
  sp.blending = true;
  sp.image = image;
  

  loco_t::shape_t shader_shape = sp;

  bool shader_compiled = true;

  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) {
    loco.shader_set_vertex(shader, loco.shader_get(loco.shaper.GetShader(loco_t::shape_type_t::shader_shape)).svertex);
    loco.shader_set_fragment(shader, shader_code);
    shader_compiled = loco.shader_compile(shader);
  });

  fan::time::clock c;
  c.start();

  loco.loop([&] {

    loco.shader_set_value(shader, "time", c.elapsed() / 1e+9f);

    static bool toggle_color = false;
    if (ImGui::Checkbox("toggle color", &toggle_color)) {
      loco.image_unload(background);
      background = loco.create_image(toggle_color == false ? fan::colors::black : fan::colors::white);
    }

    if (shader_compiled == false) {
      ImGui::TextColored(fan::colors::green, "failed to compile shader");
    }
    
    initial_position = fan::vec2(loco.window.get_size() / 2);
    initial_size = loco.window.get_size().y / 2;

    static fan::vec2 offset = 0;
    if (ImGui::DragFloat2("offset", offset.data())) {
      sprite.set_position(initial_position + offset);
      shader_shape.set_position(initial_position + offset);
    }

    static fan::vec2 size_offset = 0;
    if (ImGui::DragFloat("size offset aspect", size_offset.data())) {
      size_offset.y = size_offset.x;
      sprite.set_size(initial_size + size_offset);
      shader_shape.set_size(initial_size + size_offset);
    }
    if (ImGui::DragFloat2("size_offset", size_offset.data())) {
      sprite.set_size(initial_size + size_offset);
      shader_shape.set_size(initial_size + size_offset);
    }

    if (ImGui::ColorEdit4("##c0", input_color.data())) {
      loco.shader_set_value(shader, "input_color", input_color);
    }
    if (ImGui::InputTextMultiline("##TextFileContents", shader_code.data(), shader_code.size(), ImVec2(-1.0f, -1.0f), ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_AutoSelectAll)) {
      fan::io::file::write("2.glsl", shader_code.c_str(), std::ios_base::binary);
    }

  });

  return 0;
}
