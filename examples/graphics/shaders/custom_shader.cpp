#include fan_pch

int main() {

  loco_t loco;

  fan::graphics::sprite_t sprite{ {
    .position = fan::vec3(600, 600, 0),
    .size = fan::vec2(200, 200)
  }};

fan::string shader_code =
R"(
#version 330

layout (location = 0) out vec4 o_attachment0;
uniform vec4 ccc;

void main() {
  o_attachment0 = ccc;
}
)";

  shader_code.resize(4096);

  loco_t::shader_t shader = loco.create_sprite_shader(shader_code);
  loco_t::shader_t shader2 = loco.create_sprite_shader(shader_code);

  fan::color c0 = fan::colors::black, c1 = fan::colors::black;
  
  loco_t::shapes_t::shader_t::properties_t sp;
  sp.position = fan::vec3(100, 100, 1);
  sp.size = 100;
  sp.shader = &shader;
  sp.shader->get_shader().on_activate = [&] (loco_t::shader_t* shader){
    shader->set_vec4("ccc", c0);
  };
  sp.color.a = 0.25;
  sp.blending = true;

  loco_t::shape_t shape0 = sp;

  sp.position.x += 400;
  sp.position.z = 2;
  sp.shader = &shader2;
  sp.shader->get_shader().on_activate = [&](loco_t::shader_t* shader) {
    shader->set_vec4("ccc", c1);
  };

  loco_t::shape_t shape1 = sp;

  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) {
    shader.set_vertex(loco.get_sprite_vertex_shader());
    shader.set_fragment(shader_code);
    shader.compile();
    });

  loco.loop([&] {
    if (ImGui::InputTextMultiline("##TextFileContents", shader_code.data(), shader_code.size(), ImVec2(-1.0f, -1.0f), ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_AutoSelectAll)) {
      
    }
    ImGui::ColorEdit4("##c0", c0.data());
    ImGui::ColorEdit4("##c1", c1.data());

  });

  return 0;
}
