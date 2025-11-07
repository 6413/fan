#include <fan/pch.h>

int main() {

  loco_t loco;
  loco.lighting.ambient = fan::vec3(0.3, 0.3, 0.3);
  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(1);

  loco_t::gradient_t::properties_t gp;
  gp.position = fan::vec3(loco.window.get_size() / 2, 0);
  gp.size = 300;
  gp.color[0] = fan::color(1, 0, 0, 1);
  gp.color[1] = fan::color(1, 0, 0, 1);
  gp.color[2] = fan::color(0, 0, 1, 1);
  gp.color[3] = fan::color(0, 0, 1, 1);
  fan::graphics::shape_t rect = gp;

  fan::graphics::image_t image;
  
  p.image = loco.image_load("images/tire.webp");
  p.position = fan::vec3(loco.window.get_size() / 2, 1);
  p.size = 300;
  p.color.a = 1;
  fan::graphics::shape_t s0 = p;

  loco_t::light_t::properties_t lp;
  lp.position = fan::vec3(400, 400, 0);
  lp.size = 100;
  lp.color = fan::colors::yellow * 10;

    fan::graphics::shape_t l0 = lp;

    //auto g0 = loco_t::imgui_fs_var_t<f32_t>(&gloco->m_fbo_final_shader, "edge0", 0.2, 0.1);
    //auto g1 = loco_t::imgui_fs_var_t<f32_t>(&gloco->m_fbo_final_shader, "edge1", 0.4, 0.1);
    //auto g2 = loco_t::imgui_fs_var_t<f32_t>(&gloco->m_fbo_final_shader, "exposure", 2.0, 0.01);
    //auto g3 = loco_t::imgui_fs_var_t<f32_t>(&gloco->m_fbo_final_shader, "gamma", 2.0, 0.01);
    //auto g4 = loco_t::imgui_fs_var_t<f32_t>(&gloco->m_fbo_final_shader, "bloom_strength", 0.04, 0.01);

    loco.clear_color = fan::colors::black;
  f32_t x = 0;
  loco.loop([&] {
    //fan_imgui_dragfloat(loco.blur.bloom_filter_radius, 0.01, 0, 3);
    l0.set_position(loco.get_mouse_position());
    if (loco.window.key_pressed(fan::key_w)) {
      l0.set_size(l0.get_size() * 10);
    }
  });

  return 0;
}