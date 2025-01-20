#include <fan/pch.h>

int main() {
  loco_t loco;
  loco.lighting.ambient = 0;

  fan::image::image_info_t ii;
  fan::image::load("images/cursor.webp", &ii);


  GLFWimage image_cursor;
  image_cursor.width = ii.size.x;
  image_cursor.height = ii.size.y;
  image_cursor.pixels = (unsigned char*)ii.data;

  GLFWcursor* cursor = glfwCreateCursor(&image_cursor, 0, 0);

  fan::image::free(&ii);



  loco_t::image_load_properties_t lp;
  lp.format = GL_RGBA; // Change this to GL_RGB
  lp.internal_format = GL_RGBA; // Change this to GL_RGB
  lp.min_filter = loco_t::image_filter::linear;
  lp.mag_filter = loco_t::image_filter::linear;
  lp.visual_output = GL_MIRRORED_REPEAT;

  loco_t::image_t image_dirt = loco.image_load("images/Dirt3.webp", lp);
  loco_t::image_t image_lava_top_layer = loco.image_load("images/lava_layer.webp", lp);
  loco_t::image_t image = loco.image_load("images/LavaFull0.webp", lp);
  loco_t::image_t noise_image = loco.create_noise_image(256);

  fan::vec2 window_size = loco.window.get_size();
  std::vector<loco_t::shape_t> sprites;

  sprites.push_back(fan::graphics::sprite_t{ {
    .position = fan::vec3(fan::vec2(640 - 160, 360), 254),
    .size = 160 / 2,
    .image = image_dirt,
    .blending = true
  } });
  sprites.push_back(fan::graphics::sprite_t{ {
  .position = fan::vec3(fan::vec2(640 + 160, 360), 254),
  .size = 160 / 2,
  .image = image_dirt,
  .blending = true
} });
  sprites.push_back(fan::graphics::sprite_t{ {
    .position = fan::vec3(fan::vec2(640, 360 - 160), 254),
    .size = 160 / 2,
    .image = image_dirt,
    .blending = true
  } });
  sprites.push_back(fan::graphics::sprite_t{ {
.position = fan::vec3(fan::vec2(640, 360 + 160), 254),
.size = 160 / 2,
.image = image_dirt,
.blending = true
} });
  sprites.push_back(fan::graphics::sprite_t{ {
  .position = fan::vec3(fan::vec2(640, 360), 254),
  .size = 160 / 2,
  .image = image,
  .images = {noise_image},
  .blending = true,
  .flags = 2
} });
  sprites.push_back(fan::graphics::sprite_t{ {
.position = fan::vec3(fan::vec2(640, 360), 255),
.size = 160 / 2,
.image = image_lava_top_layer,
.blending = true
} });


  loco_t::imgui_fs_var_t bloom[8];
  bloom[1] = loco_t::imgui_fs_var_t(gloco->m_fbo_final_shader, "bloom_intensity", 0.07, 0.01);
  bloom[2] = loco_t::imgui_fs_var_t(gloco->m_fbo_final_shader, "gamma2", 1.02, 0.01);
  bloom[3] = loco_t::imgui_fs_var_t(gloco->m_fbo_final_shader, "bloom_gamma", 0.120, 0.01);
  bloom[4] = loco_t::imgui_fs_var_t(gloco->m_fbo_final_shader, "exposure", 2.0, 0.01);
  bloom[5] = loco_t::imgui_fs_var_t(gloco->m_fbo_final_shader, "gamma", 0.9, 0.01);
  bloom[6] = loco_t::imgui_fs_var_t(gloco->m_fbo_final_shader, "bloom_strength", 1.270, 0.01);

  fan::graphics::light_t l{ {
      .position = sprites[sprites.size() - 1].get_position(),
      .size = sprites[0].get_size() * 2.35,
      .color = fan::color(0.5, 0.5, 0.5, 1),
      .flags = 2
  } };
  //sprites.push_back(fan::graphics::sprite_t{ {
  //.position = fan::vec3(window_size / 2 + fan::vec2(90, 90), 254),
  //.size = window_size.x / 8,
  //.image = image,
  //.images[0] = noise_image,
  //.blending = true,
  //} });


  int x = 5;

  auto shader_nr = gloco->shaper.GetShader(loco_t::shape_type_t::sprite);
  auto offset = loco_t::imgui_fs_var_t(shader_nr, "offset", fan::vec2(0, 0), 0.1);

  loco.loop([&] {
    gloco->shader_set_value(shader_nr, "offset", loco.get_mouse_position() / 8000);
    
    glfwSetCursor(loco.window.glfw_window, cursor);
    //static 
    //fan_imgui_dragfloat()

    //ImGui::Image(image, image_size);

    });

  return 0;
}