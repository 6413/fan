#include fan_pch

// when unlit sprite with gl_repeat texture is behind of sprite and that sprite has lighting there comes weird flickering
// unable to produce in here. Produceable in tilemap_editor.cpp when changing grid_visualize.background to be unlit_sprite_t

int main() {
  loco_t loco;

  loco_t::image_t image;
  image.load("images/brick.webp");

  fan::graphics::unlit_sprite_t unlit_sprite0{ {
    .position = fan::vec3(loco.window.get_size() / 2.0f, 0),
    .size = 600,
    .image = &image,
    .tc_size = 4
  } };

  loco.lighting.ambient = 0;

  fan::graphics::sprite_t sprite0{ {
  .position = fan::vec3(loco.window.get_size() / 2.0f, 1),
  .size = 400,
  .image = &image,
} };

  loco_t::shapes_t::light_t::properties_t lp;
  lp.position = fan::vec3(loco.window.get_size() / 2.0f, 0);
  lp.size = 400;
  lp.color = fan::colors::yellow * 10;
  // {
  loco_t::shape_t l0 = lp;

  loco.loop([&] {
    ImGui::Begin("wnd");
    loco.set_imgui_viewport(loco.default_camera->viewport);
    ImGui::End();

  });
}