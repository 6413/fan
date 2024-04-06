#include fan_pch

int main() {
  loco_t loco;

  loco_t::viewport_t viewport, viewport2;
  viewport.open();
  viewport2.open();

  fan::graphics::camera_t camera;
  fan::vec2 window_size = loco.window.get_size();
  camera.camera = loco.default_camera->camera;
  viewport.set(0, window_size, window_size);
  camera.viewport = viewport;
  

  fan::graphics::rectangle_t r{ {
     .camera = &camera,
     .position = 200,
     .size = 100,
     .color = fan::colors::red,
 } };

  camera.viewport = viewport2;

  fan::graphics::rectangle_t r2{ {
    .camera = &camera,
     .position = 600,
     .size = 100,
     .color = fan::colors::blue
 } };
  
  loco.loop([&] {

    if (ImGui::Begin("a")) {
      loco.set_imgui_viewport(viewport);
      fan::print("a", ImGui::GetCurrentContext());
    }
    else  {
      viewport.zero();
    }

    ImGui::End();


    if (ImGui::Begin("b")) {
      loco.set_imgui_viewport(viewport2);
      fan::print("b", ImGui::GetCurrentContext());
    }
    
    else {
      viewport2.zero();
    }
   
    ImGui::End();

  });
}