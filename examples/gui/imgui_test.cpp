#include fan_pch

int main() {
  loco_t loco;

  fan::graphics::imgui_element_t element([&] {
    ImGui::Begin("Editor");
    fan::vec2 window_size = gloco->window.get_size();
    fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
    fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
    fan::vec2 offset = viewport_size - viewport_size;
    fan::vec2 s = viewport_size;
    gloco->default_camera->camera.set_ortho(
      fan::vec2(-s.x, s.x),
      fan::vec2(-s.y, s.y)
    );
    gloco->default_camera->viewport.set(viewport_pos, viewport_size, window_size);
    ImGui::End();
  });

  fan::graphics::rectangle_t rect{ {
      .position = 0,
      .size = loco.default_camera->viewport.get_size() / 4
  } };

  loco.loop([&] {    
    
  });

  return 0;
}
