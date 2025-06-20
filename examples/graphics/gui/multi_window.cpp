#include <fan/pch.h>

int main() {
  loco_t loco;

  loco_t::viewport_t viewport, viewport2;
  viewport = loco.viewport_create();
  viewport2 = loco.viewport_create();

  fan::graphics::render_view_t render_view;
  fan::vec2 window_size = loco.window.get_size();
  camera.camera = loco.orthographic_render_view.camera;
  loco.viewport_set(
    viewport,
    0, window_size, window_size
  );
  camera.viewport = viewport;
  
  struct vec3_hasher {
    std::size_t operator()(const fan::vec3& k) const {
      std::hash<f32_t> hasher;
      std::size_t hash_value = 17;
      hash_value = hash_value * 31 + hasher(k.x);
      hash_value = hash_value * 31 + hasher(k.y);
      hash_value = hash_value * 31 + hasher(k.z);
      return hash_value;
    }
  };

  std::unordered_map<fan::vec3, std::vector<loco_t::shape_t>, vec3_hasher> shapes;


  shapes[fan::vec3()].push_back(fan::graphics::rectangle_t{{
     .camera = &camera,
     .position = 200,
     .size = 100,
     .color = fan::colors::red,
 } });

  camera.viewport = viewport2;

  shapes[fan::vec3(1, 0, 0)].push_back(fan::graphics::rectangle_t{ {
    .camera = &camera,
     .position = 600,
     .size = 100,
     .color = fan::colors::blue
 } });


  shapes[fan::vec3(1, 0, 0)].clear();
  
  loco.loop([&] {

    if (ImGui::Begin("a")) {
      static bool x = 0;
      ImGui::Checkbox("d", &x);
      loco.set_imgui_viewport(viewport);
      fan::print("a", ImGui::GetCurrentContext());
    }
    else  {
      loco.viewport_zero(
        viewport
      );
    }

    ImGui::End();


    if (ImGui::Begin("b")) {
      static bool x = 0;
      ImGui::Checkbox("c", &x);
      loco.set_imgui_viewport(viewport2);
      fan::print("b", ImGui::GetCurrentContext());
    }
    
    else {
      loco.viewport_zero(
        viewport2
      );
    }
   
    ImGui::End();

  });
}