#include fan_pch

int main() {
  loco_t loco;

  fan::graphics::imgui_shape_element_t button_and_rectangle(
     loco_t::rectangle_t::properties_t{{
      .position = 0.3,
      .size = 0.2,
      .color = fan::colors::red
    }},
    [&] {
      static float f = 0.0f;
      static int counter = 0;

      ImGui::Begin("Hello, world!");

      ImGui::Text("This is some useful text.");

      ImGui::SliderFloat("float", &f, 0.0f, 1.0f);

      if (ImGui::Button("Button")) {
        counter++;
      }
      ImGui::SameLine();
      ImGui::Text("counter = %d", counter);

      ImGui::End();
    }
  );

  loco.loop([&] {    

    
  });

  return 0;
}
