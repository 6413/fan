#include fan_pch

int main() {
  loco_t loco;

  fan::graphics::imgui_element_t element([&] {
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
    fan::string str;
    str.resize(10);
    ImGui::InputText("input:", str.data(), str.size());

    ImGui::End();
  });


  loco.loop([&] {    
    
  });

  return 0;
}
