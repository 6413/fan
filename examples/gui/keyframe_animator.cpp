#include fan_pch

#include _FAN_PATH(graphics/gui/keyframe_animator/editor.h)


int main() {
  loco_t loco;
  auto&io = ImGui::GetIO();
  io.FontGlobalScale = 1.5;
  ImGui::GetNeoSequencerStyle().Colors[ImGuiNeoSequencerCol_Keyframe] = ImVec4(0.7, 0.7, 0, 1);
  ImGui::GetNeoSequencerStyle().Colors[ImGuiNeoSequencerCol_Bg] = loco.clear_color + 0.05;
  ImGui::GetNeoSequencerStyle().Colors[ImGuiNeoSequencerCol_TopBarBg] = loco.clear_color + 0.1;

  fan::graphics::animation_editor_t editor;
  editor.texturepack.open_compiled("texture_packs/tilemap.ftp");

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t([&] {editor.handle_imgui(); });

  loco.loop([&] {

  });
}