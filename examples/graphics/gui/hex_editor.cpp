#include <vector>
#include <numeric>
#include <string>

import fan;

using namespace fan::graphics;

int main() {
  engine_t engine;

  fan::bytes_t data(0x200); 

  // fill gradient
  std::iota(data.begin(), data.begin() + 256, 0);

  std::string duplicate_test = "AAAAAABBBBBBCCCCCC      ";
  std::copy(duplicate_test.begin(), duplicate_test.end(), data.begin() + 256);

  std::string markers = "DEBUG_MARKER_9999\n\r\t!@#$%^&*()";
  std::copy(markers.begin(), markers.end(), data.begin() + 300);

  gui::hex_editor_t editor;
  fan::bytes_t data2;
  editor.set_file_drop_callback([&data2](const fan::bytes_t& bytes) {
    data2 = bytes;
  });

  engine.loop([&] {
    gui::hex_editor("hex_editor", data);
    fan::io::memory_provider_t provider(data2);
    editor.render("hex_editor2", provider);
    if (data2.empty()) {
      auto* wnd_data = gui::find_window("hex_editor2");
      gui::text("Drag a file to me", {
        .pos     = fan::vec2(wnd_data->Pos) + fan::vec2(wnd_data->Size) / 2.f,
        .align   = gui::text_style_t::align_e::center,
        .overlay = true
      });
    }
  });
}