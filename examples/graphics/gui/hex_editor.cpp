#include <vector>
#include <numeric>
#include <string>

import fan;

using namespace fan::graphics;

#define INCLUDE_TEST_DATA 1

int main(int argc, char** argv) {
  fan::args_t args(argc, argv);

  engine_t engine;

#if INCLUDE_TEST_DATA
  fan::bytes_t data(0x200); 

  // fill gradient
  std::iota(data.begin(), data.begin() + 256, 0);

  std::string duplicate_test = "AAAAAABBBBBBCCCCCC      ";
  std::copy(duplicate_test.begin(), duplicate_test.end(), data.begin() + 256);

  std::string markers = "DEBUG_MARKER_9999\n\r\t!@#$%^&*()";
  std::copy(markers.begin(), markers.end(), data.begin() + 300);
#endif

  gui::hex_editor_t editor;

  fan::bytes_t user_data;
  if (args.size() > 1) {
    user_data = fan::io::file::read_binary(args[1]);
  }

  fan::io::memory_provider_t provider(user_data);
  editor.set_file_drop_callback([&user_data, &provider](const fan::bytes_t& bytes) {
    user_data = bytes;
    provider = {user_data};
  });

  engine.loop([&] {
#if INCLUDE_TEST_DATA
    gui::hex_editor("hex_editor", data);
#endif
    constexpr auto editor_name = "HexEditor";
    editor.render(editor_name, provider);
    if (user_data.empty()) {
      auto* wnd_data = gui::find_window(editor_name);
      gui::text("Drop a file here to inspect it", {
        .pos     = fan::vec2(wnd_data->Pos) + fan::vec2(wnd_data->Size) / 2.f,
        .align   = gui::text_style_t::align_e::center,
        .overlay = true
      });
    }
  });
}