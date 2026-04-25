#include <vector>
#include <numeric>
#include <string>

import fan;
using namespace fan::graphics;

int main() {
  fan::graphics::engine_t engine;

  std::vector<uint8_t> data(0x200); 

  // fill gradient
  std::iota(data.begin(), data.begin() + 256, 0);

  std::string duplicate_test = "AAAAAABBBBBBCCCCCC      ";
  std::copy(duplicate_test.begin(), duplicate_test.end(), data.begin() + 256);

  std::string markers = "DEBUG_MARKER_9999\n\r\t!@#$%^&*()";
  std::copy(markers.begin(), markers.end(), data.begin() + 300);
  engine.loop([&] {
    gui::hex_editor("hex_editor", data);
  });
}