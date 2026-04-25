#include <vector>
#include <string>

import fan;

using namespace fan::graphics;

struct app_t : engine_t {
  app_t() : data(rows * columns) {}
  void run() {
    engine_t::loop([&] {
      gui::hex_editor("hex_viewer", data);
    });
  }
  uint32_t rows = 0x10;
  uint32_t max_table = 0x400;
  uint32_t columns = (max_table + 0x10) / 0x10;
  std::vector<uint8_t> data;
};

int main() {
  app_t app;
  app.run();
}