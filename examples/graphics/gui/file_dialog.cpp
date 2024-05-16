#include <fan/pch.h>
#include <fan/graphics/file_dialog.h>

#include <thread>
#include <atomic>
#include <mutex>

int main() {
  loco_t loco;

  std::string out;
  fan::graphics::file_open_dialog_t fd;

  loco.loop([&] {
    if (ImGui::Button("open file")) {
      fd.load("png,jpg;pdf", &out);
    }

    if (fd.is_finished()) {

      fd.finished = false;
    }
  });
}