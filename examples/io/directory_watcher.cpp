#include <fan/ev/ev.h>
#include <fan/types/print.h>

int main() {
  uv_loop_t* loop = uv_default_loop();

  fan::ev::fs_watcher_t watcher(loop, "./watch_dir");

  watcher.start([](const std::string& filename, int events) {
    fan::print("latest event for file: " + filename);
    if (events & UV_CHANGE) {
      fan::print("file modified");
    }
    if (events & UV_RENAME) {
      fan::print("file renamed/deleted");
    }
  });

  uv_run(loop, UV_RUN_DEFAULT);
  uv_loop_close(loop);

  return 0;
}