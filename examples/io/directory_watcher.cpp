import fan.print;
import fan.event;

int main() {
  fan::event::fs_watcher_t watcher("./");

  watcher.start([](const std::string& filename, int events) {
    fan::print("latest event for file: " + filename);
    if (events & fan::fs_change) {
      fan::print("file modified");
    }
    if (events & fan::fs_rename) {
      fan::print("file renamed/deleted");
    }
  });

  fan::event::loop();

  return 0;
}