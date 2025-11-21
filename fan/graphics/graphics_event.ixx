module;

export module fan.graphics.event;

import std;
import fan.event;
import fan.types.vector;
import fan.graphics;

export namespace fan::graphics::event {
  void image_to_texture_pack_watcher(
    fan::event::fs_watcher_t& fs_watcher,
    const std::string& file_name,
    const fan::vec2& tile_size,
    const std::string& texture_pack_path, // path and name
    std::function<void()> fs_event_cb = [] {},
    const std::string& exe_path = "image2texturepack.exe" // todo: use real code
  ) {
    fs_watcher.start([=, &fs_watcher](const std::string& filename, int events) {

      if (!(events & fan::fs_change)) {
        return;
      }

      if (!filename.contains(file_name)) {
        return;
      }

      std::string full_path = fs_watcher.watch_path + file_name;

      fan::graphics::image_t img = fan::graphics::image_load(full_path);
      auto& img_data = fan::graphics::image_get_data(img);
      fan::vec2 size = img_data.size / tile_size;
      fan::graphics::image_unload(img);

      std::string cmd = exe_path + " " +
        std::to_string(size.x) + " " +
        std::to_string(size.y) +
        " \"" + full_path + "\"" +
        " \"" + texture_pack_path + "\"";

      system(cmd.c_str());

      fs_event_cb();
    });
  }
}