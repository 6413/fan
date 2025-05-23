#include <fan/time/timer.h>
import fan;

int main() {
  using namespace fan::graphics;

  fan::graphics::engine_t engine;
  std::vector<fan::graphics::image_t> images;

  fan::io::async_directory_iterator_t iterator;
  iterator.sort_alphabetically = true;
  iterator.callback = [&](const std::filesystem::directory_entry& entry) -> fan::event::task_t{
    std::string path_str = entry.path().string();
    if (fan::image::valid(path_str)) {
      images.emplace_back(engine.image_load(path_str));
      co_await fan::co_sleep(100);
    }
    co_return;
  };
  fan::io::async_directory_iterate(
    &iterator,
    "imagenet-sample-images-master"
  );
  engine.set_vsync(0);
  engine.set_target_fps(0);

  engine.loop([&] {
    gui::begin("images");
    f32_t thumbnail_size = 128.0f;
    f32_t panel_width = gui::get_content_region_avail().x;
    f32_t padding = 16.0f;
    int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);

    gui::columns(column_count, 0, false);
    for (auto& i : images) {
      gui::image(i, thumbnail_size);
      gui::next_column();
    }
    gui::end();
  });
}