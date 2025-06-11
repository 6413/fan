// generates image procedurally in a background thread while rendering the image as it's being generated
#include <atomic>
#include <vector>
#include <mutex>
#include <cmath>

import fan;

#include <fan/graphics/types.h>

using namespace fan::graphics;

struct image_generation_data_t {
  std::vector<uint8_t> rgb_data;
  fan::vec2ui image_size;
  std::atomic<uint32_t> generated_rows{0};
  std::mutex data_mutex;
  std::atomic<bool> should_quit{false};
  std::atomic<bool> generation_complete{false};
  std::atomic<bool> needs_update{false}; 
};

void generate_procedural_image(image_generation_data_t* data) {
  data->image_size = fan::vec2ui(1024, 1024);
  size_t total_size = data->image_size.x * data->image_size.y * 3;
  {
    std::lock_guard<std::mutex> lock(data->data_mutex); 
    data->rgb_data.resize(total_size, 0);
  }
  for (uint32_t y = 0; y < data->image_size.y; ++y) {
    if (data->should_quit) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(data->data_mutex);
      for (uint32_t x = 0; x < data->image_size.x; ++x) {
        size_t pixel_offset = (y * data->image_size.x + x) * 3;
        float cx = x * 2.0f / data->image_size.x - 1.0f, cy = y * 2.0f / data->image_size.y - 1.0f;
        float dist = std::sqrt(cx * cx + cy * cy), time_factor = y * 0.02f;
        float r = std::sin(cx * 5.0f + time_factor) * std::cos(cy * 5.0f);
        float g = std::sin(dist * 10.0f + time_factor * 2.0f) * 0.5f + 0.5f;
        float b = std::sin(std::atan2(cy, cx) * 5.0f + dist * 10.0f - time_factor * 3.0f) * 0.5f + 0.5f;
        data->rgb_data[pixel_offset] = (uint8_t)((r * 0.5f + 0.5f) * 255.0f);
        data->rgb_data[pixel_offset + 1] = (uint8_t)(g * 255.0f);
        data->rgb_data[pixel_offset + 2] = (uint8_t)(b * 255.0f);
      }
    }
    fan::event::sleep(5);
    data->generated_rows.store(y + 1);
    data->needs_update.store(true);
  }
  data->generation_complete.store(true);
}


int main() {

  engine_t engine;

  sprite_t image_sprite({
    .size = engine.window.get_size().y / 2
  });

  image_t procedural_image;

  image_generation_data_t image_data;
  image_data.needs_update.store(false);

  fan::vec2ui texture_size(1024, 1024);
  std::vector<uint8_t> initial_texture(texture_size.x * texture_size.y * 3, 0);

  fan::graphics::image_load_properties_t image_load_properties;
  image_load_properties.format = fan::graphics::image_format::rgb_unorm;
  image_load_properties.internal_format = image_load_properties.format;

  fan::image::info_t image_info;
  image_info.data = initial_texture.data();
  image_info.size = texture_size;
  procedural_image = engine.image_load(image_info, image_load_properties);

  image_sprite.set_image(procedural_image);

  auto thread_id = fan::event::thread_create([&image_data] {
    generate_procedural_image(&image_data);
  });

  std::string progress_message = "Generating image: 0%";

  fan_window_loop {

    if (image_data.needs_update.load()) {
      uint32_t current_rows = image_data.generated_rows.load();

      std::vector<uint8_t> temp_data;
      {
        std::lock_guard<std::mutex> lock(image_data.data_mutex);
        temp_data = image_data.rgb_data; 
      }

      fan::image::info_t update_info;
      update_info.channels = 3;
      update_info.data = temp_data.data();
      update_info.size = image_data.image_size;

      engine.image_reload(procedural_image, update_info, image_load_properties);

      image_data.needs_update.store(false);

      float progress = (float)current_rows / image_data.image_size.y * 100.0f;
      progress_message = "Generating image: " + std::to_string(int(progress)) + "%";
    }

    fan::graphics::gui::text(progress_message);
  };
  image_data.should_quit = true;

  return 0;
}