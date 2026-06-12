module;

module fan.graphics.async_image_subsystem;

import fan.graphics.loco;

namespace fan::graphics {
  void async_image_subsystem_t::init() {
    initialized = true;
  }

  void async_image_subsystem_t::destroy() {
    uploads.clear();
    initialized = false;
  }

  fan::graphics::async_image_t async_image_subsystem_t::load(
    const std::string& path,
    const fan::graphics::image_load_properties_t& properties
  ) {
    fan::graphics::async_image_t out;
    out.image = gloco()->default_texture;
    out.result = fan::image::async_cache().load(path);

    uploads.push_back({
      .image = out.image,
      .properties = properties,
      .result = out.result
    });

    return out;
  }

  void async_image_subsystem_t::process() {
    std::size_t uploaded = 0;

    for (std::size_t i = 0; i < uploads.size();) {
      auto& u = uploads[i];

      if (!u.result->try_finish()) {
        ++i;
        continue;
      }

      if (u.result->state == fan::image::async_result_t::state_e::ready) {
        if (uploaded >= max_uploads_per_frame) {
          ++i;
          continue;
        }

        fan::image::info_t info;
        info.data = u.result->image.data.get();
        info.size = u.result->image.size;
        info.channels = u.result->image.channels;

        fan::graphics::image_reload(u.image, info, u.properties);
        ++uploaded;
      }

      uploads[i] = std::move(uploads.back());
      uploads.pop_back();
    }
  }
  fan::graphics::async_image_t image_load_async(
    const std::string& path,
    const fan::graphics::image_load_properties_t& properties
  ) {
    return gloco()->async_image.load(path, properties);
  }
}