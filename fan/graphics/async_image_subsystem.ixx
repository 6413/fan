module;

export module fan.graphics.async_image_subsystem;

import std;

import fan.graphics.common_context;
import fan.graphics.image_load;

export namespace fan::graphics {
  struct async_image_t {
    bool ready() const {
      return result != nullptr && result->state == fan::image::async_result_t::state_e::ready;
    }

    bool failed() const {
      return result != nullptr && result->state == fan::image::async_result_t::state_e::failed;
    }

    operator fan::graphics::image_t() const {
      return image;
    }

    fan::graphics::image_t image;
    std::shared_ptr<fan::image::async_result_t> result;
  };

  struct async_image_subsystem_t {
    struct upload_t {
      fan::graphics::image_t image;
      fan::graphics::image_load_properties_t properties;
      std::shared_ptr<fan::image::async_result_t> result;
    };

    void init();
    void destroy();

    fan::graphics::async_image_t load(
      const std::string& path,
      const fan::graphics::image_load_properties_t& properties = fan::graphics::image_presets::pixel_art()
    );

    void process();

    std::vector<upload_t> uploads;
    std::size_t max_uploads_per_frame = 1;
    bool initialized = false;
  };

  fan::graphics::async_image_t image_load_async(
    const std::string& path,
    const fan::graphics::image_load_properties_t& properties = fan::graphics::image_presets::pixel_art()
  );
}