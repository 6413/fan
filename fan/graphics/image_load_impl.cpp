module;

module fan.graphics.image_load;

import std;

import fan.print;

namespace fan::image {

  bool valid(const std::string& path, const std::source_location& callers_path) {
    if (fan::webp::validate(path, callers_path)) {
      return true;
    }
    else if (fan::stb::validate(path, callers_path)) {
      return true;
    }
    return false;
  }

  bool load(fan::str_view_t path, info_t* image_info, const std::source_location& callers_path) {
    bool ret;
    if (fan::webp::validate(path, callers_path)) {
      ret = fan::webp::load(path, (fan::webp::info_t*)image_info, callers_path);
      image_info->type = image_type_e::webp;
    }
    else {
      #if !defined(loco_no_stb)
        ret = fan::stb::load(path, (fan::stb::info_t*)image_info, callers_path);
        image_info->type = image_type_e::stb;
      #endif
    }
  #if FAN_DEBUG >= fan_debug_low
    if (ret) {
      fan::print_warning("failed to load image data from path:", path);
    }
  #endif
    return ret;
  }

  bool write(fan::str_view_t path, const info_t& image_info, f32_t quality) {
    return fan::image::write(path, image_info.data, image_info.size, image_info.channels, quality);
  }

  bool write(fan::str_view_t path, void* data, fan::vec2i size, int channels, f32_t quality) {
    std::string_view p(path.data(), path.size());
    if (p.ends_with(".webp")) {
      return fan::webp::write(path, data, size, channels, quality);
    }
#if !defined(loco_no_stb)
    return fan::stb::write(path, data, size, channels, quality);
#else
    fan::print_warning("unsupported image format for writing:", path);
    return true;
#endif
  }

  bool write(fan::str_view_t path, std::span<const std::uint8_t> data, fan::vec2i size, int channels, f32_t quality) {
    return fan::image::write(path, (void*)data.data(), size, channels, quality);
  }

  void free(info_t* image_info) {
    if (image_info->type == image_type_e::webp) {
      fan::webp::free_image(image_info->data);
    }
    else if (image_info->type == image_type_e::stb) {
      #if !defined(loco_no_stb)
      fan::stb::free_image(image_info->data);
      #endif
    }
  }

  void convert_channels(const std::uint8_t* src, std::uint8_t* dst, std::size_t pixels, int src_channels, int dst_channels, std::uint8_t default_alpha) {
    if (src_channels == dst_channels) {
      std::memcpy(dst, src, pixels * src_channels);
      return;
    }
    for (std::size_t i = 0; i < pixels; ++i) {
      for (int c = 0; c < dst_channels; ++c) {
        dst[i * dst_channels + c] = (c < src_channels) ? src[i * src_channels + c] : (c == 3 ? default_alpha : 0);
      }
    }
  }

  owned_t load_owned(fan::str_view_t path, const std::source_location& callers_path) {
    info_t ii {};
    owned_t out;

    if (load(path, &ii, callers_path)) {
      return out;
    }

    out.size = ii.size;
    out.channels = ii.channels;
    out.data_size = std::size_t(out.size.x) * out.size.y * out.channels;
    out.data = std::shared_ptr<std::uint8_t>(
      new std::uint8_t[out.data_size],
      std::default_delete<std::uint8_t[]>()
    );

    std::memcpy(out.data.get(), ii.data, out.data_size);
    free(&ii);

    return out;
  }

  bool async_result_t::try_finish() {
    if (state != state_e::loading) {
      return true;
    }
    if (job.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      return false;
    }
    image = job.get();
    state = image.valid() ? state_e::ready : state_e::failed;
    return true;
  }

  void async_result_t::wait() {
    if (state != state_e::loading) {
      return;
    }
    image = job.get();
    state = image.valid() ? state_e::ready : state_e::failed;
  }

  std::shared_ptr<async_result_t> async_cache_t::load(const std::string& path) {
    std::lock_guard lock(mutex);

    if (auto it = images.find(path); it != images.end()) {
      return it->second;
    }

    auto result = std::make_shared<async_result_t>();
    result->job = std::async(std::launch::async, [path] {
      return load_owned(path);
    });

    images.emplace(path, result);

    return result;
  }

  async_cache_t& async_cache() {
    static async_cache_t cache;
    return cache;
  }

}