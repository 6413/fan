module;

module fan.graphics.image_load;

#if defined(FAN_WINDOW)

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

  bool load(fan::str_view_t path, info_t* image_info, fan::vec2ui max_size, const std::source_location& callers_path) {
    bool ret;
    if (fan::webp::validate(path, callers_path)) {
      ret = fan::webp::load(path, (fan::webp::info_t*)image_info, callers_path);
      image_info->type = image_type_e::webp;
    }
    else {
      #if !defined(loco_no_stb)
        ret = fan::stb::load(path, (fan::stb::info_t*)image_info, max_size, callers_path);
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

  owned_t load_owned(fan::str_view_t path, fan::vec2ui max_size, const std::source_location& callers_path) {
    info_t ii {};
    owned_t out;

    if (load(path, &ii, max_size, callers_path)) {
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
    return state.load(std::memory_order_acquire) != state_e::loading;
  }

  void async_result_t::wait() {
    while (state.load(std::memory_order_acquire) == state_e::loading) {
      std::this_thread::yield();
    }
  }

  struct image_load_queue_t {
    image_load_queue_t() {
      int num_workers = std::thread::hardware_concurrency() - 1;
      if (num_workers <= 0) num_workers = 1;
      for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([this] {
          while (true) {
            std::tuple<std::string, std::shared_ptr<async_result_t>, fan::vec2ui> task;
            {
              std::unique_lock lock(mutex);
              cv.wait(lock, [this] { return stop || !queue.empty(); });
              if (stop && queue.empty()) return;
              task = std::move(queue.front());
              queue.pop();
              active_tasks++;
            }

            if (std::get<1>(task).use_count() == 1) {
              std::get<1>(task)->state.store(async_result_t::state_e::failed, std::memory_order_release);
              std::lock_guard lock(mutex);
              active_tasks--;
              continue;
            }

            auto out = load_owned(std::get<0>(task), std::get<2>(task), std::source_location::current());
            std::get<1>(task)->image = std::move(out);
            std::get<1>(task)->state.store(
              std::get<1>(task)->image.valid() ? async_result_t::state_e::ready : async_result_t::state_e::failed,
              std::memory_order_release
            );
            
            {
              std::lock_guard lock(mutex);
              active_tasks--;
            }
          }
        });
      }
    }

    ~image_load_queue_t() {
      {
        std::lock_guard lock(mutex);
        stop = true;
      }
      cv.notify_all();
      for (auto& worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
    }

    void push(const std::string& path, std::shared_ptr<async_result_t> result, fan::vec2ui max_size) {
      {
        std::lock_guard lock(mutex);
        queue.push({path, result, max_size});
      }
      cv.notify_one();
    }
    
    void clear() {
      std::lock_guard lock(mutex);
      while (!queue.empty()) queue.pop();
    }

    bool has_pending_tasks() {
      std::lock_guard lock(mutex);
      return !queue.empty() || active_tasks > 0;
    }

    std::queue<std::tuple<std::string, std::shared_ptr<async_result_t>, fan::vec2ui>> queue;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::thread> workers;
    int active_tasks = 0;
    bool stop = false;
  };

  image_load_queue_t& image_queue() {
    static image_load_queue_t queue;
    return queue;
  }

  std::shared_ptr<async_result_t> async_cache_t::load(const std::string& path, fan::vec2ui max_size) {
    std::lock_guard lock(mutex);

    auto result = std::make_shared<async_result_t>();
    image_queue().push(path, result, max_size);

    return result;
  }
  
  void async_cache_t::clear() {
    std::lock_guard lock(mutex);
    image_queue().clear();
  }

  bool has_pending_async_tasks() {
    return image_queue().has_pending_tasks();
  }

  async_cache_t& async_cache() {
    static async_cache_t cache;
    return cache;
  }

}

#endif