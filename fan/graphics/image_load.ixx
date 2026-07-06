module;

export module fan.graphics.image_load;

import std;

#if !defined(loco_no_stb)
  import fan.graphics.stb;
#endif

import fan.types;
import fan.types.compile_time_string;
import fan.print.error;
import fan.utility;
import fan.types.vector;
import fan.graphics.webp;

export namespace fan::image {
  struct image_type_e {
    enum {
      webp,
      stb
    };
  };

  struct info_t {
    void* data;
    fan::vec2i size;
    int channels = -1;
    std::uint8_t type;
  };

  bool valid(const std::string& path, const std::source_location& callers_path = std::source_location::current());
  bool load(fan::str_view_t path, info_t* image_info, fan::vec2ui max_size = 0, const std::source_location& callers_path = std::source_location::current());
  bool write(fan::str_view_t path, const info_t& image_info, f32_t quality = 80.f);
  bool write(fan::str_view_t path, void* data, fan::vec2i size, int channels, f32_t quality = 80.f);
  bool write(fan::str_view_t path, std::span<const std::uint8_t> data, fan::vec2i size, int channels, f32_t quality = 80.f);
  void free(info_t* image_info);
    
  void convert_channels(const std::uint8_t* src, std::uint8_t* dst, std::size_t pixels, int src_channels, int dst_channels, std::uint8_t default_alpha = 255);

  inline constexpr std::uint8_t missing_texture_pixels[16] = {
    0, 0, 0, 255,
    255, 0, 220, 255,
    255, 0, 220, 255,
    0, 0, 0, 255
  };
  inline constexpr std::uint8_t transparent_texture_pixels[16] = {
    60, 60, 60, 255,
    40, 40, 40, 255,
    40, 40, 40, 255,
    60, 60, 60, 255
  };

  struct owned_t {
    fan::vec2ui size = 0;
    std::shared_ptr<std::uint8_t> data;
    std::size_t data_size = 0;
    int channels = 0;

    bool valid() const {
      return data != nullptr && data_size != 0 && size.x != 0 && size.y != 0 && channels != 0;
    }

    std::span<std::uint8_t> bytes() {
      return {data.get(), data_size};
    }

    std::span<const std::uint8_t> bytes() const {
      return {data.get(), data_size};
    }
  };

  owned_t load_owned(fan::str_view_t path, fan::vec2ui max_size = 0, const std::source_location& callers_path = std::source_location::current());

  struct async_result_t {
    enum class state_e {
      loading,
      ready,
      failed
    };

    bool try_finish();
    void wait();

    std::atomic<state_e> state = state_e::loading;
    owned_t image;
  };

  struct async_cache_t {
    std::mutex mutex;
    std::shared_ptr<async_result_t> load(const std::string& path, fan::vec2ui max_size = 0);
    void clear();
  };

  bool has_pending_async_tasks();
  async_cache_t& async_cache();
}