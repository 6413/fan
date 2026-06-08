module;
#define STB_IMAGE_IMPLEMENTATION
#include <fan/stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <fan/stb/stb_image_write.h>
module fan.graphics.stb;
import fan.io.file;
import fan.print;
namespace fan::stb {

  bool validate(fan::str_view_t path, const std::source_location& callers_path) {
    int x, y, channels;
    return ::stbi_info(
      fan::io::file::find_relative_path(path, callers_path).string().c_str(),
      &x, &y, &channels
    );
  }

  bool load(fan::str_view_t path, info_t* image_info, const std::source_location& callers_path) {
    auto p = fan::io::file::find_relative_path(path, callers_path).string();

    image_info->data = ::stbi_load(
      p.c_str(),
      &image_info->size.x,
      &image_info->size.y,
      &image_info->channels,
      0
    );

    image_info->type = 1;

    if (!image_info->data) {
      fan::print_warning(
        std::string("failed to load image:") +
        std::string(path) +
        ", error:" +
        ::stbi_failure_reason()
      );
      return true;
    }

    return false;
  }

  bool write(fan::str_view_t path, const info_t& image_info, f32_t quality) {
    return write(path, image_info.data, image_info.size, image_info.channels, quality);
  }

  bool write(fan::str_view_t path, void* data, fan::vec2i size, int channels, f32_t quality) {
    std::string_view p(path.data(), path.size());
    auto path_str = std::string(p);
    if (p.ends_with(".png")) {
      return ::stbi_write_png(path_str.c_str(), size.x, size.y, channels, data, size.x * channels) == 0;
    }
    if (p.ends_with(".jpg") || p.ends_with(".jpeg")) {
      return ::stbi_write_jpg(path_str.c_str(), size.x, size.y, channels, data, (int)quality) == 0;
    }
    if (p.ends_with(".bmp")) {
      return ::stbi_write_bmp(path_str.c_str(), size.x, size.y, channels, data) == 0;
    }
    if (p.ends_with(".tga")) {
      return ::stbi_write_tga(path_str.c_str(), size.x, size.y, channels, data) == 0;
    }
    fan::print_warning("unsupported image format for writing with stb:", path);
    return true;
  }

  bool write(fan::str_view_t path, std::span<const std::uint8_t> data, fan::vec2i size, int channels, f32_t quality) {
    return write(path, (void*)data.data(), size, channels, quality);
  }

  void free_image(void* data) {
    ::stbi_image_free(data);
  }

}