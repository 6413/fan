module;
#define STB_IMAGE_IMPLEMENTATION
#include <fan/stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <fan/stb/stb_image_write.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <fan/stb/stb_image_resize2.h>
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

  bool load(fan::str_view_t path, info_t* image_info, fan::vec2ui max_size, const std::source_location& callers_path) {
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

    if (max_size.x > 0 && max_size.y > 0) {
      if ((std::uint32_t)image_info->size.x > max_size.x || (std::uint32_t)image_info->size.y > max_size.y) {
        fan::print("Downscaling image:", path, "from", image_info->size, "to max", max_size);
        // Calculate aspect ratio preserving size
        f32_t ratio_x = (f32_t)max_size.x / image_info->size.x;
        f32_t ratio_y = (f32_t)max_size.y / image_info->size.y;
        f32_t ratio = std::min(ratio_x, ratio_y);
        
        int new_x = std::max(1, (int)(image_info->size.x * ratio));
        int new_y = std::max(1, (int)(image_info->size.y * ratio));

        unsigned char* resized_data = (unsigned char*)malloc(new_x * new_y * image_info->channels);
        if (resized_data) {
          auto result = stbir_resize_uint8_linear(
            image_info->data, image_info->size.x, image_info->size.y, 0,
            resized_data, new_x, new_y, 0,
            (stbir_pixel_layout)image_info->channels
          );
          if (result == NULL) {
            fan::print("Failed to resize image:", path);
            free(resized_data);
          }
          else {
            ::stbi_image_free(image_info->data);
            image_info->data = resized_data;
            image_info->size.x = new_x;
            image_info->size.y = new_y;
          }
        }
      }
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