module;

#include <source_location>
#define STB_IMAGE_IMPLEMENTATION
#include <fan/stb/stb_image.h>

module fan.graphics.stb;

import fan.io.file;
import fan.print;

bool fan::stb::validate(const std::string& file, const std::source_location& callers_path) {
  int x, y, channels;
  return stbi_info(fan::io::file::find_relative_path(file, callers_path).string().c_str(), &x, &y, &channels);
}

bool fan::stb::load(const std::string& file, info_t* image_info, const std::source_location& callers_path) {
  image_info->data = stbi_load(fan::io::file::find_relative_path(file, callers_path).string().c_str(), &image_info->size.x, &image_info->size.y, &image_info->channels, 0);
  image_info->type = 1;
  if (!image_info->data) {
    fan::print_warning(std::string("failed to load image:") + std::string(file) + ", error:" + stbi_failure_reason());
    return true;
  }
  return false;
}

void fan::stb::free_image(void* data) {
  stbi_image_free(data);
}