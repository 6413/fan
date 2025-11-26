module;

#include <string>
#include <source_location>

module fan.graphics.image_load;

bool fan::image::valid(const std::string& path, const std::source_location& callers_path) {
  if (fan::webp::validate(path, callers_path)) {
    return true;
  }
  else if (fan::stb::validate(path, callers_path)) {
    return true;
  }
  return false;
}

bool fan::image::load(const std::string& file, info_t* image_info, const std::source_location& callers_path) {
  bool ret;
  if (fan::webp::validate(file, callers_path)) {
    ret = fan::webp::load(file, (fan::webp::info_t*)image_info, callers_path);
    image_info->type = image_type_e::webp;
  }
  else {
    #if !defined(loco_no_stb)
      ret = fan::stb::load(file, (fan::stb::info_t*)image_info, callers_path);
      image_info->type = image_type_e::stb;
    #endif
  }
#if fan_debug >= fan_debug_low
  if (ret) {
    fan::print_warning("failed to load image data from path:" + file);
  }
#endif
  return ret;
}

void fan::image::free(info_t* image_info) {
  if (image_info->type == image_type_e::webp) {
    fan::webp::free_image(image_info->data);
  }
  else if (image_info->type == image_type_e::stb) {
    #if !defined(loco_no_stb)
    fan::stb::free_image(image_info->data);
    #endif
  }
}