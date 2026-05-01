module;

module fan.graphics.stb;

import fan.io.file;
import fan.print;
import fan.graphics.stb_raw;

bool fan::stb::validate(fan::str_view_t path, const std::source_location& callers_path) {
  int x, y, channels;
  return fan::stb_raw::info(
    fan::io::file::find_relative_path(path, callers_path).string().c_str(),
    &x, &y, &channels
  );
}

bool fan::stb::load(fan::str_view_t path, info_t* image_info, const std::source_location& callers_path) {
  auto p = fan::io::file::find_relative_path(path, callers_path).string();

  image_info->data = fan::stb_raw::load(
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
      fan::stb_raw::failure_reason()
    );
    return true;
  }

  return false;
}

void fan::stb::free_image(void* data) {
  fan::stb_raw::free(data);
}