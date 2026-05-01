module;

#define STB_IMAGE_IMPLEMENTATION
#include <fan/stb/stb_image.h>

export module fan.graphics.stb_raw;

export namespace fan::stb_raw {

  using stbi_uc = ::stbi_uc;

  inline int info(const char* filename, int* x, int* y, int* channels) {
    return ::stbi_info(filename, x, y, channels);
  }

  inline stbi_uc* load(const char* filename, int* x, int* y, int* channels, int desired_channels) {
    return ::stbi_load(filename, x, y, channels, desired_channels);
  }

  inline const char* failure_reason() {
    return ::stbi_failure_reason();
  }

  inline void free(void* data) {
    ::stbi_image_free(data);
  }
}