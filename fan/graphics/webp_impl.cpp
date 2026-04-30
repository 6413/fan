module;

#if defined(fan_compiler_gcc)
  // fixes collision with GLFW3 headers while doing import std;
	#ifndef _GCC_MAX_ALIGN_T
		#define _GCC_MAX_ALIGN_T
	#endif
#endif

// if windows fails with duplicate resource, remove mux folder from libwebp
#include <webp/encode.h>
#include <webp/decode.h>

module fan.graphics.webp;

import std;

import fan.io.file;
import fan.print;

bool fan::webp::get_image_size(fan::str_view_t path, fan::vec2ui* size, const std::source_location& callers_path) {
  std::string data;
  fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &data);
  return WebPGetInfo((std::uint8_t*)data.data(), data.size(), (int*)&size->x, (int*)&size->y) != 1;
}

bool fan::webp::decode(const std::uint8_t* webp_data, std::size_t size, info_t* image_info) {
  image_info->data = WebPDecodeRGBA(webp_data, size, &image_info->size.x, &image_info->size.y);
  image_info->channels = 4;
  return image_info->data == 0;
}

bool fan::webp::load(fan::str_view_t path, info_t* image_info, const std::source_location& callers_path) {
  std::string data;
  fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &data);
  bool failed = decode((const std::uint8_t*)data.data(), data.size(), image_info);
  if (failed) {
    fan::print_warning(std::string("failed to load image:") + std::string(path));
    return true;
  }
  return false;
}

std::size_t fan::webp::encode_rgba(const std::uint8_t* in, const fan::vec2& size, f32_t quality, std::uint8_t** out) {
  return WebPEncodeRGBA(in, size.x, size.y, size.x * 4, quality, out);
}

std::size_t fan::webp::encode_lossless_rgba(const std::uint8_t* in, const fan::vec2& size, std::uint8_t** out) {
  return WebPEncodeLosslessRGBA(in, size.x, size.y, size.x * 4, out);
}

void fan::webp::free_image(void* ptr) {
  WebPFree(ptr);
}

bool fan::webp::validate(fan::str_view_t path, const std::source_location& callers_path) {
  std::string data;
  static constexpr std::uint32_t webp_header_size = 32;
  data.reserve(webp_header_size);
  auto rpath = fan::io::file::find_relative_path(path, callers_path);
  if (fan::io::file::read(rpath.string(), &data, webp_header_size)) {
    return false;
  }
  int width, height;
  if (WebPGetInfo((const std::uint8_t*)data.c_str(), webp_header_size, &width, &height)) {
    return true;
  }
  return false;
}