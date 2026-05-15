module;
#include <cstdint>
#include <cstdlib>

#include <webp/encode.h>
#include <webp/decode.h>

export module fan.graphics.webp_raw;

export namespace fan::webp_raw {
  inline int get_info(
    const std::uint8_t* data,
    std::size_t size,
    int* width,
    int* height
  ) {
    return ::WebPGetInfo(data, size, width, height);
  }

  inline std::uint8_t* decode_rgba(
    const std::uint8_t* data,
    std::size_t size,
    int* width,
    int* height
  ) {
    return ::WebPDecodeRGBA(data, size, width, height);
  }

  inline std::size_t encode_rgba(
    const std::uint8_t* in,
    int width,
    int height,
    int stride,
    float quality,
    std::uint8_t** out
  ) {
    return ::WebPEncodeRGBA(in, width, height, stride, quality, out);
  }

  inline std::size_t encode_lossless_rgba(
    const std::uint8_t* in,
    int width,
    int height,
    int stride,
    std::uint8_t** out
  ) {
    return ::WebPEncodeLosslessRGBA(in, width, height, stride, out);
  }

  inline void free_image(void* ptr) {
    ::WebPFree(ptr);
  }
}