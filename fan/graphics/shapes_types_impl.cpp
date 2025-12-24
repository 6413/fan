module;

#include <fan/utility.h>

#include <cstdint>
#include <cstring>

module fan.graphics.shapes.types;

import fan.graphics.shapes;

namespace fan::graphics {
  context_shader_t::context_shader_t() {}
  context_shader_t::~context_shader_t() {}
  context_image_t::context_image_t() {}
  context_image_t::~context_image_t() {}
  context_t::context_t() {}
  context_t::~context_t() {}

#if defined(FAN_2D)

  std::uint8_t* A_resize(void* ptr, std::uintptr_t size) {
    if (ptr) {
      if (size) {
        void* rptr = (void*)__generic_realloc(ptr, size);
        if (rptr == 0) {
          fan::throw_error_impl();
        }
        return (std::uint8_t*)rptr;
      }
      else {
        __generic_free(ptr);
        return 0;
      }
    }
    else {
      if (size) {
        void* rptr = (void*)__generic_malloc(size);
        if (rptr == 0) {
          fan::throw_error_impl();
        }
        return (std::uint8_t*)rptr;
      }
      else {
        return 0;
      }
    }
  }
#endif
}

#if defined(FAN_2D)

#define shaper_set_ShapeTypeChange \
  __builtin_memcpy(new_renderdata, old_renderdata, element_count * fan::graphics::g_shapes->shaper.GetRenderDataSize(sti)); \
  __builtin_memcpy(new_data, old_data, element_count * fan::graphics::g_shapes->shaper.GetDataSize(sti));
void fan::graphics::shaper_t::_ShapeTypeChange(
  ShapeTypeIndex_t sti,
  KeyPackSize_t keypack_size,
  uint8_t* keypack,
  MaxElementPerBlock_t element_count,
  const void* old_renderdata,
  const void* old_data,
  void* new_renderdata,
  void* new_data
) {
  shaper_set_ShapeTypeChange
}
#endif