#include <fan/utility.h>
#include <cstdint>

module fan.graphics.shapes.types;

namespace fan::graphics {
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
}