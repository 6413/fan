module;
#include <fan/utility.h>
#include <cstdint>
#include <cstring>
module fan.graphics.shapes.types;

//import fan.utility;
import fan.print.error;
import fan.memory;

namespace fan::graphics {
#if defined(FAN_2D)
  std::uint8_t* A_resize(void* ptr, std::uintptr_t size) {
    if (ptr) {
      if (size) {
        void* rptr = (void*)__generic_realloc(ptr, size);
        if (rptr == 0) {
          fan::throw_error();
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
          fan::throw_error();
        }
        return (std::uint8_t*)rptr;
      }
      else {
        return 0;
      }
    }
  }
#endif

    sprite_sheet_t::sprite_sheet_t() {
  }

  sprite_sheet_t::~sprite_sheet_t() {
  }

  sprite_sheet_id_t::sprite_sheet_id_t() = default;

  sprite_sheet_id_t::sprite_sheet_id_t(uint32_t id) {
    this->id = id;
  }

  sprite_sheet_id_t::operator uint32_t() const {
    return id;
  }

  sprite_sheet_id_t::operator bool() const {
    return id != (decltype(id))-1;
  }

  sprite_sheet_id_t sprite_sheet_id_t::operator++(int) {
    sprite_sheet_id_t temp(*this);
    ++id;
    return temp;
  }

  bool sprite_sheet_id_t::operator==(const sprite_sheet_id_t& other) const {
    return id == other.id;
  }

  bool sprite_sheet_id_t::operator!=(const sprite_sheet_id_t& other) const {
    return id != other.id;
  }
}