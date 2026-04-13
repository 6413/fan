module;
#include <fan/utility.h>
#include <cstdint>
#include <cstring>
#include <source_location>
module fan.graphics.shapes.types;

//import fan.utility;
import fan.print.error;
import fan.memory;
import fan.graphics.shapes; // TODO remove: only for fan::graphics::image_to_json

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

  sprite_sheet_t::sprite_sheet_t(const std::string& name, int fps, const std::vector<fan::graphics::image_t>& frame_images) {
    this->name = name;
    this->fps = fps;
    this->loop = true;
    for (int i = 0; i < frame_images.size(); ++i) {
      sprite_sheet_t::image_t frame;
      frame.image = frame_images[i];
      frame.hframes = 1;
      frame.vframes = 1;
      this->images.push_back(frame);
      this->selected_frames.push_back(i);
    }
  }

#if defined(FAN_JSON)
  sprite_sheet_t::image_t::operator fan::json() const {
    fan::json j;
    image_t defaults;
    if (hframes != defaults.hframes) {
      j["hframes"] = hframes;
    }
    if (vframes != defaults.vframes) {
      j["vframes"] = vframes;
    }
    j.update(fan::graphics::image_to_json(image), true);
    return j;
  }

  sprite_sheet_t::image_t& sprite_sheet_t::image_t::assign(const fan::json& j, const std::source_location& callers_path) {
    image = fan::graphics::json_to_image(j, callers_path);
    if (j.contains("hframes")) {
      hframes = j.at("hframes");
    }
    if (j.contains("vframes")) {
      vframes = j.at("vframes");
    }
    return *this;
  }
#endif

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