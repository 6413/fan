module;

#if defined (FAN_WINDOW)

#if defined(FAN_2D)
  #include <fan/utility.h>
#endif

#endif

module fan.graphics.shapes.types;

#if defined (FAN_WINDOW)

#if defined(FAN_2D)

//import fan.utility;
import fan.io.file;
import fan.print.error;
import fan.memory;
import fan.types.json;
import fan.graphics.common_context;


#if defined(FAN_JSON) // this whole block is bad this shouldnt exist here its duplicate from shapes_impl.cpp
fan::json image_to_json(const fan::graphics::image_t& image) {
  fan::json image_json;
  if (image.iic()) {
    return image_json;
  }

  auto shape_data = (*fan::graphics::ctx().image_list)[image];
  if (shape_data.image_path.size()) {
    image_json["image_path"] = shape_data.image_path;
  }
  else {
    return image_json;
  }

  auto lp = fan::graphics::ctx()->image_get_settings(fan::graphics::ctx(), image);
  fan::graphics::image_load_properties_t defaults;
  if (lp.visual_output != defaults.visual_output) {
    image_json["image_visual_output"] = lp.visual_output;
  }
  if (lp.format != defaults.format) {
    image_json["image_format"] = lp.format;
  }
  if (lp.type != defaults.type) {
    image_json["image_type"] = lp.type;
  }
  if (lp.min_filter != defaults.min_filter) {
    image_json["image_min_filter"] = lp.min_filter;
  }
  if (lp.mag_filter != defaults.mag_filter) {
    image_json["image_mag_filter"] = lp.mag_filter;
  }

  return image_json;
}

fan::graphics::image_t json_to_image(const fan::json& image_json, const std::source_location& callers_path) {
  if (!image_json.contains("image_path")) {
    return fan::graphics::ctx().default_texture;
  }

  std::string path = image_json["image_path"];
  std::string relative_path = fan::io::file::find_relative_path(
    path, 
    callers_path
  ).generic_string();
  if (!fan::io::file::exists(relative_path))
  {
    return fan::graphics::ctx().default_texture;
  }
  path = std::filesystem::absolute(relative_path).generic_string();

  fan::graphics::image_load_properties_t lp;

  if (image_json.contains("image_visual_output")) {
    lp.visual_output = image_json["image_visual_output"];
  }
  if (image_json.contains("image_format")) {
    lp.format = image_json["image_format"];
  }
  if (image_json.contains("image_type")) {
    lp.type = image_json["image_type"];
  }
  if (image_json.contains("image_min_filter")) {
    lp.min_filter = image_json["image_min_filter"];
  }
  if (image_json.contains("image_mag_filter")) {
    lp.mag_filter = image_json["image_mag_filter"];
  }
  fan::graphics::image_nr_t image = fan::graphics::ctx()->image_load_path_props(
    fan::graphics::ctx(),
    path,
    lp,
    callers_path
  );
  (*fan::graphics::ctx().image_list)[image].image_path = path;
  return image;
}

#endif

namespace fan::graphics {
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

  sprite_sheet_t::sprite_sheet_t(const std::string& name, int fps, const std::vector<fan::graphics::image_t>& frame_images) {
    this->name = name;
    this->fps = fps;
    this->loop = true;
    for (std::size_t i = 0; i < frame_images.size(); ++i) {
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
    j.update(image_to_json(image), true);
    return j;
  }

  sprite_sheet_t::image_t& sprite_sheet_t::image_t::assign(const fan::json& j, const std::source_location& callers_path) {
    image = json_to_image(j, callers_path);
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
  sprite_sheet_id_t::sprite_sheet_id_t(std::uint32_t id) {
    this->id = id;
  }
  sprite_sheet_id_t::operator std::uint32_t() const {
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
#endif

#endif