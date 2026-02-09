module;

#if defined(FAN_OPENGL)
  #include <fan/graphics/opengl/init.h>
#endif

#include <source_location>
#include <cmath>
#include <string>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>


module fan.graphics.shapes;

import fan.utility;
import fan.graphics.gui.base;

#if defined(FAN_GUI)
  import fan.graphics.gui.text_logger;
#endif

#if defined(FAN_2D)

#define shaper_get_key_safe(return_type, kps_type, variable) \
  [key_pack] ()-> auto& { \
    auto o = g_shapes->shaper.GetKeyOffset( \
      offsetof(fan::graphics::kps_t::CONCAT(_, kps_type), variable), \
      offsetof(fan::graphics::kps_t::kps_type, variable) \
    );\
    static_assert(std::is_same_v<decltype(fan::graphics::kps_t::kps_type::variable), fan::graphics::return_type>, "possibly unwanted behaviour"); \
    return *(fan::graphics::return_type*)&key_pack[o];\
  }()


#define shape_get_vi(shape) (*(fan::graphics::shapes::shape##_t::vi_t*)GetRenderData(fan::graphics::g_shapes->shaper))
#define shape_get_ri(shape) (*(fan::graphics::shapes::shape##_t::ri_t*)GetData(fan::graphics::g_shapes->shaper))

#endif

namespace fan::graphics {

  // warning does deep copy, addresses can die
  fan::graphics::context_shader_t shader_get(fan::graphics::shader_nr_t nr) {
    fan::graphics::context_shader_t context_shader;
    if (0) {}
  #if defined(FAN_OPENGL)
    else if (fan::graphics::ctx().window->renderer == fan::window_t::renderer_t::opengl) {
      context_shader.gl = *(fan::opengl::context_t::shader_t*)fan::graphics::ctx()->shader_get(fan::graphics::ctx(), nr);
    }
  #endif
  #if defined(FAN_VULKAN)
    else if (fan::graphics::ctx().window->renderer == fan::window_t::renderer_t::vulkan) {
      context_shader.vk = *(fan::vulkan::context_t::shader_t*)fan::graphics::ctx()->shader_get(fan::graphics::ctx(), nr);
    }
  #endif
    return context_shader;
  }


  #if defined(FAN_JSON)
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

#if defined(FAN_2D)

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

  size_t sprite_sheet_id_hash_t::operator()(const sprite_sheet_id_t& sheet_id) const noexcept {
    return std::hash<uint32_t>()(sheet_id.id);
  }

  std::size_t sprite_sheet_pair_hash_t::operator()(const std::pair<sprite_sheet_id_t, std::string>& p) const noexcept {
    std::size_t h1 = sprite_sheet_id_hash_t{}(p.first);
    std::size_t h2 = std::hash<std::string>{}(p.second);
    return h1 ^ (h2 << 1);
  }

  // cache path + file unique sprite_sheet id
  ss_cache_t& ss_cache() {
    static ss_cache_t cache;
    return cache;
  }

  ss_map_t& all_sprite_sheets() {
    static ss_map_t sheets;
    return sheets;
  }

  sprite_sheet_id_t& ss_counter() {
    static sprite_sheet_id_t counter = 0;
    return counter;
  }

  ss_lookup_t& ss_lookup() {
    static ss_lookup_t table;
    return table;
  }

  ss_shapes_t& shape_sprite_sheets() {
    static ss_shapes_t sheets;
    return sheets;
  }

  sprite_sheet_id_t& shape_ss_counter() {
    static sprite_sheet_id_t counter = 0;
    return counter;
  }

  sprite_sheet_t& get_sprite_sheet(sprite_sheet_id_t nr) {
    auto found_sheet = fan::graphics::all_sprite_sheets().find(nr);
    if (found_sheet == fan::graphics::all_sprite_sheets().end()) {
      fan::throw_error("Sprite sheet not found");
    }
    return found_sheet->second;
  }

  sprite_sheet_t& get_sprite_sheet(sprite_sheet_id_t shape_sprite_sheet_id, const std::string& sprite_sheet_name) {
    auto found = ss_lookup().find(std::make_pair(shape_sprite_sheet_id, sprite_sheet_name));
    if (found == ss_lookup().end()) {
      fan::throw_error("Failed to find sprite sheet:" + sprite_sheet_name);
    }
    return get_sprite_sheet(found->second);
  }

  std::vector<fan::graphics::sprite_sheet_id_t>& get_shape_sprite_sheets(sprite_sheet_id_t shape_sprite_sheet_id) {
    auto found = shape_sprite_sheets().find(shape_sprite_sheet_id);
    if (found == shape_sprite_sheets().end()) {
      fan::throw_error("Failed to find sprite sheet:" + std::to_string((uint32_t)shape_sprite_sheet_id));
    }
    return found->second;
  }

  void rename_shape_sprite_sheet(sprite_sheet_id_t shape_sprite_sheet_id, const std::string& old_name, const std::string& new_name) {
    auto& previous_sheets = shape_sprite_sheets()[shape_sprite_sheet_id];
    auto found = std::find_if(previous_sheets.begin(), previous_sheets.end(), [old_name](const sprite_sheet_id_t nr) {
      auto found = fan::graphics::all_sprite_sheets().find(nr);
      if (found == fan::graphics::all_sprite_sheets().end()) {
        fan::throw_error("Sprite sheet nr expired (bug)");
      }
      return found->second.name == old_name;
    });
    if (found == previous_sheets.end()) {
      fan::throw_error("Sprite sheet:" + old_name, ", not found");
    }
    sprite_sheet_id_t previous_sheet_id = *found;
    auto prev_found = fan::graphics::all_sprite_sheets().find(previous_sheet_id);
    if (prev_found == fan::graphics::all_sprite_sheets().end()) {
      fan::throw_error("Sprite sheet nr expired (bug)");
    }
    auto& previous_sheet = prev_found->second;
    {
      auto found = fan::graphics::ss_lookup().find(std::make_pair(shape_sprite_sheet_id, previous_sheet.name));
      if (found != fan::graphics::ss_lookup().end()) {
        fan::graphics::ss_lookup().erase(found);
      }
    }
    previous_sheet.name = new_name;
    fan::graphics::ss_lookup()[std::make_pair(shape_sprite_sheet_id, new_name)] = previous_sheet_id;
  }

  sprite_sheet_id_t add_shape_sprite_sheet(sprite_sheet_id_t new_sprite_sheet) {
    shape_sprite_sheets()[shape_ss_counter()].emplace_back(new_sprite_sheet);
    return shape_ss_counter()++;
  }

  sprite_sheet_id_t add_existing_sprite_sheet_shape(sprite_sheet_id_t existing_sprite_sheet, sprite_sheet_id_t shape_sprite_sheet_id, const sprite_sheet_t& new_sprite_sheet) {
    sprite_sheet_id_t new_sheet_id = existing_sprite_sheet;
    all_sprite_sheets()[existing_sprite_sheet] = new_sprite_sheet;
    if (!shape_sprite_sheet_id) {
      shape_sprite_sheet_id = add_shape_sprite_sheet(new_sheet_id);
      ss_lookup()[std::make_pair(shape_sprite_sheet_id, new_sprite_sheet.name)] = new_sheet_id;
      return shape_sprite_sheet_id;
    }
    ss_lookup()[std::make_pair(shape_sprite_sheet_id, new_sprite_sheet.name)] = new_sheet_id;
    auto found = shape_sprite_sheets().find(shape_sprite_sheet_id);
    if (found == shape_sprite_sheets().end()) {
      fan::throw_error("add_shape_sprite_sheet:given shape sprite_sheet id not found");
    }
    found->second.emplace_back(new_sheet_id);
    return shape_sprite_sheet_id;
  }

  sprite_sheet_id_t add_shape_sprite_sheet(sprite_sheet_id_t shape_sprite_sheet_id, const sprite_sheet_t& new_sprite_sheet) {
    sprite_sheet_id_t new_sheet_id = ss_counter()++;
    return add_existing_sprite_sheet_shape(new_sheet_id, shape_sprite_sheet_id, new_sprite_sheet);
  }

  bool is_sprite_sheet_finished(sprite_sheet_id_t nr, const fan::graphics::sprite_sheet_data_t& sd) {
    auto& sprite_sheet = fan::graphics::get_sprite_sheet(nr);
    return sd.current_frame == sprite_sheet.selected_frames.size() - 1;
  }

  fan::graphics::sprite_sheet_t create_sprite_sheet(
    const std::string& name,
    const std::string& image_path,
    int hframes,
    int vframes,
    int fps,
    bool loop,
    uint32_t filter,
    const std::vector<int>& frames,
    const std::source_location& callers_path
  ) {
    fan::graphics::sprite_sheet_t sheet;
    sheet.name = name;
    sheet.fps = fps;
    sheet.loop = loop;

    fan::graphics::sprite_sheet_t::image_t img;
    fan::graphics::image_load_properties_t props;
    props.min_filter = filter;
    props.mag_filter = filter;
    img.image = fan::graphics::ctx()->image_load_path_props(fan::graphics::ctx(), image_path, props, callers_path);
    img.hframes = hframes;
    img.vframes = vframes;
    sheet.images.push_back(img);

    if (frames.empty()) {
      int total_frames = hframes * vframes;
      for (int i = 0; i < total_frames; ++i) {
        sheet.selected_frames.push_back(i);
      }
    }
    else {
      sheet.selected_frames = frames;
    }

    return sheet;
  }

  fan::graphics::sprite_sheet_t create_sprite_sheet(
    const std::string& name,
    fan::graphics::image_t image,
    int hframes,
    int vframes,
    int fps,
    bool loop,
    const std::vector<int>& frames
  ) {
    fan::graphics::sprite_sheet_t sheet;
    sheet.name = name;
    sheet.fps = fps;
    sheet.loop = loop;

    fan::graphics::sprite_sheet_t::image_t img;
    img.image = image;
    img.hframes = hframes;
    img.vframes = vframes;
    sheet.images.push_back(img);

    if (frames.empty()) {
      int total_frames = hframes * vframes;
      for (int i = 0; i < total_frames; ++i) {
        sheet.selected_frames.push_back(i);
      }
    }
    else {
      sheet.selected_frames = frames;
    }

    return sheet;
  }

#if defined(FAN_JSON)
  fan::json sprite_sheet_serialize() {
    fan::json result = fan::json::object();
    fan::json animations_arr = fan::json::array();

    for (auto& sheet : all_sprite_sheets()) {
      fan::json ss;
      ss["name"] = sheet.second.name;
      ss["selected_frames"] = sheet.second.selected_frames;
      ss["fps"] = sheet.second.fps;
      ss["id"] = sheet.first.id;

      if (!sheet.second.images.empty()) {
        fan::json images_arr = fan::json::array();
        for (const auto& img : sheet.second.images) {
          images_arr.push_back(img);
        }
        ss["images"] = images_arr;
      }

      animations_arr.push_back(ss);
    }

    if (!animations_arr.empty()) {
      result["animations"] = animations_arr;
    }

    return result;
  }

  void sprite_sheets_parse(std::string_view json_path, fan::json& json, const std::source_location& callers_path) {
    auto current_global_id = ss_counter().id;
    if (json.contains("animations")) {
      for (const auto& item : json["animations"]) {
        sprite_sheet_t sheet;
        sheet.name = item.value("name", std::string{});
        if (item.contains("selected_frames") && item["selected_frames"].is_array()) {
          sheet.selected_frames.clear();
          for (const auto& frame_json : item["selected_frames"]) {
            sheet.selected_frames.push_back(frame_json.get<int>());
          }
          if (item.contains("images")) {
            for (const auto& frame_json : item["images"]) {
              sprite_sheet_t::image_t img;
              img.assign(frame_json, callers_path);
              sheet.images.push_back(img);
            }
          }
        }
        else {
          sheet.selected_frames.clear();
        }
        sheet.fps = item.value("fps", 0.0f);

        sprite_sheet_id_t original_id = item.value("id", uint32_t());
        sprite_sheet_id_t new_id = original_id.id + current_global_id;
        auto found = ss_cache().find({original_id, std::filesystem::absolute(json_path).generic_string()});
        if (found == ss_cache().end()) {
          ss_cache()[{original_id, std::filesystem::absolute(json_path).generic_string()}] = new_id;
          ss_counter() = std::max(ss_counter().id, static_cast<uint32_t>(new_id.id + 1));
        }
        else {
          new_id = found->second;
        }
        all_sprite_sheets()[new_id] = sheet;
      }
    }

    // update sprite_sheet id table
    if (json.contains("shapes")) {
      for (auto& shape : json["shapes"]) {
        if (shape.contains("animations")) {
          for (auto& anim_id : shape["animations"]) {
            sprite_sheet_id_t original_id = anim_id.get<uint32_t>();
            sprite_sheet_id_t new_id = original_id.id + current_global_id;

            auto found = ss_cache().find({original_id, std::filesystem::absolute(json_path).generic_string()});
            if (found == ss_cache().end()) {
              ss_cache()[{original_id, std::filesystem::absolute(json_path).generic_string()}] = new_id;
              ss_counter() = std::max(ss_counter().id, static_cast<uint32_t>(new_id.id + 1));
            } else {
              new_id = found->second;
            }
            anim_id = new_id.id;
          }
        }
      }
    }
  }
#endif // FAN_JSON

#endif
}

namespace fan::graphics{
#if defined(FAN_2D)
  void shapes::shaper_deep_copy(shape_t* dst, const shape_t* const src, shaper_t::ShapeTypeIndex_t sti) {
    // alloc can be avoided inside switch
    uint8_t* KeyPack = new uint8_t[shaper.GetKeysSize(*src)];
    shaper.WriteKeys(*src, KeyPack);

    auto _vi = src->GetRenderData(shaper);
    auto vlen = shaper.GetRenderDataSize(sti);
    uint8_t* vi = new uint8_t[vlen];
    std::memcpy(vi, _vi, vlen);

    auto _ri = src->GetData(shaper);
    auto rlen = shaper.GetDataSize(sti);

    uint8_t* ri = new uint8_t[rlen];
    std::memcpy(ri, _ri, rlen);

    *dst = fan::graphics::g_shapes->shaper.add(
      sti,
      KeyPack,
      fan::graphics::g_shapes->shaper.GetKeysSize(*src),
      vi,
      ri
    );

  #if defined(debug_shape_t)
    fan::print("+", dst->NRI);
  #endif

    delete[] KeyPack;
    delete[] vi;
    delete[] ri;
  }



  shapes::shape_t::shape_t() {
    sic();
  }

  shapes::shape_t::shape_t(shape_t&& s) noexcept : shaper_t::ShapeID_t() {
    NRI = s.NRI;
    s.sic();
  }
  shapes::shape_t::shape_t(const shaper_t::ShapeID_t& s) : shape_t() {
    if (s.iic()) {
      return;
    }

    shape_nr_t new_raw;
    auto src_move = fan::graphics::culling::get_movement(g_shapes->visibility, s);

    //{ // vfi
    //  shapes::shape_ids_t::nr_t src_id;
    //  src_id.gint() = s.NRI;
    //  auto& src_sd = g_shapes->shape_ids[src_id];

    //  if (src_sd.shape_type == shape_type_t::vfi) {
    //    auto* src_ri = (vfi_t::ri_t*)src_sd.visual.GetData(g_shapes->shaper);

    //    shapes::vfi_list_t::nr_t src_vfi_nr;
    //    src_vfi_nr.gint() = src_sd.data_nr;
    //    auto props = g_shapes->vfi_list[src_vfi_nr];

    //    vfi_t::common_shape_data_t* new_shape_data = nullptr;
    //    if (src_ri && src_ri->shape_data) {
    //      new_shape_data = new vfi_t::common_shape_data_t(*src_ri->shape_data);
    //    }

    //    auto new_vfi_shape = g_shapes->vfi.push_back(props);

    //    auto* new_ri = (vfi_t::ri_t*)new_vfi_shape.GetData(g_shapes->shaper);
    //    if (new_ri && new_shape_data) {
    //      delete new_ri->shape_data;
    //      new_ri->shape_data = new_shape_data;
    //    }

    //    this->gint() = new_vfi_shape.NRI;

    //    if (!g_shapes->visibility.enabled) {
    //      push_shaper();
    //    }
    //    else {
    //      shaper_t::ShapeID_t new_sid;
    //      new_sid.NRI = new_vfi_shape.NRI;
    //      fan::graphics::culling::add_shape(g_shapes->visibility, new_sid, src_move);
    //    }
    //    return;
    //  }
    //}

    g_shapes->with_shape_list(s.NRI, [&](auto& list, auto src_nr, auto& src_sd) {
      auto props = list[src_nr];

      if constexpr (requires { props.sprite_sheet_data; }) {
        props.sprite_sheet_data.frame_update_nr.sic(); // make new one, dont use shared
      }

      auto new_gid = g_shapes->add_shape(list, props);
      new_raw = new_gid.gint();
    });

    this->gint() = new_raw;
    if (!g_shapes->visibility.enabled) {
      push_shaper();
    }
    else {
      shaper_t::ShapeID_t new_sid;
      new_sid.NRI = new_raw;
      fan::graphics::culling::add_shape(g_shapes->visibility, new_sid, src_move);
    }
  }
  shapes::shape_t::shape_t(const shape_t& s) 
    : shape_t(static_cast<const shaper_t::ShapeID_t&>(s)) {
  }
  shapes::shape_t& shapes::shape_t::operator=(shape_t&& s) noexcept {
    if (this == &s) return *this;

    if (!iic()) remove();

    NRI = s.NRI;
    s.sic();

    return *this;
  }
  shapes::shape_t& shapes::shape_t::operator=(const shape_t& s) {
    if (this == &s) return *this;

    if (!iic()) remove();
    if (s.iic()) return *this;

    shape_nr_t new_raw;
    auto src_move = fan::graphics::culling::get_movement(g_shapes->visibility, s);

    //{ // vfi
    //  shapes::shape_ids_t::nr_t src_id;
    //  src_id.gint() = s.NRI;
    //  auto& src_sd = g_shapes->shape_ids[src_id];

    //  if (src_sd.shape_type == shape_type_t::vfi) {
    //    auto* src_ri = (vfi_t::ri_t*)src_sd.visual.GetData(g_shapes->shaper);

    //    shapes::vfi_list_t::nr_t src_vfi_nr;
    //    src_vfi_nr.gint() = src_sd.data_nr;
    //    auto props = g_shapes->vfi_list[src_vfi_nr];

    //    vfi_t::common_shape_data_t* new_shape_data = nullptr;
    //    if (src_ri && src_ri->shape_data) {
    //      new_shape_data = new vfi_t::common_shape_data_t(*src_ri->shape_data);
    //    }

    //    auto new_vfi_shape = g_shapes->vfi.push_back(props);

    //    auto* new_ri = (vfi_t::ri_t*)new_vfi_shape.GetData(g_shapes->shaper);
    //    if (new_ri && new_shape_data) {
    //      delete new_ri->shape_data;
    //      new_ri->shape_data = new_shape_data;
    //    }

    //    this->gint() = new_vfi_shape.NRI;

    //    if (!g_shapes->visibility.enabled) {
    //      push_shaper();
    //    }
    //    else {
    //      shaper_t::ShapeID_t new_sid;
    //      new_sid.NRI = new_vfi_shape.NRI;
    //      fan::graphics::culling::add_shape(g_shapes->visibility, new_sid, src_move);
    //    }

    //    return *this;
    //  }
    //}

    g_shapes->with_shape_list(s.NRI, [&](auto& list, auto src_nr, auto& src_sd) {
      auto props = list[src_nr];

      if constexpr (requires { props.sprite_sheet_data; }) {
        props.sprite_sheet_data.frame_update_nr.sic(); // make new one, dont use shared
      }

      auto new_gid = g_shapes->add_shape(list, props);
      new_raw = new_gid.gint();
    });

    this->gint() = new_raw;

    if (!g_shapes->visibility.enabled) {
      push_shaper();
    }
    else {
      shaper_t::ShapeID_t new_sid;
      new_sid.NRI = new_raw;
      fan::graphics::culling::add_shape(g_shapes->visibility, new_sid, src_move);
    }

    return *this;
  }
  shapes::shape_t::shape_t(shaper_t::ShapeID_t&& s) noexcept : shaper_t::ShapeID_t() {
    NRI = s.NRI;
    s.sic();
  }

  shapes::shape_t::~shape_t() {
    remove();
  }

  shapes::shape_t::operator bool() const {
    return !iic();
  }

  bool shapes::shape_t::operator==(const shape_t& shape) const {
    return NRI == shape.NRI;
  }

  void shapes::shape_t::remove() {
    if (iic()) {
      return;
    }

    shapes::shape_ids_t::nr_t id;
    id.gint() = NRI;
    auto& sd = g_shapes->shape_ids[id];
    if(sd.shape_type == shape_type_t::vfi) {
      g_shapes->vfi.erase(get_visual_id());
      sic();
      return;
    }
    if (sd.shape_type == shape_type_t::sprite) {
      stop_sprite_sheet();
    }

    fan::graphics::culling::remove_shape(g_shapes->visibility, get_id());


    if (get_visual_id().iic() == false) {
      erase_shaper();
    }

    g_shapes->remove_shape(NRI);
    sic();
  }

  void shapes::shape_t::erase() {
    remove();
  }

  shaper_t::ShapeID_t shapes::shape_t::get_id() const {
    return static_cast<const shaper_t::ShapeID_t&>(*this);
  }

  bool shapes::shape_t::is_visible() const {
    return g_shapes->shape_functions[get_shape_type()].get_visible(this);
  }

  void shapes::shape_t::set_visible(bool flag) {
    g_shapes->shape_functions[get_shape_type()].set_visible(this, flag);
  }
  void shapes::shape_t::set_static(bool update) {
    auto& c = g_shapes->visibility;
    if (get_movement() == culling::movement_static && !update) return;
    culling::remove_shape(c, *this);
    culling::add_shape(c, *this, culling::movement_static);
  }
  void shapes::shape_t::set_dynamic() {
    auto& c = g_shapes->visibility;
    if (get_movement() == culling::movement_dynamic) return;
    culling::remove_shape(c, *this);
    culling::add_shape(c, *this, culling::movement_dynamic);
  }
  void shapes::shape_t::remove_culling() {
    auto& c = g_shapes->visibility;
    culling::remove_shape(c, *this);
  }
  fan::graphics::culling::movement_type_t shapes::shape_t::get_movement() const {
    return fan::graphics::culling::get_movement(g_shapes->visibility, *this);
  }
  void shapes::shape_t::update_dynamic() {
    if (get_movement() == fan::graphics::culling::movement_dynamic) {
      culling::update_dynamic(g_shapes->visibility, *this);
    }
  }
  void shapes::shape_t::update_culling() {
    auto& c = g_shapes->visibility;
    if (c.registry.id_to_movement.find(*this) == c.registry.id_to_movement.end()) {
      return;
    }
    if (get_movement() == culling::movement_static) {
      set_static(true);
    }
    else {
      update_dynamic();
    }
  }

  void shapes::shape_t::push_shaper() {
    shapes::shape_ids_t::nr_t id;
    id.gint() = NRI;
    auto& sd = g_shapes->shape_ids[id];

    if (!sd.visual.iic()) {
      return;
    }

    switch (sd.shape_type) {
    case shape_type_t::rectangle: {
      shapes::rectangle_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->rectangle_list[nr];

      shapes::rectangle_t::vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.color = properties.color;
      vi.outline_color = properties.outline_color;
      vi.angle = properties.angle;
      vi.rotation_point = properties.rotation_point;

      shapes::rectangle_t::ri_t ri;
      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::rectangle, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::rectangle,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::sprite: {
      shapes::sprite_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->sprite_list[nr];

      shapes::sprite_t::vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;
      vi.tc_position = properties.tc_position;
      vi.tc_size = properties.tc_size;
      vi.parallax_factor = properties.parallax_factor;
      vi.seed = properties.seed;

      shapes::sprite_t::ri_t ri;
      ri.images = properties.images;
      ri.texture_pack_unique_id = properties.texture_pack_unique_id;
      ri.sprite_sheet_data = properties.sprite_sheet_data;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::sprite, vi, ri,
        Key_e::visible, (visible_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::shader, (fan::graphics::shader_raw_t)fan::graphics::g_shapes->shaper.GetShader(shape_type_t::sprite).gint(),
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.image,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::sprite,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );

      if (properties.sprite_sheet_data.start_sprite_sheet) {
        play_sprite_sheet();
      }
      break;
    }
    case shape_type_t::line: {
      shapes::line_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->line_list[nr];

      shapes::line_t::vi_t vi;
      vi.src = properties.src;
      vi.dst = properties.dst;
      vi.color = properties.color;
      vi.thickness = properties.thickness;

      shapes::line_t::ri_t ri;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::line, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.src.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::line,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::circle: {
      shapes::circle_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->circle_list[nr];

      shapes::circle_t::vi_t vi;
      vi.position = properties.position;
      vi.radius = properties.radius;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;

      shapes::circle_t::ri_t ri;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::circle, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::circle,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::light: {
      shapes::light_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->light_list[nr];

      shapes::light_t::vi_t vi;
      vi.position = properties.position;
      vi.parallax_factor = properties.parallax_factor;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.flags = properties.flags;
      vi.angle = properties.angle;
      shapes::light_t::ri_t ri;

      sd.visual = shape_add(sd.shape_type, vi, ri,
        Key_e::light, (uint8_t)0,
        Key_e::visible, (uint8_t)true,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)sd.shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::unlit_sprite: {
      shapes::unlit_sprite_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->unlit_sprite_list[nr];

      shapes::unlit_sprite_t::vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;
      vi.tc_position = properties.tc_position;
      vi.tc_size = properties.tc_size;
      vi.parallax_factor = properties.parallax_factor;
      vi.seed = properties.seed;

      shapes::unlit_sprite_t::ri_t ri;
      ri.images = properties.images;
      ri.texture_pack_unique_id = properties.texture_pack_unique_id;
      ri.sprite_sheet_data = properties.sprite_sheet_data;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::unlit_sprite, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::shader, (fan::graphics::shader_raw_t)fan::graphics::g_shapes->shaper.GetShader(shape_type_t::unlit_sprite).gint(),
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.image,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::unlit_sprite,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::text: {
      shapes::text_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->text_list[nr];

      sd.visual = g_shapes->shaper.add(shape_type_t::text, nullptr, 0, nullptr, nullptr);
      break;
    }
    case shape_type_t::capsule: {
      shapes::capsule_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->capsule_list[nr];

      shapes::capsule_t::vi_t vi;
      vi.position = properties.position;
      vi.center0 = properties.center0;
      vi.center1 = properties.center1;
      vi.radius = properties.radius;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.outline_color = properties.outline_color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;

      shapes::capsule_t::ri_t ri;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::capsule, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::capsule,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::polygon: {
      shapes::polygon_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->polygon_list[nr];

      if (properties.vertices.empty()) {
        fan::throw_error("invalid vertices");
      }

      std::vector<polygon_vertex_t> polygon_vertices(properties.vertices.size());
      for (std::size_t i = 0; i < properties.vertices.size(); ++i) {
        polygon_vertices[i].position = properties.vertices[i].position;
        polygon_vertices[i].color = properties.vertices[i].color;
        polygon_vertices[i].offset = properties.position;
        polygon_vertices[i].angle = properties.angle;
        polygon_vertices[i].rotation_point = properties.rotation_point;
      }

      shapes::polygon_t::vi_t vi;
      shapes::polygon_t::ri_t ri;
      ri.buffer_size = sizeof(decltype(polygon_vertices)::value_type) * polygon_vertices.size();
      ri.vao.open((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))));
      ri.vao.bind((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))));
      ri.vbo.open((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))), GL_ARRAY_BUFFER);
      fan::opengl::core::write_glbuffer(
        (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))),
        ri.vbo.m_buffer,
        polygon_vertices.data(),
        ri.buffer_size,
        GL_STATIC_DRAW,
        ri.vbo.m_target
      );

      auto& shape_data = g_shapes->shaper.GetShapeTypes(shape_type_t::polygon).renderer.gl;
      fan::graphics::context_shader_t shader;
      if (!shape_data.shader.iic()) {
        shader = fan::graphics::shader_get(shape_data.shader);
      }
      uint64_t ptr_offset = 0;
      for (shape_gl_init_t& location : g_shapes->polygon.get_locations()) {
        if (((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))).opengl.major == 2 &&
          (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))).opengl.minor == 1) &&
          !shape_data.shader.iic()) {
          location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
        }
        fan_opengl_call(glEnableVertexAttribArray(location.index.first));
        switch (location.type) {
        case GL_UNSIGNED_INT:
        case GL_INT:
          fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
          break;
        default:
          fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
        }
        if (((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))).opengl.major > 3) ||
          ((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))).opengl.major == 3 &&
            (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))).opengl.minor >= 3)) {
          if (shape_data.instanced) {
            fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
          }
        }
        switch (location.type) {
        case GL_FLOAT: ptr_offset += location.size * sizeof(GLfloat); break;
        case GL_UNSIGNED_INT: ptr_offset += location.size * sizeof(GLuint); break;
        default: fan::throw_error_impl();
        }
      }

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::polygon, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::polygon,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, (uint32_t)properties.vertices.size()
      );
      break;
    }
    case shape_type_t::grid: {
      shapes::grid_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->grid_list[nr];

      shapes::grid_t::vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.grid_size = properties.grid_size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;

      shapes::grid_t::ri_t ri;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::grid, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::grid,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::particles: {
      shapes::particles_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->particles_list[nr];

      shapes::particles_t::vi_t vi;
      shapes::particles_t::ri_t ri;
      ri.loop = properties.loop;
      ri.loop_enabled_time = properties.loop_enabled_time;
      ri.loop_disabled_time = properties.loop_disabled_time;
      ri.position = properties.position;

      ri.start_size = properties.start_size;
      ri.end_size = properties.end_size;

      ri.begin_color = properties.begin_color;
      ri.end_color = properties.end_color;

      ri.begin_time = properties.begin_time;
      ri.alive_time = properties.alive_time;
      ri.respawn_time = properties.respawn_time;
      ri.count = properties.count;

      ri.start_velocity = properties.start_velocity;
      ri.end_velocity = properties.end_velocity;

      ri.start_angle_velocity = properties.start_angle_velocity;
      ri.end_angle_velocity = properties.end_angle_velocity;

      ri.begin_angle = properties.begin_angle;
      ri.end_angle = properties.end_angle;
      ri.angle = properties.angle;

      ri.spawn_spacing = properties.spawn_spacing;
      ri.expansion_power = properties.expansion_power;

      ri.start_spread = properties.start_spread;
      ri.end_spread = properties.end_spread;

      ri.jitter_start = properties.jitter_start;
      ri.jitter_end = properties.jitter_end;
      ri.jitter_speed = properties.jitter_speed;

      ri.size_random_range = properties.size_random_range;
      ri.color_random_range = properties.color_random_range;
      ri.angle_random_range = properties.angle_random_range;

      ri.shape = properties.shape;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::particles, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::shader, (fan::graphics::shader_raw_t)fan::graphics::g_shapes->shaper.GetShader(shape_type_t::particles).gint(),
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.image,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::particles,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::universal_image_renderer: {
      shapes::universal_image_renderer_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->universal_image_renderer_list[nr];

      shapes::universal_image_renderer_t::vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.tc_position = properties.tc_position;
      vi.tc_size = properties.tc_size;

      shapes::universal_image_renderer_t::ri_t ri;
      std::copy(&properties.images[1], &properties.images[0] + properties.images.size(), ri.images_rest.data());

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::universal_image_renderer, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::shader, (fan::graphics::shader_raw_t)fan::graphics::g_shapes->shaper.GetShader(shape_type_t::universal_image_renderer).gint(),
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.images[0],
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::universal_image_renderer,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::gradient: {
      shapes::gradient_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->gradient_list[nr];

      shapes::gradient_t::vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.rotation_point = properties.rotation_point;

      shapes::gradient_t::ri_t ri;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::gradient, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::gradient,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::shadow: {
      shapes::shadow_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->shadow_list[nr];

      shapes::shadow_t::vi_t vi;
      vi.position = properties.position;
      vi.shape = properties.shape;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.flags = properties.flags;
      vi.angle = properties.angle;
      vi.light_position = properties.light_position;
      vi.light_radius = properties.light_radius;

      shapes::shadow_t::ri_t ri;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::shadow, vi, ri,
        Key_e::shadow, (uint8_t)0,
        Key_e::visible, (uint8_t)true,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::shadow,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::shader_shape: {
      shapes::shader_shape_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->shader_shape_list[nr];

      shapes::shader_shape_t::vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.rotation_point = properties.rotation_point;
      vi.color = properties.color;
      vi.angle = properties.angle;
      vi.flags = properties.flags;
      vi.tc_position = properties.tc_position;
      vi.tc_size = properties.tc_size;
      vi.parallax_factor = properties.parallax_factor;
      vi.seed = properties.seed;

      shapes::shader_shape_t::ri_t ri;
      ri.images = properties.images;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::shader_shape, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::shader, (fan::graphics::shader_raw_t)properties.shader.gint(),
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::image, properties.image,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::shader_shape,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      g_shapes->shaper.GetShader(shape_type_t::shader_shape) = properties.shader;
      break;
    }
    #if defined(FAN_3D)
    case shape_type_t::rectangle3d: {
      shapes::rectangle3d_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->rectangle3d_list[nr];

      shapes::rectangle3d_t::vi_t vi;
      vi.position = properties.position;
      vi.size = properties.size;
      vi.color = properties.color;

      shapes::rectangle3d_t::ri_t ri;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::rectangle3d, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::rectangle3d,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    case shape_type_t::line3d: {
      shapes::line3d_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      auto& properties = g_shapes->line3d_list[nr];

      shapes::line3d_t::vi_t vi;
      vi.src = properties.src;
      vi.dst = properties.dst;
      vi.color = properties.color;

      shapes::line3d_t::ri_t ri;

      sd.visual = shape_add(
        (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::line3d, vi, ri,
        Key_e::visible, (uint8_t)true,
        Key_e::depth, (uint16_t)properties.src.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, (fan::graphics::shaper_t::KeyTypeIndex_t)shape_type_t::line3d,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
      break;
    }
    #endif
    default:
      fan::print("unsupported shape type in push_shaper");
    }
  }

  void shapes::shape_t::erase_shaper() {
    if (get_shape_type() == shape_type_t::polygon) {
      auto& ri = get_shape_rdata<shapes::polygon_t>();
      ri.vao.close((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))));
      ri.vbo.close((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()))));
    }


    shapes::shape_ids_t::nr_t id;
    id.gint() = NRI;

    auto& sd = g_shapes->shape_ids[id];
    if (sd.visual.iic()) {
      return;
    }

    g_shapes->shaper.remove(sd.visual);
    sd.visual.sic();
  }

  fan::graphics::shaper_t::ShapeID_t& shapes::shape_t::get_visual_id() const {
    shapes::shape_ids_t::nr_t id;
    id.gint() = NRI;

    auto& sd = g_shapes->shape_ids[id];
    return sd.visual;
  }

  shapes::shape_t* shapes::shape_t::get_visual_shape() const {
    return (fan::graphics::shapes::shape_t*)&get_visual_id();
  }

  // many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t
  uint16_t shapes::shape_t::get_shape_type() const {
    auto& vs = get_visual_id();
    if (vs) {
      return g_shapes->shaper.ShapeList[vs].sti;
    }
    shapes::shape_ids_t::nr_t id;
    id.gint() = NRI;

    auto& sd = g_shapes->shape_ids[id];
    return sd.shape_type;
  }

  void set_particle_pos(shapes::shape_t* shape, fan::vec3 position) {
    if (shape->get_shape_type() == shape_type_t::particles) {
      g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
        if constexpr (requires { props.position; }) {
          props.position = position;
        }
      });

      if (shape->get_visual_id()) {
        auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)
          shape->GetData(fan::graphics::g_shapes->shaper);
        ri.position = position;
      }
    }
  }

  void shapes::shape_t::set_position(const fan::vec2& position) {
    if (fan::vec2(get_position()) == position) {
      return;
    }
    g_shapes->shape_functions[get_shape_type()].set_position2(this, position);
    set_particle_pos(this, position);
    update_culling();
  }

  void shapes::shape_t::set_position(const fan::vec3& position) {
    if (get_position() == position) {
      return;
    }
    g_shapes->shape_functions[get_shape_type()].set_position3(this, position);
    set_particle_pos(this, position);
    update_culling();
  }

  void shapes::shape_t::set_x(f32_t x) {
    set_position(fan::vec2(x, get_position().y));
    update_culling();
  }

  void shapes::shape_t::set_y(f32_t y) {
    set_position(fan::vec2(get_position().x, y));
    update_culling();
  }

  void shapes::shape_t::set_z(f32_t z) {
    set_position(fan::vec3(get_position().x, get_position().y, z));
  }

  fan::vec3 shapes::shape_t::get_position() const {
    auto shape_type = get_shape_type();
    return g_shapes->shape_functions[shape_type].get_position(this);
  }

  f32_t shapes::shape_t::get_x() const {
    return get_position().x;
  }

  f32_t shapes::shape_t::get_y() const {
    return get_position().y;
  }

  f32_t shapes::shape_t::get_z() const {
    return get_position().z;
  }

  void shapes::shape_t::set_size(const fan::vec2& size) {
    if (get_size() == size) {
      return;
    }
    g_shapes->shape_functions[get_shape_type()].set_size(this, size);
    update_culling();
  }
  void shapes::shape_t::set_radius(f32_t radius) {
    g_shapes->shape_functions[get_shape_type()].set_size(this, radius);
    update_culling();
  }
  void shapes::shape_t::set_size3(const fan::vec3& size) {
    g_shapes->shape_functions[get_shape_type()].set_size3(this, size);
    update_culling();
  }

  // returns half extents of draw
  fan::vec2 shapes::shape_t::get_size() const {
    return fan::graphics::g_shapes->shape_functions[get_shape_type()].get_size(this);
  }

  fan::vec3 shapes::shape_t::get_size3() {
    return fan::graphics::g_shapes->shape_functions[get_shape_type()].get_size3(this);
  }

  void shapes::shape_t::set_rotation_point(const fan::vec2& rotation_point) {
    fan::graphics::g_shapes->shape_functions[get_shape_type()].set_rotation_point(this, rotation_point);
  }

  fan::vec2 shapes::shape_t::get_rotation_point() const {
    return fan::graphics::g_shapes->shape_functions[get_shape_type()].get_rotation_point(this);
  }

  void shapes::shape_t::set_color(const fan::color& color) {
    g_shapes->shape_functions[get_shape_type()].set_color(this, color);
  }

  fan::color shapes::shape_t::get_color() const {
    return g_shapes->shape_functions[get_shape_type()].get_color(this);
  }

  std::array<fan::color, 4> shapes::shape_t::get_colors() const {
    return g_shapes->shape_functions[get_shape_type()].get_colors(this);
  }
  void shapes::shape_t::set_colors(const std::array<fan::color, 4>& colors) {
    return g_shapes->shape_functions[get_shape_type()].set_colors(this, colors);
  }

  void shapes::shape_t::set_angle(const fan::vec3& angle) {
    g_shapes->shape_functions[get_shape_type()].set_angle(this, angle);
  }

  fan::vec3 shapes::shape_t::get_angle() const {
    return g_shapes->shape_functions[get_shape_type()].get_angle(this);
  }

  fan::basis shapes::shape_t::get_basis() const {
    auto zangle = get_angle().z;
    auto c = std::cos(zangle);
    auto s = std::sin(zangle);

    return fan::basis{
      .right = { c, s, 0 },
      .forward = { s, -c, 0 },
      .up = { 0, 0, 1 }
    };
  }

  fan::vec3 shapes::shape_t::get_forward() const {
    return get_basis().forward;
  }

  fan::vec3 shapes::shape_t::get_right() const {
    return get_basis().right;
  }

  fan::vec3 shapes::shape_t::get_up() const {
    return get_basis().up;
  }

  fan::mat3 shapes::shape_t::get_rotation_matrix() const {
    return get_basis();
  }

  fan::vec3 shapes::shape_t::transform(const fan::vec3& local) const {
    // sign conflict, when forward y is -1, then moving y by -1 would move it down when we want it up
    // so flip the y sign since coordinate system is +y down
    fan::vec3 flipped_y{ local.x, -local.y, local.z };
    return get_position() + get_basis() * flipped_y;
  }

  fan::mat4 shapes::shape_t::get_transform() const {
    fan::mat4 m = fan::mat4::identity();
    m = m.translate(get_position());
    m = m * get_rotation_matrix();
    m = m.scale(get_size());
    return m;
  }

  fan::physics::aabb_t shapes::shape_t::get_aabb() const {

    switch (get_shape_type()) {
    case shape_type_t::circle:
      return { get_position() - get_radius(), get_position() + get_radius()};
    case shape_type_t::capsule:
      return { get_position() - get_radius(), get_position() + get_radius() };
    }

    fan::vec2 pos = get_position();
    fan::vec2 he = get_size(); // half extents
    f32_t cs = std::cos(get_angle().z);
    f32_t sn = std::sin(get_angle().z);
    fan::vec2 pivot = get_rotation_point();
    static constexpr auto flt_max = std::numeric_limits<f32_t>::max();
    fan::vec2 minp(flt_max, flt_max), maxp(-flt_max, -flt_max);

    for (int i = -1; i <= 1; i += 2) {
      for (int j = -1; j <= 1; j += 2) {
        fan::vec2 r = { i * he.x - pivot.x, j * he.y - pivot.y };
        r = { r.x * cs - r.y * sn, r.x * sn + r.y * cs };
        r += pos + pivot;
        minp.x = std::min(minp.x, r.x);
        minp.y = std::min(minp.y, r.y);
        maxp.x = std::max(maxp.x, r.x);
        maxp.y = std::max(maxp.y, r.y);
      }
    }

    return { minp, maxp };
  }

  fan::vec2 shapes::shape_t::get_tc_position() const {
    return g_shapes->shape_functions[get_shape_type()].get_tc_position(this);
  }

  void shapes::shape_t::set_tc_position(const fan::vec2& tc_position) {
    auto st = get_shape_type();
    g_shapes->shape_functions[st].set_tc_position(this, tc_position);
  }

  fan::vec2 shapes::shape_t::get_tc_size() const {
    return g_shapes->shape_functions[get_shape_type()].get_tc_size(this);
  }

  void shapes::shape_t::set_tc_size(const fan::vec2& tc_size) {
    g_shapes->shape_functions[get_shape_type()].set_tc_size(this, tc_size);
  }
  fan::vec2 shapes::shape_t::get_image_sign() const {
    return get_tc_size().sign();
  }
  void shapes::shape_t::set_image_sign(const fan::vec2& sign) {
    fan::vec2 desired_sign = {
      (f32_t)fan::math::sgn(sign.x),
      (f32_t)fan::math::sgn(sign.y)
    };

    fan::vec2 tc = get_tc_size();
    fan::vec2 current_sign = tc.sign();

    bool did_change = current_sign != desired_sign;

    set_tc_size({
      std::abs(tc.x) * desired_sign.x,
      std::abs(tc.y) * desired_sign.y
      });

    if (did_change) {
      g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
        if constexpr (requires { props.sprite_sheet_data; }) {
          props.sprite_sheet_data.previous_frame = props.sprite_sheet_data.current_frame;
          props.sprite_sheet_data.current_frame = 0;
          props.sprite_sheet_data.last_sign = {
            (int8_t)desired_sign.x,
            (int8_t)desired_sign.y
          };
          if (get_visual_id()) {
            set_sprite_sheet_next_frame(0);
          }
        }
      });
    }
  }

  bool shapes::shape_t::load_tp(fan::graphics::texture_pack::ti_t* ti) {
    auto st = get_shape_type();

    g_shapes->visit_shape_draw_data(NRI, [&](auto& properties) {
      if constexpr (requires { properties.texture_pack_unique_id; }) {
        properties.texture_pack_unique_id = ti->unique_id;
      }
    });

    if (!get_visual_id()) {
      return false;
    }

    if (st == fan::graphics::shapes::shape_type_t::sprite ||
      st == fan::graphics::shapes::shape_type_t::unlit_sprite) {
      auto image = ti->image;
      set_image(image);
      set_tc_position(ti->position / image.get_size());
      set_tc_size(ti->size / image.get_size());
      if (st == fan::graphics::shapes::shape_type_t::sprite) {
        sprite_t::ri_t* ram_data = (sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
        ram_data->texture_pack_unique_id = ti->unique_id;
      }
      else if (st == fan::graphics::shapes::shape_type_t::unlit_sprite) {
        unlit_sprite_t::ri_t* ram_data = (unlit_sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
        ram_data->texture_pack_unique_id = ti->unique_id;
      }
      return false;
    }
    fan::throw_error("invalid function call for current shape:"_str + shape_names[st]);
    return true;
  }

  fan::graphics::texture_pack::ti_t shapes::shape_t::get_tp() const {
    fan::graphics::texture_pack::ti_t ti;
    ti.unique_id = get_tp_unique();
    ti.image = get_image();
    ti.position = get_tc_position() * ti.image.get_size();
    ti.size = get_tc_size() * ti.image.get_size();
    return ti;
    //return g_shapes->shape_functions[g_shapes->shaper.GetSTI(*this)].get_tp(this);
  }

  bool shapes::shape_t::set_tp(fan::graphics::texture_pack::ti_t* ti) {
    return load_tp(ti);
  }

  fan::graphics::texture_pack::unique_t shapes::shape_t::get_tp_unique() const {
    fan::graphics::texture_pack::unique_t unique;
    g_shapes->visit_shape_draw_data(NRI, [&](auto& properties) {
      if constexpr (requires { properties.texture_pack_unique_id; }) {
        unique = properties.texture_pack_unique_id;
      }
    });
    return unique;
  }

  fan::graphics::camera_t shapes::shape_t::get_camera() const {
    return g_shapes->shape_functions[get_shape_type()].get_camera(this);
  }

  void shapes::shape_t::set_camera(fan::graphics::camera_t camera) {
    g_shapes->shape_functions[get_shape_type()].set_camera(this, camera);
  }

  fan::graphics::viewport_t shapes::shape_t::get_viewport() const {
    return g_shapes->shape_functions[get_shape_type()].get_viewport(this);
  }

  void shapes::shape_t::set_viewport(fan::graphics::viewport_t viewport) {
    g_shapes->shape_functions[get_shape_type()].set_viewport(this, viewport);
  }

  render_view_t shapes::shape_t::get_render_view() const {
    render_view_t r;
    r.camera = get_camera();
    r.viewport = get_viewport();
    return r;
  }

  void shapes::shape_t::set_render_view(const fan::graphics::render_view_t& render_view) {
    set_camera(render_view.camera);
    set_viewport(render_view.viewport);
  }

  fan::vec2 shapes::shape_t::get_grid_size() {
    return g_shapes->shape_functions[get_shape_type()].get_grid_size(this);
  }

  void shapes::shape_t::set_grid_size(const fan::vec2& grid_size) {
    g_shapes->shape_functions[get_shape_type()].set_grid_size(this, grid_size);
  }

  fan::graphics::image_t shapes::shape_t::get_image() const {
    auto st = get_shape_type();
    if (get_shape_type() == fan::graphics::shape_type_t::universal_image_renderer) {
      if (get_visual_id()) {
        return g_shapes->shape_functions[st].get_image(this);
      }
      shapes::shape_ids_t::nr_t id;
      id.gint() = NRI;
      auto& sd = g_shapes->shape_ids[id];
      shapes::universal_image_renderer_list_t::nr_t nr;
      nr.gint() = sd.data_nr;
      return g_shapes->universal_image_renderer_list[nr].images[0];
    }
    if (g_shapes->shape_functions[st].get_image) {
      return g_shapes->shape_functions[st].get_image(this);
    }
    return fan::graphics::ctx().default_texture;
  }

  void shapes::shape_t::set_image(fan::graphics::image_t image) {
    if (get_image() == image) {
      return;
    }
    g_shapes->shape_functions[get_shape_type()].set_image(this, image);
  }

  fan::graphics::image_data_t& shapes::shape_t::get_image_data() {
    return (*fan::graphics::ctx().image_list)[get_image()];
  }

  std::array<fan::graphics::image_t, 30> shapes::shape_t::get_images() const {
    auto shape_type = get_shape_type();
    if (shape_type == shape_type_t::sprite) {
      return ((sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper))->images;
    }
    else if (shape_type == shape_type_t::unlit_sprite) {
      return ((unlit_sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper))->images;
    }
    else if (shape_type == shape_type_t::universal_image_renderer) {
      shapes::shape_ids_t::nr_t id;
      id.gint() = NRI;
      auto& sd = g_shapes->shape_ids[id];
      shapes::universal_image_renderer_list_t::nr_t nr;
      nr.gint() = sd.data_nr;

      auto& imgs = g_shapes->universal_image_renderer_list[nr].images;
      std::array<fan::graphics::image_t, 30> ret;
      std::copy(imgs.begin(), imgs.end(), ret.data());
      return ret;
    }
  #if FAN_DEBUG >= fan_debug_medium
    fan::throw_error("only for sprite and unlit_sprite");
  #endif
    return {};
  }

  void shapes::shape_t::set_images(const std::array<fan::graphics::image_t, 30>& images) {
    auto shape_type = get_shape_type();
    g_shapes->visit_shape_draw_data(NRI, [&](auto& properties) {
      if constexpr (requires { 
        properties.images; 
        requires std::tuple_size_v<std::remove_reference_t<decltype(properties.images)>> == images.size();
      }) {
        properties.images = images;
      }
    });
    if (shape_type == shape_type_t::sprite) {
      ((sprite_t::ri_t*)ShapeID_t::GetData(fan::graphics::g_shapes->shaper))->images = images;
    }
    else if (shape_type == shape_type_t::unlit_sprite) {
      ((unlit_sprite_t::ri_t*)ShapeID_t::GetData(fan::graphics::g_shapes->shaper))->images = images;
    }
  #if FAN_DEBUG >= fan_debug_medium
    else {
      fan::throw_error("only for sprite and unlit_sprite");
    }
  #endif
  }

  f32_t shapes::shape_t::get_parallax_factor() const {
    return g_shapes->shape_functions[get_shape_type()].get_parallax_factor(this);
  }

  void shapes::shape_t::set_parallax_factor(f32_t parallax_factor) {
    g_shapes->shape_functions[get_shape_type()].set_parallax_factor(this, parallax_factor);
  }

  uint32_t shapes::shape_t::get_flags() const {
    auto f = g_shapes->shape_functions[get_shape_type()].get_flags;
    if (f) {
      return f(this);
    }
    return 0;
  }

  void shapes::shape_t::set_flags(uint32_t flag) {
    auto st = get_shape_type();
    return g_shapes->shape_functions[st].set_flags(this, flag);
  }

  f32_t shapes::shape_t::get_radius() const {
    return g_shapes->shape_functions[get_shape_type()].get_radius(this);
  }

  fan::vec3 shapes::shape_t::get_src() const {
    return g_shapes->shape_functions[get_shape_type()].get_src(this);
  }

  fan::vec2 shapes::shape_t::get_dst() const {
    return g_shapes->shape_functions[get_shape_type()].get_dst(this);
  }

  f32_t shapes::shape_t::get_outline_size() const {
    return g_shapes->shape_functions[get_shape_type()].get_outline_size(this);
  }

  fan::color shapes::shape_t::get_outline_color() const {
    return g_shapes->shape_functions[get_shape_type()].get_outline_color(this);
  }

  void shapes::shape_t::set_outline_color(const fan::color& color) {
    return g_shapes->shape_functions[get_shape_type()].set_outline_color(this, color);
  }

  void shapes::shape_t::reload(uint8_t format, void** image_data, const fan::vec2& image_size) {
    auto& settings = fan::graphics::ctx()->image_get_settings(fan::graphics::ctx(), get_image());
    uint32_t filter = settings.min_filter;

    shapes::shape_ids_t::nr_t id;
    id.gint() = NRI;
    auto& sd = g_shapes->shape_ids[id];
    shapes::universal_image_renderer_list_t::nr_t nr;
    nr.gint() = sd.data_nr;

    universal_image_renderer_t::properties_t& props = g_shapes->universal_image_renderer_list[nr];
    uint8_t image_count_new = fan::graphics::get_channel_amount(format);
    if (format != props.format) {
      auto sti = get_shape_type();
      fan::graphics::image_t vi_image = get_image();

      auto shader = g_shapes->shaper.GetShader(sti);
      fan::graphics::ctx()->shader_set_vertex(
        fan::graphics::ctx(),
        shader,
        read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
      );
      {
        std::string fs;
        switch (format) {
        case fan::graphics::image_format_e::yuv420p: {
          fs = read_shader("shaders/opengl/2D/objects/yuv420p.fs");
          break;
        }
        case fan::graphics::image_format_e::nv12: {
          fs = read_shader("shaders/opengl/2D/objects/nv12.fs");
          break;
        }
        default: {
          fan::throw_error("unimplemented format");
        }
        }
        fan::graphics::ctx()->shader_set_fragment(fan::graphics::ctx(), shader, fs);
        fan::graphics::ctx()->shader_compile(fan::graphics::ctx(), shader);
      }

      uint8_t image_count_old = fan::graphics::get_channel_amount(props.format);
      if (image_count_new < image_count_old) {
        uint8_t textures_to_remove = image_count_old - image_count_new;
        if (vi_image.iic() || vi_image == fan::graphics::ctx().default_texture) { // uninitialized
          textures_to_remove = 0;
        }
        for (int i = 0; i < textures_to_remove; ++i) {
          int index = image_count_old - i - 1; // not tested
          if (index == 0) {
            fan::graphics::ctx()->image_erase(fan::graphics::ctx(), vi_image);
                
            set_image(fan::graphics::ctx().default_texture);
          }
          else {
            fan::graphics::ctx()->image_erase(fan::graphics::ctx(), props.images[index]);
            props.images[index] = fan::graphics::ctx().default_texture;
          }
        }
      }
      else if (image_count_new > image_count_old) {
        fan::graphics::image_t images[4];
        for (uint32_t i = image_count_old; i < image_count_new; ++i) {
          images[i] = fan::graphics::ctx()->image_create(fan::graphics::ctx());
        }
        set_image(images[0]);
        std::copy(&images[0], &images[0] + props.images.size(), props.images.data());
      }
    }

    auto vi_image = get_image();

    for (uint32_t i = 0; i < image_count_new; ++i) {
      if (i == 0) {
        if (vi_image.iic() || vi_image == fan::graphics::ctx().default_texture) {
          vi_image = fan::graphics::ctx()->image_create(fan::graphics::ctx());
          set_image(vi_image);
        }
      }
      else {
        if (props.images[i].iic() || props.images[i] == fan::graphics::ctx().default_texture) {
          props.images[i] = fan::graphics::ctx()->image_create(fan::graphics::ctx());
        }
      }
    }

    for (uint32_t i = 0; i < image_count_new; i++) {
      fan::image::info_t image_info;
      image_info.data = image_data[i];
      image_info.size = fan::graphics::get_image_sizes(format, image_size)[i];
      auto lp = fan::graphics::get_image_properties<image_load_properties_t>(format)[i];
      lp.min_filter = filter;
      if (filter == fan::graphics::image_filter_e::linear ||
        filter == fan::graphics::image_filter_e::nearest) {
        lp.mag_filter = filter;
      }
      else {
        lp.mag_filter = fan::graphics::image_filter_e::linear;
      }
      if (i == 0) {
            
        fan::graphics::ctx()->image_reload_image_info_props(fan::graphics::ctx(), 
          vi_image,
          image_info,
          lp
        );
      }
      else {
        fan::graphics::ctx()->image_reload_image_info_props(fan::graphics::ctx(), 
          props.images[i],
          image_info,
          lp
        );
      }
    }
    if (get_visual_id()) {
      universal_image_renderer_t::ri_t& ri = *(universal_image_renderer_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
      std::copy(props.images.begin() + 1, props.images.end(), ri.images_rest.data());
    }
    props.format = format;
  }

  void shapes::shape_t::reload(uint8_t format, const fan::vec2& image_size) {
        
    auto& settings = fan::graphics::ctx()->image_get_settings(fan::graphics::ctx(), get_image());
    void* data[4]{};
    reload(format, data, image_size);
  }

  // universal image specific
  void shapes::shape_t::reload(uint8_t format, fan::graphics::image_t images[4]) {
    universal_image_renderer_t::ri_t& ri = *(universal_image_renderer_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
    uint8_t image_count_new = fan::graphics::get_channel_amount(format);
    if (format != ri.format) {
      auto sti = g_shapes->shaper.ShapeList[get_visual_id()].sti;
      uint8_t* key_pack = g_shapes->shaper.GetKeys(*this);
      fan::graphics::image_t vi_image = shaper_get_key_safe(image_t, texture_t, image);


      auto shader = g_shapes->shaper.GetShader(sti);
          
      fan::graphics::ctx()->shader_set_vertex(fan::graphics::ctx(), 
        shader,
        read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
      );
      {
        std::string fs;
        switch (format) {
        case fan::graphics::image_format_e::yuv420p: {
          fs = read_shader("shaders/opengl/2D/objects/yuv420p.fs");
          break;
        }
        case fan::graphics::image_format_e::nv12: {
          fs = read_shader("shaders/opengl/2D/objects/nv12.fs");
          break;
        }
        default: {
          fan::throw_error("unimplemented format");
        }
        }
        fan::graphics::ctx()->shader_set_fragment(fan::graphics::ctx(), shader, fs);
            
        fan::graphics::ctx()->shader_compile(fan::graphics::ctx(), shader);
      }
      set_image(images[0]);
      std::copy(&images[1], &images[0] + ri.images_rest.size(), ri.images_rest.data());
      ri.format = format;
    }
  }

  void shapes::shape_t::set_line(const fan::vec2& src, const fan::vec2& dst) {
    auto st = get_shape_type();
    if (st == fan::graphics::shapes::shape_type_t::line) {
      g_shapes->shape_functions[get_shape_type()].set_line(this, src, dst);
    }
    update_culling();
  #if defined(FAN_3D)
    if (st == fan::graphics::shapes::shape_type_t::line3d) {
      auto data = reinterpret_cast<line3d_t::vi_t*>(GetRenderData(fan::graphics::g_shapes->shaper));
      data->src = fan::vec3(src.x, src.y, 0);
      data->dst = fan::vec3(dst.x, dst.y, 0);
      if (fan::graphics::ctx().window->renderer == fan::window_t::renderer_t::opengl) {
        auto& data = g_shapes->shaper.ShapeList[get_visual_id()];
        g_shapes->shaper.ElementIsPartiallyEdited(
          data.sti,
          data.blid,
          data.ElementIndex,
          fan::member_offset(&line3d_t::vi_t::src),
          sizeof(line3d_t::vi_t::src)
        );
        g_shapes->shaper.ElementIsPartiallyEdited(
          data.sti,
          data.blid,
          data.ElementIndex,
          fan::member_offset(&line3d_t::vi_t::dst),
          sizeof(line3d_t::vi_t::dst)
        );
      }
    }
  #endif
  }

  bool shapes::shape_t::is_mouse_inside() {
    switch (get_shape_type()) {
    case shape_type_t::rectangle: {
      return fan_2d::collision::rectangle::point_inside_no_rotation(
        get_mouse_position(get_camera(), get_viewport()),
        get_position(),
        get_size()
      );
    }
    default: {
      break;
    }
    }
    return false;
  }

#if defined(FAN_PHYSICS_2D)
  bool shapes::shape_t::intersects(const fan::graphics::shapes::shape_t& shape) const {
    switch (get_shape_type()) {
    case shape_type_t::capsule: // inaccurate
    case shape_type_t::shader_shape:
    case shape_type_t::unlit_sprite:
    case shape_type_t::sprite:
    case shape_type_t::rectangle: {
      fan::physics::aabb_t aabb = get_aabb();
      fan::physics::aabb_t aabb2 = shape.get_aabb();
      return fan_2d::collision::rectangle::check_collision(
        aabb.min + (aabb.max - aabb.min) / 2.f,
        (aabb.max - aabb.min) / 2.f,
        aabb2.min + (aabb2.max - aabb2.min) / 2.f,
        (aabb2.max - aabb2.min) / 2.f
      );
    }
    }
    fan::throw_error("todo");
    return true;
  }

  bool shapes::shape_t::collides(const fan::graphics::shapes::shape_t& shape) const {
    return intersects(shape);
  }

  bool shapes::shape_t::point_inside(const fan::vec2& point) const {
    switch (get_shape_type()) {
    case shape_type_t::capsule: // inaccurate
    case shape_type_t::shader_shape:
    case shape_type_t::unlit_sprite:
    case shape_type_t::sprite:
    case shape_type_t::rectangle: {
      fan::physics::aabb_t aabb = get_aabb();
      fan::vec2 size = aabb.max - aabb.min;
      return fan_2d::collision::rectangle::point_inside(
        aabb.min,
        fan::vec2(aabb.min.x + size.x, aabb.min.y),
        aabb.max,
        fan::vec2(aabb.min.x, aabb.min.y + size.y),
        point
      );
    }
    case shape_type_t::circle: {
      return fan_2d::collision::circle::point_inside(point, get_position(), get_radius());
    }
    }
    fan::throw_error("todo");
    return true;
  }

  bool shapes::shape_t::collides(const fan::vec2& point) const {
    return point_inside(point);
  }
#endif

  void shapes::shape_t::add_existing_sprite_sheet(sprite_sheet_id_t nr) {
    if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
      auto& sprite_sheet = fan::graphics::get_sprite_sheet(nr);

      g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
        if constexpr (requires { props.sprite_sheet_data.shape_sprite_sheets; }) {
          props.sprite_sheet_data.shape_sprite_sheets = fan::graphics::add_existing_sprite_sheet_shape(nr, props.sprite_sheet_data.shape_sprite_sheets, sprite_sheet);
          props.sprite_sheet_data.current_sprite_sheet = fan::graphics::shape_sprite_sheets()[props.sprite_sheet_data.shape_sprite_sheets].back();
        }
      });
      if (!get_visual_id()) {
        return;
      }

      auto& ri = shape_get_ri(sprite);
      ri.sprite_sheet_data.shape_sprite_sheets = fan::graphics::add_existing_sprite_sheet_shape(nr, ri.sprite_sheet_data.shape_sprite_sheets, sprite_sheet);
      ri.sprite_sheet_data.current_sprite_sheet = fan::graphics::shape_sprite_sheets()[ri.sprite_sheet_data.shape_sprite_sheets].back();
    }
    else {
      fan::throw_error("Unimplemented for this shape");
    }
  }

  bool shapes::shape_t::is_sprite_sheet_finished() const {
    return is_sprite_sheet_finished(get_current_sprite_sheet_id());
  }
  bool shapes::shape_t::is_sprite_sheet_finished(sprite_sheet_id_t nr) const {
    if (!get_visual_id()) {
      return true;
    }
    auto& sprite_sheet = fan::graphics::get_sprite_sheet(nr);
    auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
    fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
    return sheet_data.current_frame == sprite_sheet.selected_frames.size() - 1;
  }
  
  int shapes::shape_t::get_current_sprite_sheet_last_frame_index() const {
    const auto& sheet = get_current_sprite_sheet();
    return sheet.selected_frames[sheet.selected_frames.size() - 1];
  }
  void shapes::shape_t::finish_current_sprite_sheet() {
    set_current_sprite_sheet_frame(get_current_sprite_sheet_last_frame_index());
  }
  void shapes::shape_t::set_sprite_sheet_loop(sprite_sheet_id_t nr, bool flag) {
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        auto& shape_sheets = fan::graphics::shape_sprite_sheets()[props.sprite_sheet_data.shape_sprite_sheets];

        for (auto& sheet_id : shape_sheets) {
          if (sheet_id == nr) {
            sprite_sheet_id_t new_nr = ss_counter()++;
            all_sprite_sheets()[new_nr] = all_sprite_sheets()[nr];
            all_sprite_sheets()[new_nr].loop = flag;

            sheet_id = new_nr;

            auto& sheet = all_sprite_sheets()[new_nr];
            ss_lookup()[{props.sprite_sheet_data.shape_sprite_sheets, sheet.name}] = new_nr;

            if (props.sprite_sheet_data.current_sprite_sheet == nr) {
              props.sprite_sheet_data.current_sprite_sheet = new_nr;
            }

            break;
          }
        }
      }
    });
  }
  void fan::graphics::shapes::shape_t::reset_current_sprite_sheet_frame() {
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        props.sprite_sheet_data.previous_frame = props.sprite_sheet_data.current_frame;
        props.sprite_sheet_data.current_frame = 0;
        if (get_visual_id()) {
          auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
          ri.sprite_sheet_data.previous_frame = props.sprite_sheet_data.previous_frame;
          ri.sprite_sheet_data.current_frame = 0;
        }
      }
    });
  }
  void fan::graphics::shapes::shape_t::reset_current_sprite_sheet() {
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        props.sprite_sheet_data.previous_frame = props.sprite_sheet_data.current_frame;
        props.sprite_sheet_data.current_frame = 0;
        props.sprite_sheet_data.frame_accumulator = 0.f;
        if (get_visual_id()) {
          auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
          ri.sprite_sheet_data.previous_frame = props.sprite_sheet_data.previous_frame;
          ri.sprite_sheet_data = props.sprite_sheet_data;
        }
      }
    });
  }

  void fan::graphics::shapes::shape_t::set_sprite_sheet_next_frame(int advance) {
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        auto found = fan::graphics::all_sprite_sheets().find(props.sprite_sheet_data.current_sprite_sheet);
        if (found == fan::graphics::all_sprite_sheets().end()) {
          fan::throw_error("current_sprite_sheet not found");
        }
        auto& sheet_data = props.sprite_sheet_data;
        auto& sprite_sheet = found->second;

        int frame_count = sprite_sheet.selected_frames.size();

        sheet_data.previous_frame = sheet_data.current_frame;

        sheet_data.current_frame = (sheet_data.current_frame + advance);

        if (sprite_sheet.loop) {
          sheet_data.current_frame %= frame_count;
        }
        else {
          sheet_data.current_frame = std::min(sheet_data.current_frame, frame_count - 1);
        }

        if (!get_visual_id()) {
          return;
        }
        fan::vec2 sign = get_image_sign();
        fan::vec2i8 new_sign = {
          (int8_t)fan::math::sgn(sign.x),
          (int8_t)fan::math::sgn(sign.y)
        };

        if (sprite_sheet.selected_frames.empty()) {
          return;
        }
        int actual_frame = sprite_sheet.selected_frames[sheet_data.current_frame];
        int total_frames = 0;
        for (auto& img : sprite_sheet.images) {
          total_frames += img.hframes * img.vframes;
        }
        actual_frame = std::min(actual_frame, total_frames - 1);
        int image_index = 0;
        frame_count = 0;
        for (int i = 0; i < sprite_sheet.images.size(); ++i) {
          int frames_in_this_image =
            sprite_sheet.images[i].hframes * sprite_sheet.images[i].vframes;
          if (actual_frame < frame_count + frames_in_this_image) {
            image_index = i;
            break;
          }
          frame_count += frames_in_this_image;
        }
        int local_frame = actual_frame - frame_count;
        auto& current_image = sprite_sheet.images[image_index];
        bool image_changed = get_image() != current_image.image;
        if (image_changed) {
          set_image(current_image.image);
        }
        fan::vec2 image_size = get_image().get_size();
        if (image_size.x > 0 && image_size.y > 0) {
          fan::vec2 frame_pixel_size = image_size / fan::vec2(current_image.hframes, current_image.vframes);
          f32_t aspect = frame_pixel_size.x / frame_pixel_size.y;
          fan::vec2 size = get_size();
          size.x = size.y * aspect;
          set_size(size);
        }
        fan::vec2 tc_size = {
          1.0 / current_image.hframes,
          1.0 / current_image.vframes
        };
        int frame_x = local_frame % current_image.hframes;
        int frame_y = local_frame / current_image.hframes;

        fan::vec2 pos = {
          frame_x * tc_size.x,
          frame_y * tc_size.y
        };
        fan::vec2 tc_abs = tc_size.abs();
        fan::vec2 pos_clamped = pos;
        if (new_sign.x < 0) {
          pos_clamped.x = 1.0f - (pos.x + tc_abs.x);
          pos_clamped.x = 1.0f - pos_clamped.x;
        }
        if (new_sign.y < 0) {
          pos_clamped.y += tc_abs.y;
          pos_clamped.y = fan::math::clamp(pos_clamped.y, 0.0f, 1.0f - tc_abs.y);
        }

        set_tc_position(pos_clamped);
        set_tc_size(tc_size * fan::vec2(new_sign));
        auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
        ri.sprite_sheet_data = sheet_data;
      }
    });
  }

  sprite_sheet_shape_id_t shapes::shape_t::get_shape_sprite_sheet_id() const {
    sprite_sheet_shape_id_t sheet;
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data.shape_sprite_sheets; }) {
        sheet = props.sprite_sheet_data.shape_sprite_sheets;
      }
    });
    return sheet;
  }

  std::unordered_map<std::string, fan::graphics::sprite_sheet_id_t> shapes::shape_t::get_sprite_sheets() const {
    std::unordered_map<std::string, fan::graphics::sprite_sheet_id_t> result;

    for (auto& sprite_sheet_ids : fan::graphics::shape_sprite_sheets()[get_shape_sprite_sheet_id()]) {
      auto& sheet = fan::graphics::all_sprite_sheets()[sprite_sheet_ids];
      result[sheet.name] = sprite_sheet_ids;
    }
    return result;
  }

  void shapes::shape_t::set_sprite_sheet_fps(f32_t fps) {
    if (get_shape_type() != fan::graphics::shapes::shape_type_t::sprite) {
      fan::throw_error("unimplemented for this shape");
    }

    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        // Only modify sprite sheets belonging to THIS shape
        auto& shape_sheets = shape_sprite_sheets()[props.sprite_sheet_data.shape_sprite_sheets];
        for (auto& sprite_sheet_id : shape_sheets) {
          auto& sheet = all_sprite_sheets()[sprite_sheet_id];
          sheet.fps = fps;
        }
      }
    });
  }

  bool shapes::shape_t::has_sprite_sheet() {
    if (get_shape_type() != fan::graphics::shapes::shape_type_t::sprite) {
      return false;
    }
    auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
    fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
    return sheet_data.frame_update_nr.iic() == false;
  }

  void shapes::shape_t::set_sprite_sheet_frames(uint32_t image_index, int horizontal_frames, int vertical_frames) {
    if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
      auto& current_sheet = get_sprite_sheet();
      current_sheet.images[image_index].hframes = horizontal_frames;
      current_sheet.images[image_index].vframes = vertical_frames;
      play_sprite_sheet();
    }
    else {
      fan::throw_error("Unimplemented for this shape");
    }
  }

  sprite_sheet_id_t& shapes::shape_t::get_current_sprite_sheet_id() const {
    sprite_sheet_id_t* ptr = 0;
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data.current_sprite_sheet; }) {
        ptr = &props.sprite_sheet_data.current_sprite_sheet;
      }
    });
    return *ptr;
  }

  bool shapes::shape_t::sprite_sheet_on(const std::string& name, int frame_index) {
    auto cur_frame = get_current_sprite_sheet_frame();
    return cur_frame == frame_index && name == get_current_sprite_sheet().name;
  }

  bool shapes::shape_t::sprite_sheet_on(const std::string& name, const std::initializer_list<int>& arr) {
    auto cur_frame = get_current_sprite_sheet_frame();
    bool str_eq = name == get_current_sprite_sheet().name;
    for (auto& i : arr) {
      if (cur_frame == i && str_eq) {
        return true;
      }
    }
    return false;
  }

  bool shapes::shape_t::sprite_sheet_crossed(const std::string& name, int frame_index) {
    if (name != get_current_sprite_sheet().name) {
      return false;
    }

    int prev = 0;
    int curr = 0;

    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        prev = props.sprite_sheet_data.previous_frame;
        curr = props.sprite_sheet_data.current_frame;
      }
    });
    return prev < frame_index && curr >= frame_index;
  }

  void shapes::shape_t::set_current_sprite_sheet_id(sprite_sheet_id_t sprite_sheet_id) {
  #if FAN_DEBUG >= fan_debug_medium
    if (!sprite_sheet_id) {
      fan::throw_error("invalid sprite_sheet id");
    }
  #endif

    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        props.sprite_sheet_data.previous_frame = props.sprite_sheet_data.current_frame;
        props.sprite_sheet_data.current_frame = 0;
      }
    });

    get_current_sprite_sheet_id() = sprite_sheet_id;
  }


  sprite_sheet_t& shapes::shape_t::get_current_sprite_sheet() const {
    auto found = all_sprite_sheets().find(get_current_sprite_sheet_id());
    #if FAN_DEBUG >= fan_debug_medium
    if (found == all_sprite_sheets().end()) {
      fan::throw_error("sprite_sheet not found");
    }
    #endif
    return found->second;
  }
  int shapes::shape_t::get_previous_sprite_sheet_frame() const {
    int prev_frame = 0;
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        prev_frame = props.sprite_sheet_data.previous_frame;
      }
    });
    return prev_frame;
  }
  int shapes::shape_t::get_current_sprite_sheet_frame() const {
    int frame = 0;
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        frame = props.sprite_sheet_data.current_frame;
      }
    });
    return frame;
  }
  void shapes::shape_t::set_current_sprite_sheet_frame(int frame_id) {
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        props.sprite_sheet_data.current_frame = frame_id;
        if (get_visual_id()) {
          auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
          ri.sprite_sheet_data.current_frame = frame_id;
        }
      }
    });
  }
  int shapes::shape_t::get_current_sprite_sheet_frame_count() {
    int count = 0;
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data.current_sprite_sheet; }) {
        count = get_current_sprite_sheet().selected_frames.size();
      }
    });
    return count;
  }

  sprite_sheet_t* shapes::shape_t::get_sprite_sheet(const std::string& name) {
    sprite_sheet_t* sheet = nullptr;
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        auto found = ss_lookup().find({props.sprite_sheet_data.shape_sprite_sheets, name});
        if (found == ss_lookup().end()) {
          fan::throw_error("sprite_sheet not found:", name);
        }
        else {
          auto found2 = all_sprite_sheets().find(found->second);
          if (found2 == all_sprite_sheets().end()) {
            fan::throw_error("sprite_sheet not found:", name);
          }
          else {
            sheet = &found2->second;
          }
        }
      }
    });
    return sheet;
  }

  void shapes::shape_t::set_light_position(const fan::vec3& new_pos) {
    if (get_shape_type() != fan::graphics::shapes::shape_type_t::shadow) {
      fan::throw_error("invalid function call for current shape");
    }
    reinterpret_cast<shadow_t::vi_t*>(GetRenderData(fan::graphics::g_shapes->shaper))->light_position = new_pos;
    if (fan::graphics::ctx().window->renderer == fan::window_t::renderer_t::opengl) {
      auto& data = g_shapes->shaper.ShapeList[get_visual_id()];
      g_shapes->shaper.ElementIsPartiallyEdited(
        data.sti,
        data.blid,
        data.ElementIndex,
        fan::member_offset(&shadow_t::vi_t::light_position),
        sizeof(shadow_t::vi_t::light_position)
      );
    }
  }

  void shapes::shape_t::set_light_radius(f32_t radius) {
    if (get_shape_type() != fan::graphics::shapes::shape_type_t::shadow) {
      fan::throw_error("invalid function call for current shape");
    }

    reinterpret_cast<shadow_t::vi_t*>(GetRenderData(fan::graphics::g_shapes->shaper))->light_radius = radius;
    if (fan::graphics::ctx().window->renderer == fan::window_t::renderer_t::opengl) {
      auto& data = g_shapes->shaper.ShapeList[get_visual_id()];
      g_shapes->shaper.ElementIsPartiallyEdited(
        data.sti,
        data.blid,
        data.ElementIndex,
        fan::member_offset(&shadow_t::vi_t::light_radius),
        sizeof(shadow_t::vi_t::light_radius)
      );
    }
  }

  // for line
  void shapes::shape_t::set_thickness(f32_t new_thickness) {
  #if FAN_DEBUG >= 3
    if (get_shape_type() != fan::graphics::shapes::shape_type_t::line) {
      fan::throw_error("Invalid function call 'set_thickness', shape was not line");
    }
  #endif
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.thickness; }) {
        props.thickness = new_thickness;
      }
    });

    if (!get_visual_id()) {
      return;
    }

    ((line_t::vi_t*)GetRenderData(fan::graphics::g_shapes->shaper))->thickness = new_thickness;
    auto& data = g_shapes->shaper.ShapeList[get_visual_id()];
    g_shapes->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&line_t::vi_t::thickness),
      sizeof(line_t::vi_t::thickness)
    );
  }

  void shapes::shape_t::apply_floating_motion(f32_t time, f32_t amplitude, f32_t speed, f32_t phase) {
    fan::throw_error("time todo");
    f32_t y = std::sin(time * speed + phase) * amplitude;
    set_y(y);
  } // shape_t

  void shapes::shape_t::start_particles(f32_t start_offset) {
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.loop_enabled_time; }) {
        props.loop_enabled_time = fan::time::now() / 1e9 - start_offset; // negate the offset to travel to future
        props.loop_disabled_time = -1;
      }
    });
    if (!get_visual_id()) {
      return;
    }
    auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)
      GetData(fan::graphics::g_shapes->shaper);
    ri.loop_enabled_time = fan::time::now() / 1e9 - start_offset;  // negate the offset to travel to future
    ri.loop_disabled_time = -1;
  }

  void shapes::shape_t::stop_particles() {
    g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
      if constexpr (requires { props.loop_disabled_time; }) {
        props.loop_disabled_time = fan::time::now() / 1e9;
      }
    });
    if (!get_visual_id()) {
      return;
    }
    auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)
      GetData(fan::graphics::g_shapes->shaper);
    ri.loop_disabled_time = fan::time::now() / 1e9;
  }


  fan::graphics::shaper_t::ShapeRenderData_t* shapes::shape_t::GetRenderData(fan::graphics::shaper_t& shaper) const {
    return static_cast<ShapeID_t*>(get_visual_shape())->GetRenderData(shaper);
  }
  fan::graphics::shaper_t::ShapeData_t* shapes::shape_t::GetData(fan::graphics::shaper_t& shaper) const {
    return static_cast<ShapeID_t*>(get_visual_shape())->GetData(shaper);
  }

  shaper_t::ShapeTypes_t::nd_t& shapes::shape_t::get_shape_type_data() {
    return g_shapes->shaper.ShapeTypes[get_shape_type()];
  }

  uint8_t* shapes::shape_t::get_keys() {
    return g_shapes->shaper.GetKeys(get_visual_id());
  }
  shaper_t::KeyPackSize_t shapes::shape_t::get_keys_size() {
    return g_shapes->shaper.GetKeysSize(get_visual_id());
  }

  static constexpr uint8_t default_visible = 1;
  //shapes
  shapes::shape_t shapes::light_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->light_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::line_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->line_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::rectangle_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->rectangle_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::sprite_t::push_back(const properties_t& properties){
    bool uses_texture_pack = properties.texture_pack_unique_id.iic() == false && g_shapes->texture_pack;
    fan::graphics::texture_pack::ti_t ti;

    properties_t modified_props = properties;

    if(uses_texture_pack){
      uses_texture_pack = !g_shapes->texture_pack->qti((*g_shapes->texture_pack)[properties.texture_pack_unique_id].name, &ti);
      if(uses_texture_pack){
        auto& img = g_shapes->texture_pack->get_pixel_data(properties.texture_pack_unique_id).image;
        auto& img_data = image_get_data(img);
        ti.position /= img_data.size;
        ti.size /= img_data.size;
        modified_props.image = img;
        //modified_props.texture_pack_unique_id.sic();
      }
    }

    if(uses_texture_pack){
      modified_props.tc_position = ti.position;
      modified_props.tc_size = ti.size;
    }

    auto new_item = g_shapes->add_shape(g_shapes->sprite_list, modified_props);

    //g_shapes->visit_shape_draw_data(new_item.NRI, [&](auto& draw_data) {
    //  if constexpr (requires { draw_data.texture_pack_unique_id; }) {
    //    if (uses_texture_pack) {
    //      draw_data.texture_pack_unique_id = properties.texture_pack_unique_id;
    //    }
    //  }
    //});

    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::unlit_sprite_t::push_back(const properties_t& properties){
    bool uses_texture_pack = properties.texture_pack_unique_id.iic() == false && g_shapes->texture_pack;
    fan::graphics::texture_pack::ti_t ti;
    if(uses_texture_pack){
      uses_texture_pack = !g_shapes->texture_pack->qti((*g_shapes->texture_pack)[properties.texture_pack_unique_id].name, &ti);
      if(uses_texture_pack){
        auto img_size = g_shapes->texture_pack->get_pixel_data(properties.texture_pack_unique_id).image.get_size();
        ti.position /= img_size;
        ti.size /= img_size;
      }
    }

    properties_t modified_props = properties;
    if(uses_texture_pack){
      modified_props.tc_position = ti.position;
      modified_props.tc_size = ti.size;
    }

    auto new_item = g_shapes->add_shape(g_shapes->unlit_sprite_list, modified_props);

    g_shapes->dispatch_shape(new_item.NRI, [&](auto& list, auto& sd) {
      using list_t = std::decay_t<decltype(list)>;

      if constexpr (std::is_same_v<list_t, shapes::unlit_sprite_list_t>) {
        typename list_t::nr_t nr;
        nr.gint() = sd.data_nr;

        auto& draw_data = list[nr];

        if (uses_texture_pack) {
          draw_data.texture_pack_unique_id = properties.texture_pack_unique_id;
        }
      }
    });

    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::text_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->text_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::circle_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->circle_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::capsule_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->capsule_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::polygon_t::push_back(const properties_t& properties){
    if(properties.vertices.empty()){
      fan::throw_error("invalid vertices");
    }

    auto new_item = g_shapes->add_shape(g_shapes->polygon_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::grid_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->grid_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::particles_t::push_back(const properties_t& properties){
    properties_t modified_props = properties;
    modified_props.begin_time = fan::time::now();
    modified_props.loop_enabled_time = fan::time::now() / 1e9;
    modified_props.loop_disabled_time = -1;
    auto new_item = g_shapes->add_shape(g_shapes->particles_list, modified_props);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::universal_image_renderer_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->universal_image_renderer_list, properties);

    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::gradient_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->gradient_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::shadow_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->shadow_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::shader_shape_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->shader_shape_list, properties);
    g_shapes->shaper.GetShader(shape_type) = properties.shader;

    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

#if defined(FAN_3D)
  shapes::shape_t shapes::rectangle3d_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->rectangle3d_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }

  shapes::shape_t shapes::line3d_t::push_back(const properties_t& properties){
    auto new_item = g_shapes->add_shape(g_shapes->line3d_list, properties);
    fan::graphics::shaper_t::ShapeID_t ret;
    ret.gint() = new_item.NRI;
    return ret;
  }
#endif

#endif // FAN_2D
} // namespace fan::graphics

#if defined(FAN_2D)

void fan::graphics::shapes::shape_t::sprite_sheet_frame_update_cb(
  fan::graphics::shaper_t& shaper,
  fan::graphics::shapes::shape_t* shape
){
  g_shapes->visit_shape_draw_data(shape->NRI, [&](auto& props) {
    if constexpr (requires { props.sprite_sheet_data; }) {
      auto& sheet_data = props.sprite_sheet_data;
      
      if (props.sprite_sheet_data.current_sprite_sheet) {
        auto& sprite_sheet = all_sprite_sheets()[props.sprite_sheet_data.current_sprite_sheet];
        auto& selected_frames = sprite_sheet.selected_frames;
        
        if (selected_frames.empty()) {
          return;
        }

        f32_t dt = fan::graphics::get_window().m_delta_time;
        f32_t frame_duration = 1.0f / sprite_sheet.fps;
        
        sheet_data.frame_accumulator += dt;
        
        while (sheet_data.frame_accumulator >= frame_duration) {
          sheet_data.frame_accumulator -= frame_duration;
          shape->set_sprite_sheet_next_frame();
        }

        if (shape->get_visual_id()) {
          auto& ri = *(sprite_t::ri_t*)shape->GetData(shaper);
          ri.sprite_sheet_data = sheet_data;
          props.sprite_sheet_data = ri.sprite_sheet_data;
        }
      }
    }
  });
}

fan::graphics::sprite_sheet_data_t& fan::graphics::shapes::shape_t::get_sprite_sheet_data() {
  fan::graphics::sprite_sheet_data_t* data = nullptr;
  g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
    if constexpr (requires { props.sprite_sheet_data; }) {
      data = &props.sprite_sheet_data;
    }
  });
  return *data;
}

fan::graphics::sprite_sheet_t& fan::graphics::shapes::shape_t::get_sprite_sheet() {
  fan::graphics::sprite_sheet_t* result = nullptr;
  g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
    if constexpr (requires { props.sprite_sheet_data.current_sprite_sheet; }) {
      result = &::fan::graphics::get_sprite_sheet(props.sprite_sheet_data.current_sprite_sheet);
    }
  });
  if (!result) {
    fan::throw_error("sprite_sheet not available for this shape");
  }
  return *result;
}

void fan::graphics::shapes::shape_t::play_sprite_sheet(){
  fan::graphics::sprite_sheet_data_t* sheet_data = 0;
  g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
    if constexpr (requires { props.sprite_sheet_data; }) {
      props.sprite_sheet_data.start_sprite_sheet = true;
      props.sprite_sheet_data.frame_accumulator = 0.f;
      sheet_data = &props.sprite_sheet_data;
    }
  });

  set_sprite_sheet_next_frame(0);

  if (sheet_data->frame_update_nr) {
    fan::graphics::ctx().update_callback->unlrec(sheet_data->frame_update_nr);
  }

  sheet_data->frame_update_nr = fan::graphics::ctx().update_callback->NewNodeLast();

  (*fan::graphics::ctx().update_callback)[sheet_data->frame_update_nr] = [nr = NRI](void* ptr) {
    sprite_sheet_frame_update_cb(g_shapes->shaper, (fan::graphics::shapes::shape_t*)&nr);
  };
}

void fan::graphics::shapes::shape_t::stop_sprite_sheet() {
  g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
    if constexpr (requires { props.sprite_sheet_data; }) {
      auto& sheet_data = props.sprite_sheet_data;
      props.sprite_sheet_data.start_sprite_sheet = false;
      if (sheet_data.frame_update_nr) {
        fan::graphics::ctx().update_callback->unlrec(sheet_data.frame_update_nr);
        sheet_data.frame_update_nr.sic();
      }
      if (get_visual_id()) {
        auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
        ri.sprite_sheet_data = sheet_data;
      }
    }
  });
}

void fan::graphics::shapes::shape_t::play_sprite_sheet_once(const std::string& sprite_sheet_name) {
  auto* original_sheet = get_sprite_sheet(sprite_sheet_name);
  if (!original_sheet) {
    return;
  }
  
  sprite_sheet_t sheet_copy = *original_sheet;
  sheet_copy.loop = false;
  
  set_sprite_sheet(sheet_copy);
  set_current_sprite_sheet_frame(0);
}

void fan::graphics::shapes::shape_t::set_sprite_sheet(const std::string& name) {
  set_sprite_sheet(*get_sprite_sheet(name));
}

void fan::graphics::shapes::shape_t::set_sprite_sheet(const fan::graphics::sprite_sheet_t& sprite_sheet) {
  if (get_shape_type() != fan::graphics::shapes::shape_type_t::sprite) {
    fan::throw_error("unimplemented for this shape");
  }
  
  g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
    if constexpr (requires { props.sprite_sheet_data.shape_sprite_sheets; }) {
      sprite_sheet_id_t new_sheet_id = ss_counter()++;
      all_sprite_sheets()[new_sheet_id] = sprite_sheet;
      
      props.sprite_sheet_data.current_sprite_sheet = new_sheet_id;
      
      auto& shape_sheets = shape_sprite_sheets()[props.sprite_sheet_data.shape_sprite_sheets];
      auto it = std::find(shape_sheets.begin(), shape_sheets.end(), new_sheet_id);
      if (it == shape_sheets.end()) {
        shape_sheets.push_back(new_sheet_id);
      }
      
      ss_lookup()[{props.sprite_sheet_data.shape_sprite_sheets, sprite_sheet.name}] = new_sheet_id;
    }
  });
  play_sprite_sheet();
}

void fan::graphics::shapes::shape_t::add_sprite_sheet(const fan::graphics::sprite_sheet_t& sprite_sheet) {
  if (get_shape_type() != fan::graphics::shapes::shape_type_t::sprite) {
    fan::throw_error("unimplemented for this shape");
  }
  g_shapes->visit_shape_draw_data(NRI, [&](auto& props) {
    if constexpr (requires { props.sprite_sheet_data.shape_sprite_sheets; }) {
      props.sprite_sheet_data.shape_sprite_sheets = add_shape_sprite_sheet(props.sprite_sheet_data.shape_sprite_sheets, sprite_sheet);
      props.sprite_sheet_data.current_sprite_sheet = shape_sprite_sheets()[props.sprite_sheet_data.shape_sprite_sheets].back();
      if (get_visual_id()) {
        auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
        ri.sprite_sheet_data.shape_sprite_sheets = props.sprite_sheet_data.shape_sprite_sheets;
        ri.sprite_sheet_data.current_sprite_sheet = props.sprite_sheet_data.current_sprite_sheet;
      }
    }
  });
  play_sprite_sheet();
}

#endif

#if defined(FAN_2D)

#if defined(FAN_JSON)
namespace fan::graphics {
  bool shape_to_json(fan::graphics::shapes::shape_t& shape, fan::json* json) {
    fan::json& out = *json;
    switch (shape.get_shape_type()) {
    case fan::graphics::shapes::shape_type_t::light: {
      fan::graphics::shapes::light_t::properties_t defaults;
      out["shape"] = "light";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_parallax_factor() != defaults.parallax_factor) {
        out["parallax_factor"] = shape.get_parallax_factor();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_flags() != defaults.flags) {
        out["flags"] = shape.get_flags();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::line: {
      fan::graphics::shapes::line_t::properties_t defaults;
      out["shape"] = "line";
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_src() != defaults.src) {
        out["src"] = shape.get_src();
      }
      if (shape.get_dst() != defaults.dst) {
        out["dst"] = shape.get_dst();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::rectangle: {
      fan::graphics::shapes::rectangle_t::properties_t defaults;
      out["shape"] = "rectangle";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_outline_color() != defaults.outline_color) {
        out["outline_color"] = shape.get_outline_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::sprite: {
      fan::graphics::shapes::sprite_t::properties_t defaults;
      out["shape"] = "sprite";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_parallax_factor() != defaults.parallax_factor) {
        out["parallax_factor"] = shape.get_parallax_factor();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      if (shape.get_flags() != defaults.flags) {
        out["flags"] = shape.get_flags();
      }
      if (shape.get_tc_position() != defaults.tc_position) {
        out["tc_position"] = shape.get_tc_position();
      }
      if (shape.get_tc_size() != defaults.tc_size) {
        out["tc_size"] = shape.get_tc_size();
      }
      g_shapes->visit_shape_draw_data(shape.NRI, [&]<typename T>(T& properties) {
        if constexpr (requires {
          properties.texture_pack_unique_id;
        }) {
          if constexpr (!std::is_same_v<T, fan::graphics::shapes::sprite_t::properties_t>) {
            return;
          }
          if (*fan::graphics::g_shapes->texture_pack && properties.texture_pack_unique_id) {
            const auto& t = (*fan::graphics::g_shapes->texture_pack)[properties.texture_pack_unique_id];
            if (t.name.size()) {
              out["texture_pack_name"] = t.name;
            }
          }
          if (properties.sprite_sheet_data.shape_sprite_sheets) {
            fan::json sprite_sheet_array = fan::json::array();
            for (auto& sprite_sheet_ids : fan::graphics::shape_sprite_sheets()[properties.sprite_sheet_data.shape_sprite_sheets]) {
              sprite_sheet_array.push_back(sprite_sheet_ids.id);
            }
            if (sprite_sheet_array.empty() == false) {
              out["animations"] = sprite_sheet_array;
            }
          }
          fan::json images_array = fan::json::array();

          auto main_image = properties.image;
          auto img_json = fan::graphics::image_to_json(main_image);
          if (!img_json.empty()) {
            images_array.push_back(img_json);
          }

          auto images = properties.images;
          for (auto& image : images) {
            img_json = fan::graphics::image_to_json(image);
            if (!img_json.empty()) {
              images_array.push_back(img_json);
            }
          }

          if (!images_array.empty()) {
            out["images"] = images_array;
          }
        }
      });
      break;
    }
    case fan::graphics::shapes::shape_type_t::unlit_sprite: {
      fan::graphics::shapes::unlit_sprite_t::properties_t defaults;
      out["shape"] = "unlit_sprite";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_parallax_factor() != defaults.parallax_factor) {
        out["parallax_factor"] = shape.get_parallax_factor();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      if (shape.get_flags() != defaults.flags) {
        out["flags"] = shape.get_flags();
      }
      if (shape.get_tc_position() != defaults.tc_position) {
        out["tc_position"] = shape.get_tc_position();
      }
      if (shape.get_tc_size() != defaults.tc_size) {
        out["tc_size"] = shape.get_tc_size();
      }
      if (*fan::graphics::g_shapes->texture_pack) {
        auto t = (*fan::graphics::g_shapes->texture_pack)[((fan::graphics::shapes::unlit_sprite_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper))->texture_pack_unique_id];
        if (t.name.size()) {
          out["texture_pack_name"] = t.name;
        }
      }

      fan::json images_array = fan::json::array();

      auto main_image = shape.get_image();
      auto img_json = fan::graphics::image_to_json(main_image);
      if (!img_json.empty()) {
        images_array.push_back(img_json);
      }

      auto images = shape.get_images();
      for (auto& image : images) {
        img_json = fan::graphics::image_to_json(image);
        if (!img_json.empty()) {
          images_array.push_back(img_json);
        }
      }

      if (!images_array.empty()) {
        out["images"] = images_array;
      }

      break;
    }
    case fan::graphics::shapes::shape_type_t::text: {
      out["shape"] = "text";
      break;
    }
    case fan::graphics::shapes::shape_type_t::circle: {
      fan::graphics::shapes::circle_t::properties_t defaults;
      out["shape"] = "circle";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_radius() != defaults.radius) {
        out["radius"] = shape.get_radius();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::grid: {
      fan::graphics::shapes::grid_t::properties_t defaults;
      out["shape"] = "grid";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_grid_size() != defaults.grid_size) {
        out["grid_size"] = shape.get_grid_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::particles: {
      fan::graphics::shapes::particles_t::properties_t defaults;
      auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper);
      out["shape"] = "particles";

      if (ri.loop != defaults.loop) {
        out["loop"] = ri.loop;
      }
      if (ri.position != defaults.position) {
        out["position"] = ri.position;
      }

      if (ri.start_size != defaults.start_size) {
        out["start_size"] = ri.start_size;
      }
      if (ri.end_size != defaults.end_size) {
        out["end_size"] = ri.end_size;
      }

      if (ri.begin_color != defaults.begin_color) {
        out["begin_color"] = ri.begin_color;
      }
      if (ri.end_color != defaults.end_color) {
        out["end_color"] = ri.end_color;
      }

      if (ri.alive_time != defaults.alive_time) {
        out["alive_time"] = ri.alive_time;
      }
      if (ri.respawn_time != defaults.respawn_time) {
        out["respawn_time"] = ri.respawn_time;
      }
      if (ri.count != defaults.count) {
        out["count"] = ri.count;
      }

      if (ri.start_velocity != defaults.start_velocity) {
        out["start_velocity"] = ri.start_velocity;
      }
      if (ri.end_velocity != defaults.end_velocity) {
        out["end_velocity"] = ri.end_velocity;
      }

      if (ri.start_angle_velocity != defaults.start_angle_velocity) {
        out["start_angle_velocity"] = ri.start_angle_velocity;
      }
      if (ri.end_angle_velocity != defaults.end_angle_velocity) {
        out["end_angle_velocity"] = ri.end_angle_velocity;
      }

      if (ri.begin_angle != defaults.begin_angle) {
        out["begin_angle"] = ri.begin_angle;
      }
      if (ri.end_angle != defaults.end_angle) {
        out["end_angle"] = ri.end_angle;
      }
      if (ri.angle != defaults.angle) {
        out["angle"] = ri.angle;
      }

      if (ri.spawn_spacing != defaults.spawn_spacing) {
        out["spawn_spacing"] = ri.spawn_spacing;
      }
      if (ri.expansion_power != defaults.expansion_power) {
        out["expansion_power"] = ri.expansion_power;
      }

      if (ri.start_spread != defaults.start_spread) {
        out["start_spread"] = ri.start_spread;
      }
      if (ri.end_spread != defaults.end_spread) {
        out["end_spread"] = ri.end_spread;
      }

      if (ri.jitter_start != defaults.jitter_start) {
        out["jitter_start"] = ri.jitter_start;
      }
      if (ri.jitter_end != defaults.jitter_end) {
        out["jitter_end"] = ri.jitter_end;
      }
      if (ri.jitter_speed != defaults.jitter_speed) {
        out["jitter_speed"] = ri.jitter_speed;
      }

      if (ri.size_random_range != defaults.size_random_range) {
        out["size_random_range"] = ri.size_random_range;
      }
      if (ri.color_random_range != defaults.color_random_range) {
        out["color_random_range"] = ri.color_random_range;
      }
      if (ri.angle_random_range != defaults.angle_random_range) {
        out["angle_random_range"] = ri.angle_random_range;
      }

      if (ri.shape != defaults.shape) {
        out["particle_shape"] = ri.shape;
      }

      fan::graphics::image_t image = shape.get_image();
      if (image) {
        out.update(fan::graphics::image_to_json(image), true);
      }
      break;
    }
    default: {
      fan::throw_error("unimplemented shape");
    }
    }
    return false;
  }
  bool json_to_shape(const fan::json& in, fan::graphics::shapes::shape_t* shape, const std::source_location& callers_path) {
    if (!in.contains("shape")) {
      return false;
    }
    std::string shape_type = in["shape"];
    switch (fan::get_hash(shape_type.c_str())) {
    case fan::get_hash("rectangle"): {
      fan::graphics::shapes::rectangle_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("outline_color")) {
        p.outline_color = in["outline_color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("light"): {
      fan::graphics::shapes::light_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("parallax_factor")) {
        p.parallax_factor = in["parallax_factor"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("flags")) {
        p.flags = in["flags"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("line"): {
      fan::graphics::shapes::line_t::properties_t p;
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("src")) {
        p.src = in["src"];
      }
      if (in.contains("dst")) {
        p.dst = in["dst"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("sprite"): {
      fan::graphics::shapes::sprite_t::properties_t p;
      p.blending = true;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("parallax_factor")) {
        p.parallax_factor = in["parallax_factor"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      if (in.contains("flags")) {
        p.flags = in["flags"];
      }
      if (in.contains("tc_position")) {
        p.tc_position = in["tc_position"];
      }
      if (in.contains("tc_size")) {
        p.tc_size = in["tc_size"];
      }
      bool contains_animations = in.contains("animations");
      // load texture pack only if no sprite sheet animations
      // because sprite sheet animations use image for it
      if (contains_animations == false && in.contains("texture_pack_name") && *fan::graphics::g_shapes->texture_pack) {
        p.texture_pack_unique_id = (*fan::graphics::g_shapes->texture_pack)[in["texture_pack_name"]];
      }
      *shape = p;

      fan::graphics::image_load_properties_t lp;
      if (in.contains("image_visual_output")) {
        lp.visual_output = in["image_visual_output"];
      }
      if (in.contains("image_format")) {
        lp.format = in["image_format"];
      }
      if (in.contains("image_type")) {
        lp.type = in["image_type"];
      }
      if (in.contains("image_min_filter")) {
        lp.min_filter = in["image_min_filter"];
      }
      if (in.contains("image_mag_filter")) {
        lp.mag_filter = in["image_mag_filter"];
      }
      if (in.contains("images") && in["images"].is_array()) {
        for (const auto [i, image_json] : fan::enumerate(in["images"])) {
          // leaking (cache taking care of it)
          fan::graphics::image_t image = fan::graphics::json_to_image(image_json, callers_path);
          if (i == 0) {
            shape->set_image(image);
          }
          else {
            auto images = shape->get_images();
            images[i - 1] = image;
            shape->set_images(images);
          }
        }
      }

      if (contains_animations) {
        for (auto& item : in["animations"]) {
          uint32_t anim_id = item.get<uint32_t>();
          auto existing_animation = fan::graphics::get_sprite_sheet(anim_id);
          shape->add_existing_sprite_sheet(anim_id);
        }
      }

      break;
    }
    case fan::get_hash("unlit_sprite"): {
      fan::graphics::shapes::unlit_sprite_t::properties_t p;
      p.blending = true;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("parallax_factor")) {
        p.parallax_factor = in["parallax_factor"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      if (in.contains("flags")) {
        p.flags = in["flags"];
      }
      if (in.contains("tc_position")) {
        p.tc_position = in["tc_position"];
      }
      if (in.contains("tc_size")) {
        p.tc_size = in["tc_size"];
      }
      if (in.contains("texture_pack_name") && *fan::graphics::g_shapes->texture_pack) {
        p.texture_pack_unique_id = (*fan::graphics::g_shapes->texture_pack)[in["texture_pack_name"]];
      }
      *shape = p;
      fan::graphics::image_load_properties_t lp;
      if (in.contains("image_visual_output")) {
        lp.visual_output = in["image_visual_output"];
      }
      if (in.contains("image_format")) {
        lp.format = in["image_format"];
      }
      if (in.contains("image_type")) {
        lp.type = in["image_type"];
      }
      if (in.contains("image_min_filter")) {
        lp.min_filter = in["image_min_filter"];
      }
      if (in.contains("image_mag_filter")) {
        lp.mag_filter = in["image_mag_filter"];
      }

      if (in.contains("images") && in["images"].is_array()) {
        for (const auto [i, image_json] : fan::enumerate(in["images"])) {
          if (!image_json.contains("image_path")) continue;

          auto path = image_json["image_path"];
          if (fan::io::file::exists(path)) {
            fan::graphics::image_load_properties_t lp;

            if (image_json.contains("image_visual_output")) lp.visual_output = image_json["image_visual_output"];
            if (image_json.contains("image_format")) lp.format = image_json["image_format"];
            if (image_json.contains("image_type")) lp.type = image_json["image_type"];
            if (image_json.contains("image_min_filter")) lp.min_filter = image_json["image_min_filter"];
            if (image_json.contains("image_mag_filter")) lp.mag_filter = image_json["image_mag_filter"];

            auto image = fan::graphics::ctx()->image_load_path_props(fan::graphics::ctx(), path, lp, callers_path);

            if (i == 0) {
              shape->set_image(image);
            }
            else {
              auto images = shape->get_images();
              images[i - 1] = image;
              shape->set_images(images);
            }
            (*fan::graphics::ctx().image_list)[image].image_path = path;
          }
        }
      }
      break;
    }
    case fan::get_hash("circle"): {
      fan::graphics::shapes::circle_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("radius")) {
        p.radius = in["radius"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("grid"): {
      fan::graphics::shapes::grid_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("grid_size")) {
        p.grid_size = in["grid_size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("particles"):{
      fan::graphics::shapes::particles_t::properties_t p;
      if (in.contains("loop")) {
        p.loop = in["loop"];
      }
      if (in.contains("position")) {
        p.position = in["position"];
      }

      if (in.contains("start_size")) {
        p.start_size = in["start_size"];
      }
      if (in.contains("end_size")) {
        p.end_size = in["end_size"];
      }

      if (in.contains("begin_color")) {
        p.begin_color = in["begin_color"];
      }
      if (in.contains("end_color")) {
        p.end_color = in["end_color"];
      }

      if (in.contains("alive_time")) {
        p.alive_time = in["alive_time"];
      }
      if (in.contains("respawn_time")) {
        p.respawn_time = in["respawn_time"];
      }
      if (in.contains("count")) {
        p.count = in["count"];
      }

      if (in.contains("start_velocity")) {
        p.start_velocity = in["start_velocity"];
      }
      if (in.contains("end_velocity")) {
        p.end_velocity = in["end_velocity"];
      }

      if (in.contains("start_angle_velocity")) {
        p.start_angle_velocity = in["start_angle_velocity"];
      }
      if (in.contains("end_angle_velocity")) {
        p.end_angle_velocity = in["end_angle_velocity"];
      }

      if (in.contains("begin_angle")) {
        p.begin_angle = in["begin_angle"];
      }
      if (in.contains("end_angle")) {
        p.end_angle = in["end_angle"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }

      if (in.contains("spawn_spacing")) {
        p.spawn_spacing = in["spawn_spacing"];
      }
      if (in.contains("expansion_power")) {
        p.expansion_power = in["expansion_power"];
      }

      if (in.contains("start_spread")) {
        p.start_spread = in["start_spread"];
      }
      if (in.contains("end_spread")) {
        p.end_spread = in["end_spread"];
      }

      if (in.contains("jitter_start")) {
        p.jitter_start = in["jitter_start"];
      }
      if (in.contains("jitter_end")) {
        p.jitter_end = in["jitter_end"];
      }
      if (in.contains("jitter_speed")) {
        p.jitter_speed = in["jitter_speed"];
      }

      if (in.contains("size_random_range")) {
        p.size_random_range = in["size_random_range"];
      }
      if (in.contains("color_random_range")) {
        p.color_random_range = in["color_random_range"];
      }
      if (in.contains("angle_random_range")) {
        p.angle_random_range = in["angle_random_range"];
      }

      if (in.contains("particle_shape")) {
        p.shape = in["particle_shape"];
      }
      p.image = fan::graphics::json_to_image(in, callers_path);
      *shape = p;
      break;
    }
    default: {
      fan::throw_error("unimplemented shape");
    }
    }
    return false;
  }
  bool shape_serialize(fan::graphics::shapes::shape_t& shape, fan::json* out) {
    return shape_to_json(shape, out);
  }
}
#endif

namespace fan::graphics {
  bool shape_to_bin(fan::graphics::shapes::shape_t& shape, std::vector<uint8_t>* data) {
    std::vector<uint8_t>& out = *data;
    fan::write_to_vector(out, shape.get_shape_type());
    fan::write_to_vector(out, shape.gint());
    switch (shape.get_shape_type()) {
    case fan::graphics::shapes::shape_type_t::light: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_parallax_factor());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_flags());
      fan::write_to_vector(out, shape.get_angle());
      break;
    }
    case fan::graphics::shapes::shape_type_t::line: {
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_src());
      fan::write_to_vector(out, shape.get_dst());
      break;
    }
    case fan::graphics::shapes::shape_type_t::rectangle: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      break;
    }
    case fan::graphics::shapes::shape_type_t::sprite: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_parallax_factor());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      fan::write_to_vector(out, shape.get_flags());
      fan::write_to_vector(out, shape.get_image_data().image_path);
      fan::graphics::image_load_properties_t ilp = fan::graphics::ctx()->image_get_settings(fan::graphics::ctx(), shape.get_image());
      fan::write_to_vector(out, ilp.visual_output);
      fan::write_to_vector(out, ilp.format);
      fan::write_to_vector(out, ilp.type);
      fan::write_to_vector(out, ilp.min_filter);
      fan::write_to_vector(out, ilp.mag_filter);
      fan::write_to_vector(out, shape.get_tc_position());
      fan::write_to_vector(out, shape.get_tc_size());
      break;
    }
    case fan::graphics::shapes::shape_type_t::unlit_sprite: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_parallax_factor());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      fan::write_to_vector(out, shape.get_flags());
      fan::write_to_vector(out, shape.get_image_data().image_path);
      fan::graphics::image_load_properties_t ilp = fan::graphics::ctx()->image_get_settings(fan::graphics::ctx(), shape.get_image());
      fan::write_to_vector(out, ilp.visual_output);
      fan::write_to_vector(out, ilp.format);
      fan::write_to_vector(out, ilp.type);
      fan::write_to_vector(out, ilp.min_filter);
      fan::write_to_vector(out, ilp.mag_filter);
      fan::write_to_vector(out, shape.get_tc_position());
      fan::write_to_vector(out, shape.get_tc_size());
      break;
    }
    case fan::graphics::shapes::shape_type_t::circle: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_radius());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      break;
    }
    case fan::graphics::shapes::shape_type_t::grid: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_grid_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      break;
    }
    case fan::graphics::shapes::shape_type_t::particles: {
      auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper);
      fan::write_to_vector(out, ri.position);

      fan::write_to_vector(out, ri.start_size);
      fan::write_to_vector(out, ri.end_size);

      fan::write_to_vector(out, ri.begin_color);
      fan::write_to_vector(out, ri.end_color);

      fan::write_to_vector(out, ri.begin_time);
      fan::write_to_vector(out, ri.alive_time);
      fan::write_to_vector(out, ri.respawn_time);
      fan::write_to_vector(out, ri.count);

      fan::write_to_vector(out, ri.start_velocity);
      fan::write_to_vector(out, ri.end_velocity);

      fan::write_to_vector(out, ri.start_angle_velocity);
      fan::write_to_vector(out, ri.end_angle_velocity);

      fan::write_to_vector(out, ri.begin_angle);
      fan::write_to_vector(out, ri.end_angle);
      fan::write_to_vector(out, ri.angle);

      fan::write_to_vector(out, ri.spawn_spacing);
      fan::write_to_vector(out, ri.expansion_power);

      fan::write_to_vector(out, ri.start_spread);
      fan::write_to_vector(out, ri.end_spread);

      fan::write_to_vector(out, ri.jitter_start);
      fan::write_to_vector(out, ri.jitter_end);
      fan::write_to_vector(out, ri.jitter_speed);

      fan::write_to_vector(out, ri.size_random_range);
      fan::write_to_vector(out, ri.color_random_range);
      fan::write_to_vector(out, ri.angle_random_range);

      fan::write_to_vector(out, ri.shape);
      fan::write_to_vector(out, ri.blending);
      break;
    }
    case fan::graphics::shapes::shape_type_t::light_end: {
      break;
    }
    default: {
      fan::throw_error("unimplemented shape");
    }
    }
    return false;
  }
  bool bin_to_shape(const std::vector<uint8_t>& in, fan::graphics::shapes::shape_t* shape, uint64_t& offset, const std::source_location& callers_path) {
    using sti_t = std::remove_reference_t<decltype(fan::graphics::shapes::shape_t().get_shape_type())>;
    using nr_t = std::remove_reference_t<decltype(fan::graphics::shapes::shape_t().gint())>;
    sti_t shape_type = fan::vector_read_data<sti_t>(in, offset);
    nr_t nri = fan::vector_read_data<nr_t>(in, offset);
    switch (shape_type) {
    case fan::graphics::shapes::shape_type_t::rectangle: {
      fan::graphics::shapes::rectangle_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      p.outline_color = p.color;
      *shape = p;
      return false;
    }
    case fan::graphics::shapes::shape_type_t::light: {
      fan::graphics::shapes::light_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::line: {
      fan::graphics::shapes::line_t::properties_t p;
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.src = fan::vector_read_data<decltype(p.src)>(in, offset);
      p.dst = fan::vector_read_data<decltype(p.dst)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::sprite: {
      fan::graphics::shapes::sprite_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);

      std::string image_path = fan::vector_read_data<std::string>(in, offset);
      fan::graphics::image_load_properties_t ilp;
      ilp.visual_output = fan::vector_read_data<decltype(ilp.visual_output)>(in, offset);
      ilp.format = fan::vector_read_data<decltype(ilp.format)>(in, offset);
      ilp.type = fan::vector_read_data<decltype(ilp.type)>(in, offset);
      ilp.min_filter = fan::vector_read_data<decltype(ilp.min_filter)>(in, offset);
      ilp.mag_filter = fan::vector_read_data<decltype(ilp.mag_filter)>(in, offset);
      p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
      p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
      *shape = p;
      if (image_path.size()) {
        shape->get_image_data().image_path = image_path;
        shape->set_image(fan::graphics::ctx()->image_load_path_props(fan::graphics::ctx(), image_path, ilp, callers_path));
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::unlit_sprite: {
      fan::graphics::shapes::unlit_sprite_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
      std::string image_path = fan::vector_read_data<std::string>(in, offset);
      fan::graphics::image_load_properties_t ilp;
      ilp.visual_output = fan::vector_read_data<decltype(ilp.visual_output)>(in, offset);
      ilp.format = fan::vector_read_data<decltype(ilp.format)>(in, offset);
      ilp.type = fan::vector_read_data<decltype(ilp.type)>(in, offset);
      ilp.min_filter = fan::vector_read_data<decltype(ilp.min_filter)>(in, offset);
      ilp.mag_filter = fan::vector_read_data<decltype(ilp.mag_filter)>(in, offset);
      p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
      p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
      *shape = p;
      if (image_path.size()) {
        shape->get_image_data().image_path = image_path;
        shape->set_image(fan::graphics::ctx()->image_load_path_props(fan::graphics::ctx(), image_path, ilp, callers_path));
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::circle: {
      fan::graphics::shapes::circle_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.radius = fan::vector_read_data<decltype(p.radius)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::grid: {
      fan::graphics::shapes::grid_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.grid_size = fan::vector_read_data<decltype(p.grid_size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::particles: {
      fan::graphics::shapes::particles_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);

      p.start_size = fan::vector_read_data<decltype(p.start_size)>(in, offset);
      p.end_size = fan::vector_read_data<decltype(p.end_size)>(in, offset);

      p.begin_color = fan::vector_read_data<decltype(p.begin_color)>(in, offset);
      p.end_color = fan::vector_read_data<decltype(p.end_color)>(in, offset);

      p.begin_time = fan::vector_read_data<decltype(p.begin_time)>(in, offset);
      p.alive_time = fan::vector_read_data<decltype(p.alive_time)>(in, offset);
      p.respawn_time = fan::vector_read_data<decltype(p.respawn_time)>(in, offset);
      p.count = fan::vector_read_data<decltype(p.count)>(in, offset);

      p.start_velocity = fan::vector_read_data<decltype(p.start_velocity)>(in, offset);
      p.end_velocity = fan::vector_read_data<decltype(p.end_velocity)>(in, offset);

      p.start_angle_velocity = fan::vector_read_data<decltype(p.start_angle_velocity)>(in, offset);
      p.end_angle_velocity = fan::vector_read_data<decltype(p.end_angle_velocity)>(in, offset);

      p.begin_angle = fan::vector_read_data<decltype(p.begin_angle)>(in, offset);
      p.end_angle = fan::vector_read_data<decltype(p.end_angle)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);

      p.spawn_spacing = fan::vector_read_data<decltype(p.spawn_spacing)>(in, offset);
      p.expansion_power = fan::vector_read_data<decltype(p.expansion_power)>(in, offset);

      p.start_spread = fan::vector_read_data<decltype(p.start_spread)>(in, offset);
      p.end_spread = fan::vector_read_data<decltype(p.end_spread)>(in, offset);

      p.jitter_start = fan::vector_read_data<decltype(p.jitter_start)>(in, offset);
      p.jitter_end = fan::vector_read_data<decltype(p.jitter_end)>(in, offset);
      p.jitter_speed = fan::vector_read_data<decltype(p.jitter_speed)>(in, offset);

      p.size_random_range = fan::vector_read_data<decltype(p.size_random_range)>(in, offset);
      p.color_random_range = fan::vector_read_data<decltype(p.color_random_range)>(in, offset);
      p.angle_random_range = fan::vector_read_data<decltype(p.angle_random_range)>(in, offset);

      p.shape = fan::vector_read_data<decltype(p.shape)>(in, offset);
      p.blending = fan::vector_read_data<decltype(p.blending)>(in, offset);

      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::light_end: {
      return false;
    }
    default: {
      fan::throw_error("unimplemented");
    }
    }
    if (shape->gint() != nri) {
      fan::throw_error("");
    }
    return false;
  }

  bool shape_serialize(fan::graphics::shapes::shape_t& shape, std::vector<uint8_t>* out) {
    return shape_to_bin(shape, out);
  }
  bool shape_deserialize_t::iterate(const fan::json& json, fan::graphics::shapes::shape_t* shape, const std::source_location& callers_path) {
    if (init == false) {
      data.it = json.cbegin();
      init = true;
    }
    if (data.it == json.cend() || was_object) {
      return 0;
    }
    if (json.type() == fan::json::value_t::object) {
      json_to_shape(json, shape, callers_path);
      was_object = true;
      return 1;
    }
    else {
      json_to_shape(*data.it, shape, callers_path);
      ++data.it;
    }
    return 1;
  }
  bool shape_deserialize_t::iterate(const std::vector<uint8_t>& bin_data, fan::graphics::shapes::shape_t* shape) {
    if (bin_data.empty()) {
      return 0;
    }
    else if (data.offset >= bin_data.size()) {
      return 0;
    }
    bin_to_shape(bin_data, shape, data.offset);
    return 1;
  }
  fan::graphics::shapes::shape_t extract_single_shape(const fan::json& json_data, const std::source_location& callers_path) {
    fan::graphics::shapes::shape_t shape;
    fan::graphics::shape_deserialize_t iterator;
    if (json_data.contains("shapes")) {
      iterator.iterate(json_data["shapes"], &shape, callers_path);
    }
    else if (json_data.contains("shape")) {
      iterator.iterate(json_data, &shape, callers_path);
    }
    #if defined(FAN_GUI)
    else {
      fan::graphics::gui::print_warning("Failed to load shape - extract_single_shape");
    }
    #endif
    return shape;
  }
  fan::json read_json(std::string_view path, const std::source_location& callers_path) {
    std::string json_bytes;
    fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &json_bytes);
    return fan::json::parse(json_bytes);
  }

  void sprite_sheet_controller_t::add_state(const animation_state_t& state) {
    states.emplace_back(state);
  }

  void sprite_sheet_controller_t::update(fan::graphics::shapes::shape_t& shape, const fan::vec2& velocity) {
    last_direction = velocity;
    if (auto_flip_sprite) {
      update_image_sign(shape, velocity);
    }

    g_shapes->visit_shape_draw_data(shape.NRI, [&](auto& props) {
      if constexpr (requires { props.sprite_sheet_data; }) {
        for (auto& state : states) {
          if (state.animation_id.id == (uint32_t)-1) {
            auto found = ss_lookup().find({props.sprite_sheet_data.shape_sprite_sheets, state.name});
            if (found != ss_lookup().end()) {
              state.animation_id = found->second;
            }
          }
        }
      }
    });

    for (auto& state : states) {
      bool triggered = state.condition(shape);

      if (state.trigger_type == animation_state_t::one_shot) {
        if (triggered && !state.is_playing) {
          for (auto& other : states) {
            other.is_playing = false;
          }
          state.is_playing = true;
          if (prev_animation_id != state.animation_id) {
            shape.set_current_sprite_sheet_id(state.animation_id);
            shape.reset_current_sprite_sheet();
            current_animation_requires_velocity_fps = state.velocity_based_fps;
            if (!state.velocity_based_fps) {
              shape.set_sprite_sheet_fps(state.fps);
            }
            prev_animation_id = state.animation_id;
          }
        }
        if (state.is_playing && shape.is_sprite_sheet_finished(state.animation_id)) {
          state.is_playing = false;
          continue;
        }
        if (state.is_playing) {
          return;
        }
      }
      else if (triggered) {
        if (prev_animation_id != state.animation_id) {
          shape.set_current_sprite_sheet_id(state.animation_id);
          shape.reset_current_sprite_sheet();
          current_animation_requires_velocity_fps = state.velocity_based_fps;
          if (!state.velocity_based_fps) {
            shape.set_sprite_sheet_fps(state.fps);
          }
          prev_animation_id = state.animation_id;
        }

        if (state.velocity_based_fps) {
          f32_t speed = fan::math::clamp((f32_t)velocity.length() / 100.f + 0.35f, 0.f, 1.f);
          shape.set_sprite_sheet_fps(state.fps * speed);
        }
        return;
      }
    }
  }

  void sprite_sheet_controller_t::cancel_current() {
    for (auto& state : states) {
      state.is_playing = false;
    }
    prev_animation_id = -1;
  }

  sprite_sheet_controller_t::animation_state_t& sprite_sheet_controller_t::get_state(const std::string& name) {
    for (auto& state : states) {
      if (name == state.name) {
        return state;
      }
    }
    fan::throw_error("state not found");
    __unreachable();
  }

  void sprite_sheet_controller_t::update_image_sign(fan::graphics::shapes::shape_t& shape, const fan::vec2& direction) {
  #if defined(FAN_GUI)
    if (fan::graphics::gui::want_io()) {
      return;
    }
  #endif

    fan::vec2 sign = shape.get_image_sign();

    //if (direction.x > 0) {
    //  if (sign.x < 0) shape.set_image_sign({1, sign.y});
    //  desired_facing.x = 1;
    //  return;
    //}
    //if (direction.x < 0) {
    //  if (sign.x > 0) shape.set_image_sign({-1, sign.y});
    //  desired_facing.x = -1;
    //  return;
    //}

    //if (desired_facing.x != 0) {
    //  int desired = (int)fan::math::sgn(desired_facing.x);
    //  if ((int)fan::math::sgn(sign.x) != desired) {
    //    shape.set_image_sign({(f32_t)desired, sign.y});
    //  }
    //}
  }

  void sprite_sheet_controller_t::enable_directional(const directional_config_t& config) {
    idle_threshold = config.idle_threshold;
    use_8_directions = config.use_8_directions;

    direction_map[direction_e::idle] = config.idle;
    direction_map[direction_e::up] = config.move_up;
    direction_map[direction_e::down] = config.move_down;
    direction_map[direction_e::left] = config.move_left;
    direction_map[direction_e::right] = config.move_right;

    if (use_8_directions) {
      direction_map[direction_e::up_left] = config.move_up_left;
      direction_map[direction_e::up_right] = config.move_up_right;
      direction_map[direction_e::down_left] = config.move_down_left;
      direction_map[direction_e::down_right] = config.move_down_right;
    }

    add_state({
      .name = config.idle,
      .condition = [this](fan::graphics::shapes::shape_t& s) {
        return last_direction.length() < idle_threshold;
      }
    });

    auto make_directional_condition = [this](f32_t x_min, f32_t x_max, f32_t y_min, f32_t y_max) {
      return [this, x_min, x_max, y_min, y_max](fan::graphics::shapes::shape_t& s) {
        if (last_direction.length() < idle_threshold) return false;
        fan::vec2 norm = last_direction.normalized();
        return norm.x >= x_min && norm.x <= x_max && norm.y >= y_min && norm.y <= y_max;
      };
    };

    if (use_8_directions) {
      add_state({.name = config.move_up,  
        .condition = make_directional_condition(-0.4f, 0.4f, -1.f, -0.4f)});
      add_state({.name = config.move_down, 
        .condition = make_directional_condition(-0.4f, 0.4f, 0.4f, 1.f)});
      add_state({.name = config.move_left, 
        .condition = make_directional_condition(-1.f, -0.4f, -0.4f, 0.4f)});
      add_state({.name = config.move_right, 
        .condition = make_directional_condition(0.4f, 1.f, -0.4f, 0.4f)});
      add_state({.name = config.move_up_left, 
        .condition = make_directional_condition(-1.f, -0.4f, -1.f, -0.4f)});
      add_state({.name = config.move_up_right, 
        .condition = make_directional_condition(0.4f, 1.f, -1.f, -0.4f)});
      add_state({.name = config.move_down_left, 
        .condition = make_directional_condition(-1.f, -0.4f, 0.4f, 1.f)});
      add_state({.name = config.move_down_right, 
        .condition = make_directional_condition(0.4f, 1.f, 0.4f, 1.f)});
    }
    else {
      add_state({.name = config.move_up, 
        .condition = make_directional_condition(-1.f, 1.f, -1.f, -0.4f)});
      add_state({.name = config.move_down, 
        .condition = make_directional_condition(-1.f, 1.f, 0.4f, 1.f)});
      add_state({.name = config.move_left, 
        .condition = make_directional_condition(-1.f, -0.4f, -1.f, 1.f)});
      add_state({.name = config.move_right, 
        .condition = make_directional_condition(0.4f, 1.f, -1.f, 1.f)});
    }

    auto_update_animations = true;
  }

  void sprite_sheet_controller_t::add_directional_state(const std::string& animation_name, uint8_t direction) {
    direction_map[direction] = animation_name;
  }

  void sprite_sheet_controller_t::set_idle_animation(const std::string& name, f32_t threshold) {
    idle_threshold = threshold;
    direction_map[direction_e::idle] = name;

    add_state({
      .name = name,
      .condition = [this](fan::graphics::shapes::shape_t& s) {
        return last_direction.length() < idle_threshold;
      }
    });
  }

  void sprite_sheet_controller_t::override_animation(uint8_t direction, const std::string& name) {
    direction_map[direction] = name;
    for (auto& state : states) {
      auto it = std::find_if(direction_map.begin(), direction_map.end(),
        [&](const auto& pair) { return pair.second == state.name; });
      if (it != direction_map.end() && it->first == direction) {
        state.name = name;
        break;
      }
    }
  }

  sprite_sheet_controller_t& sprite_sheet_controller_t::set_direction_animation(uint8_t direction, const std::string& name) {
    direction_map[direction] = name;
    return *this;
  }

  void sprite_sheet_controller_t::use_preset_2d() {
    enable_directional({});
  }
}

#if defined(FAN_JSON)
  fan::graphics::shapes::shape_t::operator fan::json() {
    fan::json out;
    fan::graphics::shape_to_json(*this, &out);
    return out;
  }
  fan::graphics::shapes::shape_t::operator std::string() {
    fan::json out;
    fan::graphics::shape_to_json(*this, &out);
    return out.dump(2);
  }
  fan::graphics::shapes::shape_t::shape_t(const fan::json& json) : fan::graphics::shapes::shape_t() {
    fan::graphics::json_to_shape(json, this);
  }
  fan::graphics::shapes::shape_t::shape_t(const std::string& json_string) : fan::graphics::shapes::shape_t() {
    *this = fan::json::parse(json_string);
  }
  fan::graphics::shapes::shape_t& fan::graphics::shapes::shape_t::operator=(const fan::json& json) {
    fan::graphics::json_to_shape(json, this);
    return *this;
  }
  fan::graphics::shapes::shape_t& fan::graphics::shapes::shape_t::operator=(const std::string& json_string) {
    return fan::graphics::shapes::shape_t::operator=(fan::json::parse(json_string));
  }
#endif
#endif