module;

#if defined(FAN_OPENGL)
  #include <fan/utility.h>
#endif

module fan.texture_pack.tp0;

#if defined(FAN_OPENGL)

import fan.graphics.webp;
import fan.io.file;
import fan.print.error;
import fan.memory;
import fan.types.fstring;
import fan.graphics.image_load;

namespace fan::graphics {

  #include <fan/fan_bll_preset.h>
  #define BLL_set_prefix texture_unique_map
  #define BLL_set_Language 1
  #define BLL_set_Usage 1
  #define BLL_set_NodeDataType texture_pack::unique_data_t
  #define BLL_set_type_node std::uint32_t
  #include <BLL/BLL.h>

  struct internal_state_t {
    struct pack_t {
      struct internal_texture_t {
        std::shared_ptr<internal_texture_t> d[2];
        fan::vec2ui position;
        fan::vec2ui size;
      };
      internal_texture_t root;
      fan::vec2ui pack_size;
      struct texture_t {
        fan::vec2ui position;
        fan::vec2ui size;
        std::string image_name;
      };
      std::vector<texture_t> texture_list;
      std::uint32_t visual_output;
      std::uint32_t min_filter;
      std::uint32_t mag_filter;
      std::uint32_t group_id;
      std::vector<std::uint8_t> pixel_data;
    };

    struct texture_t {
      fan::vec2ui size;
      std::vector<std::uint8_t> decoded_data;
      std::string image_name;
      std::uint32_t visual_output;
      std::uint32_t min_filter;
      std::uint32_t mag_filter;
      std::uint32_t group_id;
    };

    std::vector<texture_t> texture_list;
    std::vector<pack_t> pack_list;
    std::vector<pack_t> loaded_pack;

    fan::vec2ui preferred_pack_size;
    std::uint32_t visual_output;
    std::uint32_t min_filter;
    std::uint32_t mag_filter;

    pack_t::internal_texture_t* push(pack_t::internal_texture_t* node, const fan::vec2ui& size) {
      if (node->d[0] || node->d[1]) {
        if (node->d[0]) {
          pack_t::internal_texture_t* newNode = push(node->d[0].get(), size);
          if (newNode) return newNode;
        }
        if (node->d[1]) {
          pack_t::internal_texture_t* newNode = push(node->d[1].get(), size);
          if (newNode) return newNode;
        }
        return nullptr;
      }

      if (size.x > node->size.x || size.y > node->size.y) {
        return nullptr;
      }

      int w = node->size.x - size.x;
      int h = node->size.y - size.y;
      node->d[0] = std::make_shared<pack_t::internal_texture_t>();
      node->d[1] = std::make_shared<pack_t::internal_texture_t>();
      
      if (w <= h) {
        node->d[0]->position.x = node->position.x + size.x;
        node->d[0]->position.y = node->position.y;
        node->d[0]->size.x = w;
        node->d[0]->size.y = size.y;

        node->d[1]->position.x = node->position.x;
        node->d[1]->position.y = node->position.y + size.y;
        node->d[1]->size.x = node->size.x;
        node->d[1]->size.y = h;
      } else {
        node->d[0]->position.x = node->position.x;
        node->d[0]->position.y = node->position.y + size.y;
        node->d[0]->size.x = size.x;
        node->d[0]->size.y = h;

        node->d[1]->position.x = node->position.x + size.x;
        node->d[1]->position.y = node->position.y;
        node->d[1]->size.x = w;
        node->d[1]->size.y = node->size.y;
      }
      node->size = size;
      return node;
    }
  };

  texture_pack::internal_t::internal_t() {
    internal_state = new internal_state_t();
  }

  texture_pack::internal_t::~internal_t() {
    delete static_cast<internal_state_t*>(internal_state);
  }

  void texture_pack::internal_t::open() {
    open(open_properties_t{});
  }

  void texture_pack::internal_t::open(const open_properties_t& op) {
    auto* s = static_cast<internal_state_t*>(internal_state);
    s->preferred_pack_size = op.preferred_pack_size;
    s->visual_output = op.visual_output;
    s->min_filter = op.min_filter;
    s->mag_filter = op.mag_filter;
  }

  void texture_pack::internal_t::close() {}

  std::size_t texture_pack::internal_t::push_pack(const pack_properties_t& p) {
    auto* s = static_cast<internal_state_t*>(internal_state);
    internal_state_t::pack_t pack;
    pack.pack_size = p.pack_size;
    pack.root.d[0] = pack.root.d[1] = 0;
    pack.root.position = 0;
    pack.root.size = p.pack_size;
    pack.visual_output = p.visual_output;
    pack.min_filter = p.min_filter;
    pack.mag_filter = p.mag_filter;
    pack.group_id = p.group_id;
    s->pack_list.push_back(pack);
    return s->pack_list.size() - 1;
  }

  std::size_t texture_pack::internal_t::push_pack() {
    auto* s = static_cast<internal_state_t*>(internal_state);
    pack_properties_t p;
    p.pack_size = s->preferred_pack_size;
    p.visual_output = s->visual_output;
    p.min_filter = s->min_filter;
    p.mag_filter = s->mag_filter;
    p.group_id = -1;
    return push_pack(p);
  }

  bool texture_pack::internal_t::push_texture(const std::string& image_path, const texture_properties_t& texture_properties) {
    auto* s = static_cast<internal_state_t*>(internal_state);
    if (texture_properties.image_name.empty()) return 1;

    fan::image::info_t image_info;
    if (fan::image::load(image_path, &image_info)) return 1;

    if (image_info.size.x % 2 != 0 || image_info.size.y % 2 != 0) {
      fan::image::free(&image_info);
      return 1;
    }

    std::erase_if(s->texture_list, [&](const auto& t) { return t.image_name == texture_properties.image_name; });

    internal_state_t::texture_t t;
    t.size = image_info.size;
    t.decoded_data.resize(t.size.multiply() * 4);

    fan::image::convert_channels(
      static_cast<const std::uint8_t*>(image_info.data), 
      t.decoded_data.data(), 
      t.size.multiply(), 
      image_info.channels, 
      4
    );

    fan::image::free(&image_info);
    
    t.image_name = texture_properties.image_name;
    t.visual_output = texture_properties.visual_output;
    t.min_filter = texture_properties.min_filter;
    t.mag_filter = texture_properties.mag_filter;
    t.group_id = texture_properties.group_id;

    s->texture_list.push_back(std::move(t));
    return 0;
  }

  bool texture_pack::internal_t::push_texture(fan::graphics::image_t image, const texture_properties_t& texture_properties) {
    auto* s = static_cast<internal_state_t*>(internal_state);
    if (texture_properties.image_name.empty()) return 1;

    for (std::uint32_t gti = 0; gti < s->texture_list.size(); gti++) {
      if (s->texture_list[gti].image_name == texture_properties.image_name) {
        s->texture_list.erase(s->texture_list.begin() + gti);
        break;
      }
    }

    auto data = fan::graphics::ctx()->image_get_pixel_data(
      fan::graphics::ctx(), 
      image, 
      fan::graphics::image_format_e::rgba_unorm,
      texture_properties.uv_pos, 
      texture_properties.uv_size
    );

    fan::vec2ui image_size = image.get_size();
    image_size = {
      (std::uint32_t)(image_size.x * texture_properties.uv_size.x),
      (std::uint32_t)(image_size.y * texture_properties.uv_size.y)
    };

    if ((int)image_size.x % 2 != 0 || (int)image_size.y % 2 != 0) return 1;

    internal_state_t::texture_t t;
    t.size = image_size;
    t.decoded_data.resize(t.size.multiply() * 4);
    std::memcpy(t.decoded_data.data(), data.data(), t.size.multiply() * 4);
    t.image_name = texture_properties.image_name;
    t.visual_output = texture_properties.visual_output;
    t.min_filter = texture_properties.min_filter;
    t.mag_filter = texture_properties.mag_filter;
    t.group_id = texture_properties.group_id;

    s->texture_list.push_back(t);
    return 0;
  }

  void texture_pack::internal_t::process() {
    auto* s = static_cast<internal_state_t*>(internal_state);
    s->pack_list.clear();
    const std::uint32_t PadPixel = 8;

    for (std::uint32_t ci = 0; ci < s->texture_list.size(); ci++) {
      fan::vec2ui size = s->texture_list[ci].size;
      texture_properties_t texture_properties;
      texture_properties.image_name = s->texture_list[ci].image_name;
      texture_properties.visual_output = s->texture_list[ci].visual_output;
      texture_properties.min_filter = s->texture_list[ci].min_filter;
      texture_properties.mag_filter = s->texture_list[ci].mag_filter;
      texture_properties.group_id = s->texture_list[ci].group_id;

      std::size_t selected_pack = -1;
      std::size_t pack_start = 0;
  gt_pack_search:
      for (std::size_t i = pack_start; i < s->pack_list.size(); i++) {
        std::uint32_t score = 0;
        if (texture_properties.visual_output != (std::uint32_t)-1) {
          if (s->pack_list[i].visual_output == texture_properties.visual_output) score++;
        }
        if (texture_properties.min_filter != (std::uint32_t)-1) {
          if (s->pack_list[i].min_filter == texture_properties.min_filter) score++;
        }
        if (texture_properties.mag_filter != (std::uint32_t)-1) {
          if (s->pack_list[i].mag_filter == texture_properties.mag_filter) score++;
        }
        if (texture_properties.group_id != (std::uint32_t)-1) {
          if (s->pack_list[i].group_id == texture_properties.group_id) score++;
        }
        if (size.x < s->pack_list[i].pack_size.x && size.y < s->pack_list[i].pack_size.y) {
          score++;
        }

        std::uint32_t needed_score = 1;
        needed_score += texture_properties.visual_output != (std::uint32_t)-1;
        needed_score += texture_properties.min_filter != (std::uint32_t)-1;
        needed_score += texture_properties.mag_filter != (std::uint32_t)-1;
        needed_score += texture_properties.group_id != (std::uint32_t)-1;

        if (score >= needed_score) {
          selected_pack = i;
          break;
        }
      }

      if (selected_pack == (std::size_t)-1) {
        if (size.x > s->preferred_pack_size.x || size.y > s->preferred_pack_size.y) {
          fan::throw_error("texture size is bigger than preferred_pack_size");
        }
        pack_properties_t p;
        p.pack_size = s->preferred_pack_size;
        p.visual_output = texture_properties.visual_output != (std::uint32_t)-1 ? texture_properties.visual_output : s->visual_output;
        p.min_filter = texture_properties.min_filter != (std::uint32_t)-1 ? texture_properties.min_filter : s->min_filter;
        p.mag_filter = texture_properties.mag_filter != (std::uint32_t)-1 ? texture_properties.mag_filter : s->mag_filter;
        p.group_id = texture_properties.group_id;
        selected_pack = push_pack(p);
      }

      fan::vec2ui push_size = size + PadPixel * 2;
      push_size = push_size.min(s->pack_list[selected_pack].pack_size);

      internal_state_t::pack_t::internal_texture_t* it = s->push(&s->pack_list[selected_pack].root, push_size);
      if (it == nullptr) {
        pack_start = selected_pack + 1ull;
        selected_pack = (std::size_t)-1;
        goto gt_pack_search;
      }
      internal_state_t::pack_t::texture_t texture;
      texture.position = it->position;
      texture.position += (push_size - size) / 2;
      texture.size = size;
      texture.image_name = texture_properties.image_name;
      s->pack_list[selected_pack].texture_list.push_back(texture);
    }

    for (std::uint32_t i = 0; i < s->pack_list.size(); i++) {
      std::size_t count = s->pack_list[i].texture_list.size();
      s->pack_list[i].pixel_data.resize(s->pack_list[i].pack_size.x * s->pack_list[i].pack_size.y * 4);
      memset(s->pack_list[i].pixel_data.data(), 0, s->pack_list[i].pack_size.x * s->pack_list[i].pack_size.y * 4);

      for (std::size_t j = 0; j < count; j++) {
        internal_state_t::pack_t::texture_t* t = &s->pack_list[i].texture_list[j];
        internal_state_t::texture_t* gt = nullptr;
        for (std::size_t gti = 0; gti < s->texture_list.size(); gti++) {
          if (s->texture_list[gti].image_name == t->image_name) {
            gt = &s->texture_list[gti];
            break;
          }
        }
        if (gt == nullptr) fan::throw_error("gt nullptr");

        for (std::uint32_t y = t->position.y; y < t->position.y + t->size.y; y++) {
          memcpy(
            s->pack_list[i].pixel_data.data() + (y * s->pack_list[i].pack_size.x + t->position.x) * 4,
            &gt->decoded_data[(y - t->position.y) * t->size.x * 4],
            t->size.x * 4
          );
        }

        fan::vec2ui Pad = t->size + PadPixel * 2;
        Pad = s->pack_list[i].pack_size.min(Pad) - t->size;
        fan::vec2ui pp = t->position - Pad / 2;
        fan::vec2ui ps = t->size + Pad;
        fan::vec2ui center = t->position + t->size / 2;

        for (std::uint32_t y = pp.y; y != pp.y + ps.y; y++) {
          for (std::uint32_t x = pp.x; x != pp.x + ps.x; x++) {
            if (y >= t->position.y && y < t->position.y + t->size.y &&
                x >= t->position.x && x < t->position.x + t->size.x) continue;

            fan::vec2 size_ratio = fan::vec2(t->size).square_normalize();
            fan::vec2 ray_angle = fan::vec2((fan::vec2si(x, y) - center) / size_ratio).square_normalize();
            fan::vec2si offset_from_center = ray_angle * (t->size / 2);
            if (offset_from_center.x > 0) offset_from_center.x--;
            if (offset_from_center.y > 0) offset_from_center.y--;
            fan::vec2ui from = center + offset_from_center;

            s->pack_list[i].pixel_data[(y * s->pack_list[i].pack_size.x + x) * 4 + 0] = s->pack_list[i].pixel_data[(from.y * s->pack_list[i].pack_size.x + from.x) * 4 + 0];
            s->pack_list[i].pixel_data[(y * s->pack_list[i].pack_size.x + x) * 4 + 1] = s->pack_list[i].pixel_data[(from.y * s->pack_list[i].pack_size.x + from.x) * 4 + 1];
            s->pack_list[i].pixel_data[(y * s->pack_list[i].pack_size.x + x) * 4 + 2] = s->pack_list[i].pixel_data[(from.y * s->pack_list[i].pack_size.x + from.x) * 4 + 2];
            s->pack_list[i].pixel_data[(y * s->pack_list[i].pack_size.x + x) * 4 + 3] = s->pack_list[i].pixel_data[(from.y * s->pack_list[i].pack_size.x + from.x) * 4 + 3];
          }
        }
      }
    }
  }

  void texture_pack::internal_t::save_compiled(const std::string& filename) {
    auto* s = static_cast<internal_state_t*>(internal_state);
    fan::io::file::file_t* f;
    fan::io::file::properties_t fp;
    fp.mode = "w+b";
    if (fan::io::file::open(&f, filename, fp)) {
      fan::throw_error(std::string("failed to open file:") + filename);
    }

    fan::write_to_file(f, s->pack_list.size());

    for (std::size_t i = 0; i < s->pack_list.size(); i++) {
      fan::write_to_file(f, s->pack_list[i].texture_list.size());
      for (std::size_t k = 0; k < s->pack_list[i].texture_list.size(); k++) {
        internal_state_t::pack_t::texture_t& t = s->pack_list[i].texture_list[k];
        fan::write_to_file(f, t.image_name);
        fan::write_to_file(f, t.position);
        fan::write_to_file(f, t.size);
      }
      std::uint8_t* ptr;
      std::uint32_t ptr_size = (std::uint32_t)fan::webp::encode_lossless_rgba(s->pack_list[i].pixel_data.data(), s->pack_list[i].pack_size, &ptr);
      fan::write_to_file(f, std::vector<std::uint8_t>(ptr, ptr + ptr_size));
      fan::webp::free_image(ptr);
      fan::write_to_file(f, s->pack_list[i].visual_output);
      fan::write_to_file(f, s->pack_list[i].min_filter);
      fan::write_to_file(f, s->pack_list[i].mag_filter);
    }
    fan::io::file::close(f);
  }

  void texture_pack::internal_t::load_compiled(const char* filename) { }

  std::size_t texture_pack::internal_t::size() const {
    return static_cast<internal_state_t*>(internal_state)->pack_list.size();
  }

  // -------------------------------------------------------------
  // texture_pack_t 
  // -------------------------------------------------------------

  struct tp_runtime_state_t {
    struct single_texturepack_decoded_t {
      std::uint32_t minor_count;
      texture_pack_t::texture_minor_decoded_t texture_minor_list[texture_pack_t::MAX_TEXTURE_MINOR];
      std::uint32_t image_list_id;
    };
    std::vector<single_texturepack_decoded_t> texture_major_list;
    std::vector<texture_pack_t::pixel_data_t> image_list;
    texture_unique_map_t unique_map;
    std::unordered_map<std::string, fan::graphics::texture_pack::unique_t> name_to_unique;
    std::unordered_map<std::uint64_t, fan::graphics::texture_pack::unique_t> hash_to_unique;
  };

  texture_pack_t::texture_pack_t() {
    internal_state = new tp_runtime_state_t();
  }

  texture_pack_t::texture_pack_t(const std::string& filename, const std::source_location& callers_path) {
    internal_state = new tp_runtime_state_t();
    open_compiled(filename, callers_path);
  }

  texture_pack_t::~texture_pack_t() {
    delete static_cast<tp_runtime_state_t*>(internal_state);
  }

  void texture_pack_t::open_compiled(const std::string& filename, const std::source_location& callers_path) {
    open_compiled(filename, fan::graphics::image_presets::pixel_art(), callers_path);
  }

  void texture_pack_t::open_compiled(const std::string& filename, fan::graphics::image_load_properties_t lp, const std::source_location& callers_path) {
    auto* s = static_cast<tp_runtime_state_t*>(internal_state);
    s->texture_major_list.clear();
    s->image_list.clear();
    s->unique_map.Clear();
    s->name_to_unique.clear();
    s->hash_to_unique.clear();

    file_path = filename;
    std::string in;
    fan::io::file::read(fan::io::file::find_relative_path(filename, callers_path), &in);

    std::size_t offset = 0;
    std::size_t pack_list_size = fan::string_read_data<std::size_t>(in, offset);

    s->image_list.resize(pack_list_size);
    s->texture_major_list.resize(pack_list_size);

    for (std::size_t i = 0; i < pack_list_size; i++) {
      std::size_t texture_list_size = fan::string_read_data<std::size_t>(in, offset);
      s->texture_major_list[i].minor_count = texture_list_size;
      s->texture_major_list[i].image_list_id = i;

      for (std::size_t k = 0; k < texture_list_size; k++) {
        texture_pack_t::texture_minor_decoded_t texture;
        texture.name = fan::string_read_data<std::string>(in, offset);
        texture.position = fan::string_read_data<fan::vec2ui>(in, offset);
        texture.size = fan::string_read_data<fan::vec2ui>(in, offset);
        
        auto it = s->unique_map.NewNodeLast();
        s->unique_map[it].major = i;
        s->unique_map[it].minor = k;
        
        texture.unique_id.id = *(std::uint32_t*)&it;
        s->texture_major_list[i].texture_minor_list[k] = texture;
        s->name_to_unique[texture.name] = texture.unique_id;
        s->hash_to_unique[fan::get_hash(texture.name.c_str())] = texture.unique_id;
      }

      std::vector<std::uint8_t> pixel_data = fan::string_read_data<std::vector<std::uint8_t>>(in, offset);
      fan::webp::info_t image_info;
      if (fan::webp::decode(pixel_data.data(), pixel_data.size(), &image_info)) {
        fan::throw_error_impl();
      }
      image_info.type = fan::image::image_type_e::webp;
      image_info.channels = 4;
      s->image_list[i].image = fan::graphics::ctx()->image_load_info_props(
        fan::graphics::ctx(), *(fan::image::info_t*)&image_info, lp
      );
      fan::webp::free_image(image_info.data);

      fan::string_read_data<std::uint32_t>(in, offset);
      fan::string_read_data<std::uint32_t>(in, offset);
      fan::string_read_data<std::uint32_t>(in, offset);
    }
  }

  texture_pack_t::pixel_data_t& texture_pack_t::get_pixel_data(fan::graphics::texture_pack::unique_t unique) {
    auto* s = static_cast<tp_runtime_state_t*>(internal_state);
    return s->image_list[s->unique_map[*(texture_unique_map_NodeReference_t*)&unique.id].major];
  }

  void texture_pack_t::iterate_loaded_images_raw(void* user_data, void(*cb)(const texture_minor_decoded_t&, void*)) {
    auto* s = static_cast<tp_runtime_state_t*>(internal_state);
    for (std::uint32_t i = 0; i < s->texture_major_list.size(); i++) {
      for (std::uint32_t j = 0; j < s->texture_major_list[i].minor_count; j++) {
        cb(s->texture_major_list[i].texture_minor_list[j], user_data);
      }
    }
  }

  texture_pack_t::operator bool() const {
    return size();
  }

  std::size_t texture_pack_t::size() const {
    auto* s = static_cast<tp_runtime_state_t*>(internal_state);
    return s->texture_major_list.size();
  }

  texture_pack_t::texture_minor_decoded_t texture_pack_t::operator[](fan::graphics::texture_pack::unique_t unique_id) {
    if (unique_id.iic()) return texture_minor_decoded_t{};
    auto* s = static_cast<tp_runtime_state_t*>(internal_state);
    auto& entry = s->unique_map[*(texture_unique_map_NodeReference_t*)&unique_id.id];
    return s->texture_major_list[entry.major].texture_minor_list[entry.minor];
  }

  fan::graphics::texture_pack::unique_t texture_pack_t::operator[](const std::string& name) {
    ti_t ti;
    qti(name, &ti);
    return ti.unique_id;
  }

  bool texture_pack_t::qti(const std::string& name, ti_t* ti) {
    auto* s = static_cast<tp_runtime_state_t*>(internal_state);
    auto it = s->name_to_unique.find(name);
    if (it == s->name_to_unique.end()) return 1;
    auto unique_id = it->second;
    if (unique_id.iic()) return 1;
    auto& entry = s->unique_map[*(texture_unique_map_NodeReference_t*)&unique_id.id];
    auto& minor = s->texture_major_list[entry.major].texture_minor_list[entry.minor];
    ti->unique_id = unique_id;
    ti->position = minor.position;
    ti->size = minor.size;
    ti->image = get_pixel_data(ti->unique_id).image;
    return 0;
  }

  bool texture_pack_t::qti(std::uint64_t hash, ti_t* ti) {
    auto* s = static_cast<tp_runtime_state_t*>(internal_state);
    auto it = s->hash_to_unique.find(hash);
    if (it == s->hash_to_unique.end()) return 1;
    auto unique_id = it->second;
    if (unique_id.iic()) return 1;
    auto& entry = s->unique_map[*(texture_unique_map_NodeReference_t*)&unique_id.id];
    auto& minor = s->texture_major_list[entry.major].texture_minor_list[entry.minor];
    ti->unique_id = unique_id;
    ti->position = minor.position;
    ti->size = minor.size;
    ti->image = get_pixel_data(ti->unique_id).image;
    return 0;
  }
}
#endif