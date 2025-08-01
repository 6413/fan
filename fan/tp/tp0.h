struct texture_pack_unique_data_t {
  uint32_t major;
  uint32_t minor;
};

#include <fan/fan_bll_preset.h>
#define BLL_set_prefix texture_unique_map
#define BLL_set_Language 1
#define BLL_set_Usage 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeDataType texture_pack_unique_data_t
#define BLL_set_type_node uint32_t
#include <BLL/BLL.h>

using texture_pack_unique_t = texture_unique_map_NodeReference_t;

struct ti_t {

  bool qti(auto* texture_pack, const std::string& name) {
    return texture_pack->qti(name, this);
  }
  bool qti(auto* texture_pack, uint64_t hash) {
    return texture_pack->qti(hash, this);
  }

  bool valid() const {
    return image.iic() == false;
  }

  texture_pack_unique_t unique_id;
  fan::vec2 position;
  fan::vec2 size;
  fan::graphics::image_nr_t image;
};

struct texture_packe0 {

  struct texture_t {
    fan::vec2ui size;
    std::vector<uint8_t> decoded_data;
    std::string image_name;
    uint32_t visual_output;
    uint32_t min_filter;
    uint32_t mag_filter;
    uint32_t group_id;
  };

  std::vector<texture_t> texture_list;

  struct open_properties_t {
    open_properties_t() {}

    fan::vec2ui preferred_pack_size = 1024;
    uint32_t visual_output = fan::opengl::context_t::image_load_properties_defaults::visual_output;
    uint32_t min_filter = GL_LINEAR_MIPMAP_LINEAR;
    uint32_t mag_filter = fan::opengl::context_t::image_load_properties_defaults::mag_filter;
  };

  struct pack_properties_t {
    pack_properties_t() {}

    fan::vec2ui pack_size;
    uint32_t visual_output;
    uint32_t min_filter;
    uint32_t mag_filter;
    uint32_t group_id;
  };

  struct texture_properties_t {
    texture_properties_t() {}

    fan::vec2 uv_pos = 0;
    fan::vec2 uv_size = 1;

    std::string image_name;
    uint32_t visual_output = -1;
    uint32_t min_filter = -1;
    uint32_t mag_filter = -1;
    uint32_t group_id = -1;
  };

  fan::vec2ui preferred_pack_size;
  uint32_t visual_output;

  uint32_t min_filter;
  uint32_t mag_filter;

  void open(const open_properties_t& op = open_properties_t()) {
    preferred_pack_size = op.preferred_pack_size;
    visual_output = op.visual_output;
    min_filter = op.min_filter;
    mag_filter = op.mag_filter;
  }
  void close() {
  }

  std::size_t push_pack(const pack_properties_t& p) {
    pack_t pack;
    pack.pack_size = p.pack_size;
    pack.root.d[0] = pack.root.d[1] = 0;
    pack.root.position = 0;
    pack.root.size = p.pack_size;
    pack.visual_output = p.visual_output;
    pack.min_filter = p.min_filter;
    pack.mag_filter = p.mag_filter;
    pack.group_id = p.group_id;
    pack_list.push_back(pack);
    return pack_list.size() - 1;
  }
  std::size_t push_pack() {
    pack_properties_t p;
    p.pack_size = preferred_pack_size;
    p.visual_output = visual_output;
    p.min_filter = min_filter;
    p.mag_filter = mag_filter;
    p.group_id = -1;
    return push_pack(p);
  }

  bool push_texture(const std::string& image_path, const texture_properties_t& texture_properties) {

    if (texture_properties.image_name.empty()) {
      fan::print_warning("texture properties name empty");
      return 1;
    }

    for (uint32_t gti = 0; gti < texture_list.size(); gti++) {
      if (texture_list[gti].image_name == texture_properties.image_name) {
        texture_list.erase(texture_list.begin() + gti);
        break;
      }
    }

    fan::webp::info_t image_info;
    if (fan::webp::load(image_path, &image_info)) {
      fan::print_warning("failed to load");
      return 1;
    }

    if (image_info.size.x % 2 != 0 || image_info.size.y % 2 != 0) {
      fan::webp::free_image(image_info.data);
      fan::print_warning("failed to load, image size is not divideable by 2");
      fan::print(texture_properties.image_name, image_info.size);
      return 1;
    }

    texture_t t;
    t.size = image_info.size;
    t.decoded_data.resize(t.size.multiply() * 4);
    std::memcpy(t.decoded_data.data(), image_info.data, t.size.multiply() * 4);
    fan::webp::free_image(image_info.data);
    t.image_name = texture_properties.image_name;
    t.visual_output = texture_properties.visual_output;
    t.min_filter = texture_properties.min_filter;
    t.mag_filter = texture_properties.mag_filter;
    t.group_id = texture_properties.group_id;

    texture_list.push_back(t);
    return 0;
  }

  bool push_texture(fan::graphics::image_nr_t image, const texture_properties_t& texture_properties) {

    if (texture_properties.image_name.empty()) {
      fan::print_warning("texture properties name empty");
      return 1;
    }

    for (uint32_t gti = 0; gti < texture_list.size(); gti++) {
      if (texture_list[gti].image_name == texture_properties.image_name) {
        texture_list.erase(texture_list.begin() + gti);
        break;
      }
    }

    auto data = gloco->image_get_pixel_data(image, GL_RGBA, texture_properties.uv_pos, texture_properties.uv_size);

    fan::vec2ui image_size = 0;
    auto& image_data = gloco->image_get_data(image);
    image_size = {
      (uint32_t)(image_data.size.x * texture_properties.uv_size.x),
      (uint32_t)(image_data.size.y * texture_properties.uv_size.y)
    };

    if ((int)image_size.x % 2 != 0 || (int)image_size.y % 2 != 0) {
      fan::print_warning("failed to load, image size is not divideable by 2");
      fan::print(texture_properties.image_name, image_size);
      return 1;
    }

    texture_t t;
    t.size = image_size;
    t.decoded_data.resize(t.size.multiply() * 4);
    std::memcpy(t.decoded_data.data(), data.data(), t.size.multiply() * 4);
    t.image_name = texture_properties.image_name;
    t.visual_output = texture_properties.visual_output;
    t.min_filter = texture_properties.min_filter;
    t.mag_filter = texture_properties.mag_filter;
    t.group_id = texture_properties.group_id;

    texture_list.push_back(t);
    return 0;
  }

  void process() {
    pack_list.clear();

    const uint32_t PadPixel = 8;

    for (uint32_t ci = 0; ci < texture_list.size(); ci++) {

      fan::vec2ui size = texture_list[ci].size;

      texture_properties_t texture_properties;
      texture_properties.image_name = texture_list[ci].image_name;
      texture_properties.visual_output = texture_list[ci].visual_output;
      texture_properties.min_filter = texture_list[ci].min_filter;
      texture_properties.mag_filter = texture_list[ci].mag_filter;
      texture_properties.group_id = texture_list[ci].group_id;

      std::size_t selected_pack = -1;
      std::size_t pack_start = 0;
    gt_pack_search:
      for (std::size_t i = pack_start; i < this->size(); i++) {
        uint32_t score = 0;

        if (texture_properties.visual_output != (uint32_t)-1) {
          if (pack_list[i].visual_output == texture_properties.visual_output) {
            score++;
          }
        }
        if (texture_properties.min_filter != (uint32_t)-1) {
          if (pack_list[i].min_filter == texture_properties.min_filter) {
            score++;
          }
        }
        if (texture_properties.mag_filter != (uint32_t)-1) {
          if (pack_list[i].mag_filter == texture_properties.mag_filter) {
            score++;
          }
        }
        if (texture_properties.group_id != (uint32_t)-1) {
          if (pack_list[i].group_id == texture_properties.group_id) {
            score++;
          }
        }
        if (size.x < pack_list[i].pack_size.x && size.y < pack_list[i].pack_size.y) {
          score++;
        }

        uint32_t needed_score = 1;
        needed_score += texture_properties.visual_output != (uint32_t)-1;
        needed_score += texture_properties.min_filter != (uint32_t)-1;
        needed_score += texture_properties.mag_filter != (uint32_t)-1;
        needed_score += texture_properties.group_id != (uint32_t)-1;

        if (score >= needed_score) {
          selected_pack = i;
          break;
        }
      }
      if (selected_pack == (std::size_t)-1) {
        if (size.x > preferred_pack_size.x || size.y > preferred_pack_size.y) {
          fan::throw_error("texture size is bigger than preferred_pack_size");
        }

        pack_properties_t p;
        p.pack_size = preferred_pack_size;
        if (texture_properties.visual_output != (uint32_t)-1) {
          p.visual_output = texture_properties.visual_output;
        }
        else {
          p.visual_output = visual_output;
        }
        if (texture_properties.min_filter != (uint32_t)-1) {
          p.min_filter = texture_properties.min_filter;
        }
        else {
          p.min_filter = min_filter;
        }
        if (texture_properties.mag_filter != (uint32_t)-1) {
          p.mag_filter = texture_properties.mag_filter;
        }
        else {
          p.mag_filter = mag_filter;
        }
        p.group_id = texture_properties.group_id;
        selected_pack = push_pack(p);
      }

      fan::vec2ui push_size = size + PadPixel * 2;
      push_size = push_size.min(pack_list[selected_pack].pack_size);

      pack_t::internal_texture_t* it = push(&pack_list[selected_pack].root, push_size);
      if (it == nullptr) {
        pack_start = selected_pack + 1ull;
        selected_pack = (std::size_t)-1;
        goto gt_pack_search;
      }
      pack_t::texture_t texture;
      texture.position = it->position;
      texture.position += (push_size - size) / 2;
      texture.size = size;
      texture.image_name = texture_properties.image_name;
      pack_list[selected_pack].texture_list.push_back(texture);
    }

    for (uint32_t i = 0; i < pack_list.size(); i++) {
      std::size_t count = pack_list[i].texture_list.size();

      pack_list[i].pixel_data.resize(pack_list[i].pack_size.x * pack_list[i].pack_size.y * 4);

      memset(
        pack_list[i].pixel_data.data(),
        0,
        pack_list[i].pack_size.x * pack_list[i].pack_size.y * 4
      );

      for (std::size_t j = 0; j < count; j++) {
        pack_t::texture_t* t = &pack_list[i].texture_list[j];
        texture_t* gt = nullptr;
        for (std::size_t gti = 0; gti < texture_list.size(); gti++) {
          if (texture_list[gti].image_name == t->image_name) {
            gt = &texture_list[gti];
            break;
          }
        }
        if (gt == nullptr) {
          fan::throw_error("gt nullptr");
        }
        for (uint32_t y = t->position.y; y < t->position.y + t->size.y; y++) {
          memcpy(
            pack_list[i].pixel_data.data() + (y * pack_list[i].pack_size.x + t->position.x) * 4,
            &gt->decoded_data[(y - t->position.y) * t->size.x * 4],
            t->size.x * 4
          );
        }
        {
          // find pad by size
          fan::vec2ui Pad = t->size + PadPixel * 2;
          Pad = pack_list[i].pack_size.min(Pad) - t->size;

          fan::vec2ui pp = t->position - Pad / 2;
          fan::vec2ui ps = t->size + Pad;

          fan::vec2ui center = t->position + t->size / 2;

          for (uint32_t y = pp.y; y != pp.y + ps.y; y++) {
            for (uint32_t x = pp.x; x != pp.x + ps.x; x++) {
              if (
                y >= t->position.y && y < t->position.y + t->size.y &&
                x >= t->position.x && x < t->position.x + t->size.x
                ) {
                continue;
              }

              fan::vec2 size_ratio = fan::vec2(t->size).square_normalize();
              fan::vec2 ray_angle = fan::vec2((fan::vec2si(x, y) - center) / size_ratio).square_normalize();
              fan::vec2si offset_from_center = ray_angle * (t->size / 2);
              if (offset_from_center.x > 0) {
                offset_from_center.x--;
              }
              if (offset_from_center.y > 0) {
                offset_from_center.y--;
              }
              fan::vec2ui from = center + offset_from_center;

              pack_list[i].pixel_data[(y * pack_list[i].pack_size.x + x) * 4 + 0] = pack_list[i].pixel_data[(from.y * pack_list[i].pack_size.x + from.x) * 4 + 0];
              pack_list[i].pixel_data[(y * pack_list[i].pack_size.x + x) * 4 + 1] = pack_list[i].pixel_data[(from.y * pack_list[i].pack_size.x + from.x) * 4 + 1];
              pack_list[i].pixel_data[(y * pack_list[i].pack_size.x + x) * 4 + 2] = pack_list[i].pixel_data[(from.y * pack_list[i].pack_size.x + from.x) * 4 + 2];
              pack_list[i].pixel_data[(y * pack_list[i].pack_size.x + x) * 4 + 3] = pack_list[i].pixel_data[(from.y * pack_list[i].pack_size.x + from.x) * 4 + 3];
            }
          }
        }
      }
    }
  }

  void save_compiled(const std::string& filename) {
    fan::io::file::file_t* f;
    fan::io::file::properties_t fp;
    fp.mode = "w+b";
    if (fan::io::file::open(&f, filename, fp)) {
      fan::throw_error(std::string("failed to open file:") + filename);
    }

    fan::write_to_file(f, pack_list.size());

    for (std::size_t i = 0; i < pack_list.size(); i++) {
      fan::write_to_file(f, pack_list[i].texture_list.size());

      for (std::size_t k = 0; k < pack_list[i].texture_list.size(); k++) {
        pack_t::texture_t& t = pack_list[i].texture_list[k];
        fan::write_to_file(f, t.image_name);
        fan::write_to_file(f, t.position);
        fan::write_to_file(f, t.size);
      }

      uint8_t* ptr;
      uint32_t ptr_size = (uint32_t)fan::webp::encode_lossless_rgba(pack_list[i].pixel_data.data(), pack_list[i].pack_size, &ptr);
      fan::write_to_file(f, std::vector<uint8_t>(ptr, ptr + ptr_size));
      fan::webp::free_image(ptr);
      fan::write_to_file(f, pack_list[i].visual_output);
      fan::write_to_file(f, pack_list[i].min_filter);
      fan::write_to_file(f, pack_list[i].mag_filter);
    }
    fan::print("file written to", filename);
    fan::io::file::close(f);
  }

  void load_compiled(const char* filename);

  std::size_t size() const {
    return pack_list.size();
  }

  fan::webp::info_t get_pixel_data(uint32_t pack_id) {
    fan::webp::info_t image_info;
    image_info.data = pack_list[pack_id].pixel_data.data();
    image_info.size = pack_list[pack_id].pack_size;
    return image_info;
  }

private:
  struct pack_t {
    struct internal_texture_t {
      std::shared_ptr<internal_texture_t> d[2];

      // The top-left coordinate of the rectangle.
      fan::vec2ui position;
      fan::vec2ui size;

      friend std::ostream& operator<<(std::ostream& os, const internal_texture_t& tex) {
        os << '{' << "\n position:" << tex.position << "\n size:" << tex.size << "\n}";
        return os;
      }
    };
    internal_texture_t root;
    fan::vec2ui pack_size;
    struct texture_t {
      fan::vec2ui position;
      fan::vec2ui size;
      std::string image_name;
    };
    std::vector<texture_t> texture_list;
    uint32_t visual_output;
    uint32_t min_filter;
    uint32_t mag_filter;
    uint32_t group_id;
    std::vector<uint8_t> pixel_data;
  };
  std::vector<pack_t> pack_list;
  std::vector<pack_t> loaded_pack;

public:

  void tree_debug(uint32_t pack_id, pack_t::internal_texture_t* node) {
    if (node->d[0]) {
      tree_debug(pack_id, node->d[0].get());
    }
    if (node->d[1]) {
      tree_debug(pack_id, node->d[1].get());
    }
  }

  void tree_debug(uint32_t pack_id) {
    tree_debug(pack_id, &pack_list[pack_id].root);
  }

private:

  unsigned long used_surface_area(const pack_t::internal_texture_t& node) const {
    if (node.d[0] || node.d[1]) {
      unsigned long usedSurfaceArea = node.size.x * node.size.y;
      if (node.d[0]) {
        usedSurfaceArea += used_surface_area(*node.d[0]);
      }
      if (node.d[1]) {
        usedSurfaceArea += used_surface_area(*node.d[1]);
      }
      return usedSurfaceArea;
    }
    return 0;
  }

  pack_t::internal_texture_t* push(pack_t::internal_texture_t* node, const fan::vec2ui& size) {
    if (node->d[0] || node->d[1]) {
      if (node->d[0]) {
        pack_t::internal_texture_t* newNode = push(node->d[0].get(), size);
        if (newNode)
          return newNode;
      }
      if (node->d[1]) {
        pack_t::internal_texture_t* newNode = push(node->d[1].get(), size);
        if (newNode)
          return newNode;
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
    }
    else {
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