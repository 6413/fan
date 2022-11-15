struct texture_packe0 {

  struct texture_t {
    fan::vec2ui size;
    std::vector<uint8_t> decoded_data;
    fan::string name;
    uint32_t visual_output;
    uint32_t filter;
    uint32_t group_id;
  };

  std::vector<texture_t> texture_list;

  struct open_properties_t {
    open_properties_t() {}

    fan::vec2ui preferred_pack_size = 1024;
    uint32_t visual_output = loco_t::image_t::load_properties_defaults::visual_output;
    uint32_t filter = loco_t::image_t::load_properties_defaults::filter;
  };

  struct pack_properties_t {
    pack_properties_t() {}

    fan::vec2ui pack_size;
    uint32_t visual_output;
    uint32_t filter;
    uint32_t group_id;
  };

  struct texture_properties_t {
    texture_properties_t() {}

    fan::string name;
    uint32_t visual_output = -1;
    uint32_t filter = -1;
    uint32_t group_id = -1;
  };

  fan::vec2ui preferred_pack_size;
  uint32_t visual_output;
  uint32_t filter;

  void open(const open_properties_t& op = open_properties_t()) {
    preferred_pack_size = op.preferred_pack_size;
    visual_output = op.visual_output;
    filter = op.filter;
  }
  void close() {
  }

  uint32_t push_pack(const pack_properties_t& p) {
    pack_t pack;
    pack.pack_size = p.pack_size;
    pack.root.d[0] = pack.root.d[1] = 0;
    pack.root.position = 0;
    pack.root.size = p.pack_size;
    pack.visual_output = p.visual_output;
    pack.filter = p.filter;
    pack.group_id = p.group_id;
    pack_list.push_back(pack);
    return pack_list.size() - 1;
  }
  uint32_t push_pack() {
    pack_properties_t p;
    p.pack_size = preferred_pack_size;
    p.visual_output = visual_output;
    p.filter = filter;
    p.group_id = -1;
    return push_pack(p);
  }

  bool push_texture(const fan::string& image_path, const texture_properties_t& texture_properties = texture_properties_t()) {

    if (texture_properties.name.empty()) {
      fan::print_warning("texture properties name empty");
      return 1;
    }

    for (uint32_t gti = 0; gti < texture_list.size(); gti++) {
      if (texture_list[gti].name == texture_properties.name) {
        texture_list.erase(texture_list.begin() + gti);
        break;
      }
    }

    fan::webp::image_info_t image_info;
    if (fan::webp::load(image_path, &image_info)) {
      fan::print_warning("failed to load");
      return 1;
    }

    texture_t t;
    t.size = image_info.size;
    t.decoded_data.resize(t.size.multiply() * 4);
    std::memcpy(t.decoded_data.data(), image_info.data, t.size.multiply() * 4);
    fan::webp::free_image(image_info.data);
    t.name = texture_properties.name;
    t.visual_output = texture_properties.visual_output;
    t.filter = texture_properties.filter;
    t.group_id = texture_properties.group_id;

    texture_list.push_back(t);
    return 0;
  }

  void process() {
    pack_list.clear();

    for (uint32_t ci = 0; ci < texture_list.size(); ci++) {

      fan::vec2ui size = texture_list[ci].size;

      texture_properties_t texture_properties;
      texture_properties.name = texture_list[ci].name;
      texture_properties.visual_output = texture_list[ci].visual_output;
      texture_properties.filter = texture_list[ci].filter;
      texture_properties.group_id = texture_list[ci].group_id;

      uint32_t selected_pack = -1;
      uint32_t pack_start = 0;
    gt_pack_search:
      for (uint32_t i = pack_start; i < this->size(); i++) {
        uint32_t score = 0;

        if (texture_properties.visual_output != -1) {
          if (pack_list[i].visual_output == texture_properties.visual_output) {
            score++;
          }
        }
        if (texture_properties.filter != -1) {
          if (pack_list[i].filter == texture_properties.filter) {
            score++;
          }
        }
        if (texture_properties.group_id != -1) {
          if (pack_list[i].group_id == texture_properties.group_id) {
            score++;
          }
        }
        uint32_t needed_score = 0;

        needed_score += texture_properties.visual_output != -1;
        needed_score += texture_properties.filter != -1;
        needed_score += texture_properties.group_id != -1;

        if (score >= needed_score) {
          selected_pack = i;
          break;
        }
      }
      if (selected_pack == -1) {
        pack_properties_t p;
        p.pack_size = preferred_pack_size;
        if (texture_properties.visual_output != -1) {
          p.visual_output = texture_properties.visual_output;
        }
        else {
          p.visual_output = visual_output;
        }
        if (texture_properties.filter != -1) {
          p.filter = texture_properties.filter;
        }
        else {
          p.filter = filter;
        }
        p.group_id = texture_properties.group_id;
        selected_pack = push_pack(p);
      }

      fan::vec2ui push_size = size;
      if (size.x != pack_list[selected_pack].pack_size.x) {
        push_size.x += 2;
      }
      if (size.y != pack_list[selected_pack].pack_size.y) {
        push_size.y += 2;
      }
      pack_t::internal_texture_t* it = push(&pack_list[selected_pack].root, push_size);
      if (it == nullptr) {
        if (push_size.x > pack_list[selected_pack].pack_size.x ||
          push_size.y > pack_list[selected_pack].pack_size.y) {
          fan::throw_error("too big");
        }
        pack_start = selected_pack + 1;
        selected_pack = -1;
        goto gt_pack_search;
      }
      pack_t::texture_t texture;
      texture.position = it->position;
      texture.size = size;
      if (texture.size.x != pack_list[selected_pack].pack_size.x) {
        texture.position.x++;
      }
      if (texture.size.y != pack_list[selected_pack].pack_size.y) {
        texture.position.y++;
      }
      texture.name = texture_properties.name;
      pack_list[selected_pack].texture_list.push_back(texture);
    }

    for (uint32_t i = 0; i < pack_list.size(); i++) {
      uint32_t count = pack_list[i].texture_list.size();

      pack_list[i].pixel_data.resize(pack_list[i].pack_size.x * pack_list[i].pack_size.y * 4);

      for (uint32_t j = 0; j < count; j++) {
        pack_t::texture_t* t = &pack_list[i].texture_list[j];
        texture_t *gt = nullptr;
        for (uint32_t gti = 0; gti < texture_list.size(); gti++) {
          if (texture_list[gti].name == t->name) {
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
          fan::vec2ui pp = t->position;
          if (pp.x != 0) {
            pp.x--;
          }
          if (pp.y != 0) {
            pp.y--;
          }
          fan::vec2ui ps = t->size + 2;
          if (ps.x > pack_list[i].pack_size.x) {
            continue;
          }
          if (ps.y > pack_list[i].pack_size.y) {
            continue;
          }

          for (uint32_t y = pp.y; y != pp.y + ps.y; y++) {
            for (uint32_t x = pp.x; x != pp.x + ps.x; x++) {

              static auto fill_pad = [](decltype(pack_list)* pack_list, uint32_t i, uint32_t x, uint32_t y, sint32_t px, sint32_t py) {

                switch (px) {
                  case -1: {
                    if (x == 0) {
                      return;
                    }
                    break;
                  }
                  case 1: {
                    if (x == (*pack_list)[i].pack_size.x) {
                      return;
                    }
                  }
                };
                switch (py) {
                  case -1: {
                    if (y == 0) {
                      return;
                    }
                    break;
                  }
                  case 1: {
                    if (y == (*pack_list)[i].pack_size.y) {
                      return;
                    }
                  }
                };

                (*pack_list)[i].pixel_data[(y * (*pack_list)[i].pack_size.x + x) * 4 + 0] = (*pack_list)[i].pixel_data[((y + py) * (*pack_list)[i].pack_size.x + x + px) * 4 + 0];
                (*pack_list)[i].pixel_data[(y * (*pack_list)[i].pack_size.x + x) * 4 + 1] = (*pack_list)[i].pixel_data[((y + py) * (*pack_list)[i].pack_size.x + x + px) * 4 + 1];
                (*pack_list)[i].pixel_data[(y * (*pack_list)[i].pack_size.x + x) * 4 + 2] = (*pack_list)[i].pixel_data[((y + py) * (*pack_list)[i].pack_size.x + x + px) * 4 + 2];
                (*pack_list)[i].pixel_data[(y * (*pack_list)[i].pack_size.x + x) * 4 + 3] = (*pack_list)[i].pixel_data[((y + py) * (*pack_list)[i].pack_size.x + x + px) * 4 + 3];
              };

              if (x == pp.x && y == pp.y) {
                fill_pad(&pack_list, i, x, y, 1, 1);
              }
              else if (x == pp.x + ps.x - 1 && y == pp.y) {
                fill_pad(&pack_list, i, x, y, -1, 1);
              }
              else if (x == pp.x + ps.x - 1 && y == pp.y + ps.y - 1) {
                fill_pad(&pack_list, i, x, y, -1, -1);
              }
              else if (x == pp.x && y == pp.y + ps.y - 1) {
                fill_pad(&pack_list, i, x, y, 1, -1);
              }
              else if (x == pp.x) {
                fill_pad(&pack_list, i, x, y, 1, 0);
              }
              else if (x == pp.x + ps.x - 1) {
                fill_pad(&pack_list, i, x, y, -1, 0);
              }
              else if (y == pp.y) {
                fill_pad(&pack_list, i, x, y, 0, 1);
              }
              else if (y == pp.y + ps.y - 1) {
                fill_pad(&pack_list, i, x, y, 0, -1);
              }
            }
          }
        }
      }
    }
  }

  void save(const char* filename) {
    FILE* f = fopen(filename, "w+b");
    if (!f) {
      fan::throw_error(fan::string("failed to open file:") + filename);
    }

    fan::throw_error("read comment xd");
    // where do we use this?
    uint32_t pack_amount = pack_list.size();

    fan::io::file::write(f, &visual_output, sizeof(uint32_t), 1);
    fan::io::file::write(f, &filter, sizeof(uint32_t), 1);

    uint32_t texture_list_size = texture_list.size();
    fan::io::file::write(f, &texture_list_size, sizeof(texture_list_size), 1);
    for (uint32_t i = 0; i < texture_list.size(); i++) {
      fan::io::file::write(f, &texture_list[i].size, sizeof(texture_list[i].size), 1);
      fan::io::file::write(f, texture_list[i].decoded_data.data(), texture_list[i].decoded_data.size(), 1);
      uint32_t name_s = texture_list[i].name.size();
      fan::io::file::write(f, &name_s, sizeof(name_s), 1);
      fan::io::file::write(f, texture_list[i].name.data(), texture_list[i].name.size(), 1);
      fan::io::file::write(f, &texture_list[i].visual_output, sizeof(texture_list[i].visual_output), 1);
      fan::io::file::write(f, &texture_list[i].filter, sizeof(texture_list[i].filter), 1);
      fan::io::file::write(f, &texture_list[i].group_id, sizeof(texture_list[i].group_id), 1);
    }
    fan::io::file::close(f);
  }
  void save_compiled(const char* filename) {
    fan::io::file::file_t* f;
    fan::io::file::properties_t fp;
    fp.mode = "w+b";
    if (fan::io::file::open(&f, filename, fp)) {
      fan::throw_error(fan::string("failed to open file:") + filename);
    }

    uint32_t pack_amount = pack_list.size();
    fan::io::file::write(f, &pack_amount, sizeof(pack_amount), 1);

    for (uint32_t i = 0; i < pack_amount; i++) {
      uint32_t count = pack_list[i].texture_list.size();
      fan::io::file::write(f, &count, sizeof(count), 1);

      for (uint32_t j = 0; j < count; j++) {
        pack_t::texture_t* t = &pack_list[i].texture_list[j];
        uint64_t hashed = fan::get_hash(t->name);
        fan::io::file::write(f, &hashed, sizeof(hashed), 1);
        fan::io::file::write(f, t->position.data(), sizeof(t->position), 1);
        fan::io::file::write(f, t->size.data(), sizeof(t->size), 1);
      }

      uint8_t* ptr;
      uint32_t ptr_size = fan::webp::encode_lossless_rgba(pack_list[i].pixel_data.data(), pack_list[i].pack_size, &ptr);
      fan::io::file::write(f, &ptr_size, sizeof(ptr_size), 1);
      fan::io::file::write(f, ptr, ptr_size, 1);
      fan::webp::free_image(ptr);
      fan::io::file::write(f, &pack_list[i].visual_output, sizeof(uint32_t), 1);
      fan::io::file::write(f, &pack_list[i].filter, sizeof(uint32_t), 1);
    }
    fan::io::file::close(f);
  }

  void load(const char* filename) {
    fan::string data;
    fan::io::file::read(filename, &data);
    if (data.empty()) {
      return;
    }
    uint32_t data_index = 0;

    visual_output = *(uint32_t*)&data[data_index];
    data_index += sizeof(visual_output);

    filter = *(uint32_t*)&data[data_index];
    data_index += sizeof(filter);

    uint32_t texture_amount = *(uint32_t*)&data[data_index];
    data_index += sizeof(texture_amount);

    for (uint32_t i = 0; i < texture_amount; i++) {
      texture_t t;
      t.size.x = *(uint32_t*)&data[data_index];
      data_index += sizeof(uint32_t);
      t.size.y = *(uint32_t*)&data[data_index];
      data_index += sizeof(uint32_t);
      uint32_t r = t.size.multiply() * 4;
      t.decoded_data.resize(r);
      std::memcpy(t.decoded_data.data(), &data[data_index], r);
      data_index += r;
      uint32_t name_s = *(uint32_t*)&data[data_index];
      t.name.resize(name_s);
      data_index += sizeof(uint32_t);
      std::memcpy(t.name.data(), &data[data_index], name_s);
      data_index += name_s;
      t.visual_output = *(uint32_t*)&data[data_index];
      data_index += sizeof(uint32_t);
      t.filter = *(uint32_t*)&data[data_index];
      data_index += sizeof(uint32_t);
      t.group_id = *(uint32_t*)&data[data_index];
      data_index += sizeof(uint32_t);
      texture_list.push_back(t);
    }
  }

  uint32_t size() const {
    return pack_list.size();
  }

  fan::webp::image_info_t get_pixel_data(uint32_t pack_id) {
    fan::webp::image_info_t image_info;
    image_info.data = pack_list[pack_id].pixel_data.data();
    image_info.size = pack_list[pack_id].pack_size;
    return image_info;
  }

  bool qti(const fan::string& name, loco_t::ti_t* ti) {
    //std::find_if(texture_list[0].begin(), texture_list[texture_list.size()].end(),
    //  [](const texture_t& a, const texture_t& b) {
    //  return a.hash == b.hash;
    //});

    for (uint32_t i = 0; i < pack_list.size(); i++) {
      for (uint32_t j = 0; j < pack_list[i].texture_list.size(); j++) {
        if (pack_list[i].texture_list[j].name == name) {
          ti->pack_id = i;
          ti->position = pack_list[i].texture_list[j].position;
          ti->size = pack_list[i].texture_list[j].size;
          return 0;
        }
      }
    }
    return 1;
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
      fan::string name;
    };
    std::vector<texture_t> texture_list;
    uint32_t visual_output;
    uint32_t filter;
    uint32_t group_id;
    std::vector<uint8_t> pixel_data;
  };
  std::vector<pack_t> pack_list;

public:

  void tree_debug(uint32_t pack_id, pack_t::internal_texture_t * node) {
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

  unsigned long used_surface_area(const pack_t::internal_texture_t & node) const {
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

  pack_t::internal_texture_t* push(pack_t::internal_texture_t * node, const fan::vec2ui & size) {
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