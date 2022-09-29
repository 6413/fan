#pragma once

#include <memory>

#include _FAN_PATH(graphics/webp.h)
#include _FAN_PATH(types/memory.h)
#include _FAN_PATH(graphics/opengl/gl_image.h)

namespace fan {
  namespace tp {
    struct ti_t {
      uint32_t pack_id;
      fan::vec2 position;
      fan::vec2 size;
    };

  }
}

#include _FAN_PATH(tp/tp0.h)

// reference https://web.archive.org/web/20170703203916/http://clb.demon.fi/projects/rectangle-bin-packing

namespace fan {
  namespace tp {

    struct texture_packe {

      struct texture_t {
        fan::string filepath;
        fan::string name;
        uint32_t visual_output;
        uint32_t filter;
        uint32_t group_id;
      };

      std::vector<texture_t> texture_list;

      struct open_properties_t {
        open_properties_t() {}

        fan::vec2ui preferred_pack_size = 1024;
        uint32_t visual_output = fan::opengl::image_t::load_properties_defaults::visual_output;
        uint32_t filter = fan::opengl::image_t::load_properties_defaults::filter;
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

      void open(const open_properties_t& op) {
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

      void push_texture(const fan::string& filepath, const texture_properties_t& texture_properties = texture_properties_t()) {

        if (texture_properties.name.empty()) {
          fan::print_warning("texture properties name empty");
          return;
        }

        texture_t t;
        t.filepath = filepath;
        t.name = texture_properties.name;
        t.visual_output = texture_properties.visual_output;
        t.filter = texture_properties.filter;
        t.group_id = texture_properties.group_id;

        texture_list.push_back(t);
      }

      void process() {
        pack_list.clear();

        for (uint32_t ci = 0; ci < texture_list.size(); ci++) {

          fan::string filepath = texture_list[ci].filepath;

          fan::vec2ui size;
          if (fan::webp::get_image_size(filepath, &size)) {
            fan::throw_error("failed to open image:" + filepath);
          }
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
          texture.filepath = filepath;
          texture.name = texture_properties.name;
          pack_list[selected_pack].texture_list.push_back(texture);
        }

        for (uint32_t i = 0; i < pack_list.size(); i++) {
          uint32_t count = pack_list[i].texture_list.size();

          pack_list[i].pixel_data.resize(pack_list[i].pack_size.x * pack_list[i].pack_size.y * 4);

          for (uint32_t j = 0; j < count; j++) {
            pack_t::texture_t* t = &pack_list[i].texture_list[j];
            fan::webp::image_info_t image_info;
            fan::webp::load(t->filepath, &image_info);
            uint64_t hashed = fan::get_hash(t->name);
            for (uint32_t y = t->position.y; y < t->position.y + t->size.y; y++) {
              memcpy(
                pack_list[i].pixel_data.data() + (y * pack_list[i].pack_size.x + t->position.x) * 4,
                &((uint8_t*)image_info.data)[(y - t->position.y) * t->size.x * 4],
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

              for (uint32_t y = pp.y; y != pp.y + ps.y; y++) {
                for (uint32_t x = pp.x; x != pp.x + ps.x; x++) {

                  static auto fill_pad = [&](int px, int py) {
                    pack_list[i].pixel_data[(y * pack_list[i].pack_size.x + x) * 4 + 0] = pack_list[i].pixel_data[((y + py) * pack_list[i].pack_size.x + x + px) * 4 + 0];
                    pack_list[i].pixel_data[(y * pack_list[i].pack_size.x + x) * 4 + 1] = pack_list[i].pixel_data[((y + py) * pack_list[i].pack_size.x + x + px) * 4 + 1];
                    pack_list[i].pixel_data[(y * pack_list[i].pack_size.x + x) * 4 + 2] = pack_list[i].pixel_data[((y + py) * pack_list[i].pack_size.x + x + px) * 4 + 2];
                    pack_list[i].pixel_data[(y * pack_list[i].pack_size.x + x) * 4 + 3] = pack_list[i].pixel_data[((y + py) * pack_list[i].pack_size.x + x + px) * 4 + 3];
                  };

                  if (x == pp.x && y == pp.y) {
                    fill_pad(1, 1);
                  }
                  else if (x == pp.x + ps.x - 1 && y == pp.y) {
                    fill_pad(-1, 1);
                  }
                  else if (x == pp.x + ps.x - 1 && y == pp.y + ps.y - 1) {
                    fill_pad(-1, -1);
                  }
                  else if (x == pp.x && y == pp.y + ps.y - 1) {
                    fill_pad(1, -1);
                  }
                  else if (x == pp.x) {
                    fill_pad(1, 0);
                  }
                  else if (x == pp.x + ps.x - 1) {
                    fill_pad(-1, 0);
                  }
                  else if (y == pp.y) {
                    fill_pad(0, 1);
                  }
                  else if (y == pp.y + ps.y - 1) {
                    fill_pad(0, -1);
                  }
                }
              }
            }
            fan::webp::free_image(image_info.data);
          }
        }
      }

      void save(const char* filename) {
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
          WebPFree(ptr);
          fan::io::file::write(f, &pack_list[i].visual_output, sizeof(uint32_t), 1);
          fan::io::file::write(f, &pack_list[i].filter, sizeof(uint32_t), 1);
        }
        fan::io::file::close(f);
      }

      uint32_t size() const {
        return pack_list.size();
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
          fan::string filepath;
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
  }
}
