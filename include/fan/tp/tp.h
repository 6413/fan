#pragma once

#include <memory>

#include _FAN_PATH(graphics/webp.h)
#include _FAN_PATH(types/memory.h)
#include _FAN_PATH(graphics/opengl/gl_image.h)

// reference https://web.archive.org/web/20170703203916/http://clb.demon.fi/projects/rectangle-bin-packing

namespace fan {
  namespace tp {

    struct texture_packe {
      struct texture_t {
        fan::vec2i position;
        fan::vec2i size;
        std::string filepath;
        std::string name;
      };
      struct internal_texture_t {
        std::shared_ptr<internal_texture_t> d[2];

        // The top-left coordinate of the rectangle.
        fan::vec2i position;
        fan::vec2i size;

        friend std::ostream& operator<<(std::ostream& os, const internal_texture_t& tex) {
          os << '{' << "\n position:" << tex.position << "\n size:" << tex.size << "\n}";
          return os;
        }
      };

      void open() {
      }
      void close() {
      }

      uint32_t push_pack(const fan::vec2i& size) {
        pack_t pack;
        pack.bin_size = size;
        pack.root.d[0] = pack.root.d[1] = 0;
        pack.root.position = 0;
        pack.root.size = size;
        pack_list.push_back(pack);
        return pack_list.size() - 1;
      }

      void push_texture(uint32_t pack_id, const std::string& filepath, const std::string& name) {
        fan::vec2i size;
        if (fan::webp::get_image_size(filepath, &size)) {
          fan::throw_error("failed to open image:" + filepath);
        }

        internal_texture_t* it = push(&pack_list[pack_id].root, size);
        if (it == nullptr) {
          fan::print_warning("failed to push to pack:" + filepath);
          return;
        }
        texture_t texture;
        texture.position = it->position;
        texture.size = it->size;
        texture.filepath = filepath;
        texture.name = name;
        pack_list[pack_id].texture_list.push_back(texture);
      }

      void save(const char* filename) {
        FILE* f = fopen(filename, "w+b");
        if (!f) {
          fan::throw_error(std::string("failed to open file:") + filename);
        }

        uint32_t pack_amount = pack_list.size();
        fwrite(&pack_amount, sizeof(pack_amount), 1, f);

        for (uint32_t i = 0; i < pack_amount; i++) {
          uint32_t count = pack_list[i].texture_list.size();
          fwrite(&count, sizeof(count), 1, f);

          std::vector<uint8_t> r(pack_list[i].bin_size.x * pack_list[i].bin_size.y * 4);
          for (uint32_t j = 0; j < count; j++) {
            texture_t* t = &pack_list[i].texture_list[j];
            fan::webp::image_info_t image_info;
            fan::webp::load(t->filepath, &image_info);
            uint64_t hashed = fan::get_hash(t->name);
            fwrite(&hashed, sizeof(hashed), 1, f);
            fwrite(t->position.data(), sizeof(t->position), 1, f);
            fwrite(t->size.data(), sizeof(t->size), 1, f);
            for (uint32_t y = t->position.y; y < t->position.y + t->size.y; y++) {
              memcpy(
                r.data() + (y * pack_list[i].bin_size.x + t->position.x) * 4,
                &image_info.data[(y - t->position.y) * t->size.x * 4],
                t->size.x * 4
              );
            }
            fan::webp::free_image(image_info.data);
          }

          uint8_t* ptr;
          uint32_t ptr_size = fan::webp::encode_rgba(r.data(), pack_list[i].bin_size, 100, &ptr);
          fwrite(&ptr_size, sizeof(ptr_size), 1, f);
          fwrite(ptr, ptr_size, 1, f);
          WebPFree(ptr);
        }
        fclose(f);
      }

    private:
      struct pack_t {
        internal_texture_t root;
        fan::vec2ui bin_size;
        std::vector<texture_t> texture_list;
      };
      std::vector<pack_t> pack_list;

      unsigned long used_surface_area(const internal_texture_t& node) const {
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

      internal_texture_t* push(internal_texture_t* node, const fan::vec2i& size) {
        if (node->d[0] || node->d[1]) {
          if (node->d[0]) {
            internal_texture_t* newNode = push(node->d[0].get(), size);
            if (newNode)
              return newNode;
          }
          if (node->d[1]) {
            internal_texture_t* newNode = push(node->d[1].get(), size);
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
        node->d[0] = std::make_shared<internal_texture_t>();
        node->d[1] = std::make_shared<internal_texture_t>();
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
