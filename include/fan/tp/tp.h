#pragma once

#include <memory>

#include _FAN_PATH(graphics/webp.h)
#include _FAN_PATH(types/memory.h)
#include _FAN_PATH(graphics/opengl/gl_image.h)

// reference https://web.archive.org/web/20170703203916/http://clb.demon.fi/projects/rectangle-bin-packing

namespace fan {
  namespace tp {

    struct texture_packd {
      struct texture_t {

        // The top-left coordinate of the rectangle.
        uint32_t pack_id;
        uint64_t hash;
        fan::vec2i position;
        fan::vec2i size;


        friend std::ostream& operator<<(std::ostream& os, const texture_t& tex) {
          os << '{' << "\n position:" << tex.position << "\n size:" << tex.size << "\n}";
          return os;
        }

      };

      struct ti_t {
        uint32_t pack_id;
        fan::vec2i position;
        fan::vec2i size;
      };

      struct pixel_data_t {
        fan::vec2i size;
        uint8_t* data;
      };
      uint32_t pack_amount;
      fan::hector_t<fan::hector_t<texture_t>> texture_list;
      fan::hector_t<pixel_data_t> pixel_data_list;

      pixel_data_t get_pixel_data(uint32_t pack_id) {
        return pixel_data_list[pack_id];
      }

      fan::opengl::image_t* load_image(fan::opengl::context_t* context, uint32_t pack_id) {
        fan::webp::image_info_t image_info;
        image_info.data = pixel_data_list[pack_id].data;
        image_info.size = pixel_data_list[pack_id].size;
        return fan::opengl::load_image(context, image_info);
      }

      void open(const char* filename) {
        texture_list.open();
        pixel_data_list.open();

        std::string data = fan::io::file::read(filename);
        uint32_t data_index = 0;
        pack_amount = *(uint32_t*)&data[data_index];
        texture_list.resize(pack_amount);
        pixel_data_list.resize(pack_amount);
        data_index += sizeof(pack_amount);
        for (uint32_t i = 0; i < pack_amount; i++) {
          uint32_t texture_amount = *(uint32_t*)&data[data_index];
          data_index += sizeof(pack_amount);
          texture_list[i].open();
          for (uint32_t j = 0; j < texture_amount; j++) {
            texture_packd::texture_t texture;
            texture.hash = *(uint64_t*)&data[data_index];
            data_index += sizeof(uint64_t);
            texture.position = *(fan::vec2i*)&data[data_index];
            data_index += sizeof(fan::vec2i);
            texture.size = *(fan::vec2i*)&data[data_index];
            data_index += sizeof(fan::vec2i);
            texture_list[i].push_back(texture);
          }
          uint32_t size = *(uint32_t*)&data[data_index];
          data_index += sizeof(uint32_t);

          pixel_data_list[i].data = WebPDecodeRGBA(
            (const uint8_t*)&data[data_index],
            size,
            &pixel_data_list[i].size.x,
            &pixel_data_list[i].size.y
          );
          data_index += size;
        }

      }
      void close() {
        for (uint32_t i = 0; i < pack_amount; i++) {
          texture_list[i].close();
          WebPFree(pixel_data_list[i].data);
        }
        texture_list.close();
        pixel_data_list.close();
      }

      bool qti(const std::string& name, ti_t* ti) {

        std::hash<std::string> hasher;
        uint64_t hash = hasher(name);

        //std::find_if(texture_list[0].begin(), texture_list[texture_list.size()].end(),
        //  [](const texture_t& a, const texture_t& b) {
        //  return a.hash == b.hash;
        //});

        for (uint32_t i = 0; i < texture_list.size(); i++) {
          for (uint32_t j = 0; j < texture_list[i].size(); j++) {
            if (texture_list[i][j].hash == hash) {
              ti->pack_id = i;
              ti->position = texture_list[i][j].position;
              ti->size = texture_list[i][j].size;
              return 1;
            }
          }
        }

        return 0;
      }

    };

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
            fan::webp::image_info_t image_info = fan::webp::load_image(t->filepath);
            std::hash<std::string> hasher;
            uint64_t hashed = hasher(t->name);
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
          uint32_t ptr_size = WebPEncodeRGBA(r.data(), pack_list[i].bin_size.x, pack_list[i].bin_size.y, pack_list[i].bin_size.x * 4, 100, &ptr);
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
