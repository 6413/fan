#pragma once

#include _FAN_PATH(graphics/opengl/gl_image.h)

namespace fan {
  namespace opengl {
    struct texturepack {
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
        fan::opengl::image_t image;
      };
      uint32_t pack_amount;
      fan::hector_t<fan::hector_t<texture_t>> texture_list;
      fan::hector_t<pixel_data_t> pixel_data_list;

      pixel_data_t get_pixel_data(uint32_t pack_id) {
        return pixel_data_list[pack_id];
      }

      void open(fan::opengl::context_t* context, const char* filename) {
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
            texturepack::texture_t texture;
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

          fan::webp::image_info_t image_info;
          image_info.data = WebPDecodeRGBA(
            (const uint8_t*)&data[data_index],
            size,
            &image_info.size.x,
            &image_info.size.y
          );
          data_index += size;
          pixel_data_list[i].image.load(context, image_info); 
          WebPFree(image_info.data);
        }

      }
      void close() {
        for (uint32_t i = 0; i < pack_amount; i++) {
          texture_list[i].close();
        }
        texture_list.close();
        pixel_data_list.close();
      }

      bool qti(const std::string& name, ti_t* ti) {
        uint64_t hash = fan::get_hash(name);

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
  }
}
