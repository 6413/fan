struct texturepack_t {

  #include _FAN_PATH(tp/tp0.h)

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

  struct pixel_data_t {
    loco_t::image_t image;
  };
  uint32_t pack_amount;
  std::vector<std::vector<texture_t>> texture_list;
  std::vector<pixel_data_t> pixel_data_list;

  pixel_data_t& get_pixel_data(uint32_t pack_id) {
    return pixel_data_list[pack_id];
  }

  void open_compiled(const fan::string& filename) {
    loco_t::image_t::load_properties_t lp;
    lp.visual_output = loco_t::image_t::sampler_address_mode::clamp_to_edge;
    lp.min_filter = fan::opengl::GL_LINEAR;
    lp.mag_filter = fan::opengl::GL_LINEAR;
    /*
    lp.min_filter = (decltype(lp.min_filter))min_filter;
    lp.mag_filter = (decltype(lp.mag_filter))mag_filter;
    */
    open_compiled(filename, lp);
  }
  void open_compiled(const fan::string& filename, loco_t::image_t::load_properties_t lp) {
    auto& context = gloco->get_context();

    fan::string data;
    fan::io::file::read(filename, &data);
    uint32_t data_index = 0;
    pack_amount = *(uint32_t*)&data[data_index];
    texture_list.resize(pack_amount);
    pixel_data_list.resize(pack_amount);
    data_index += sizeof(pack_amount);
    for (uint32_t i = 0; i < pack_amount; i++) {
      uint32_t texture_amount = *(uint32_t*)&data[data_index];
      data_index += sizeof(pack_amount);
      for (uint32_t j = 0; j < texture_amount; j++) {
        texturepack_t::texture_t texture;
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
      //#if defined(loco_vulkan)
      //	fan::throw_error("only implemented for opengl, bcause of visual output type");
      //#endif
      uint32_t visual_output = *(uint32_t*)&data[data_index];
      data_index += sizeof(uint32_t);
      uint32_t min_filter = *(uint32_t*)&data[data_index];
      data_index += sizeof(uint32_t);

      uint32_t mag_filter = *(uint32_t*)&data[data_index];
      data_index += sizeof(uint32_t);

      pixel_data_list[i].image.load(image_info, lp);
      WebPFree(image_info.data);
    }
  }

  void iterate_loaded_images(auto lambda) {
    for (uint32_t i = 0; i < texture_list.size(); i++) {
      for (uint32_t j = 0; j < texture_list[i].size(); j++) {
        lambda(texture_list[i][j], i);
      }
    }
  }

  // query texturepack image
  bool qti(const fan::string& name, ti_t* ti) {
    return qti(fan::get_hash(name), ti);
  }
  bool qti(uint64_t hash, ti_t* ti) {
    bool ret = 1;
    iterate_loaded_images([&] (auto& image, uint32_t pack_id){
      if (ret == 0) {
        return;
      }
      if (image.hash == hash) {
        ti->pack_id = pack_id;
        ti->position = image.position;
        ti->size = image.size;
        ti->image = &get_pixel_data(ti->pack_id).image;
        ret = 0;
        return;
      }
    });
    return ret;
  }
};