struct texturepack_t {

  using ti_t = loco_t::ti_t;

  struct texture_t {

    // The top-left coordinate of the rectangle.
    uint32_t pack_id;
    std::string image_name;
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
  std::string file_path;

  pixel_data_t& get_pixel_data(uint32_t pack_id) {
    return pixel_data_list[pack_id];
  }

  void open_compiled(const std::string& filename) {
    fan::graphics::image_load_properties_t lp;
    lp.visual_output =fan::graphics::image_sampler_address_mode::clamp_to_edge;
    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;
    /*
    lp.min_filter = (decltype(lp.min_filter))min_filter;
    lp.mag_filter = (decltype(lp.mag_filter))mag_filter;
    */
    open_compiled(filename, lp);
  }
  void open_compiled(const std::string& filename, fan::graphics::image_load_properties_t lp) {
    texture_list.clear();
    pixel_data_list.clear();

    file_path = filename;

    std::string in;
    fan::io::file::read(filename, &in);

    std::size_t offset = 0;
    std::size_t pack_list_size = fan::string_read_data<std::size_t>(in, offset);


    pixel_data_list.resize(pack_list_size);
    texture_list.resize(pack_list_size);
    for (std::size_t i = 0; i < pack_list_size; i++) {
      std::size_t texture_list_size = fan::string_read_data<std::size_t>(in, offset);
      texture_list[i].resize(texture_list_size);
      for (std::size_t k = 0; k < texture_list_size; k++) {
        texturepack_t::texture_t texture;
        texture.image_name = fan::string_read_data<std::string>(in, offset);
        texture.position = fan::string_read_data<fan::vec2ui>(in, offset);
        texture.size = fan::string_read_data<fan::vec2ui>(in, offset);
        texture_list[i][k] = texture;
      }

      std::vector<uint8_t> pixel_data = fan::string_read_data<std::vector<uint8_t>>(in, offset);
      fan::image::image_info_t image_info;
      image_info.data = WebPDecodeRGBA(
        pixel_data.data(),
        pixel_data.size(),
        &image_info.size.x,
        &image_info.size.y
      );
      image_info.channels = 4;
      pixel_data_list[i].image = gloco->image_load(image_info, lp);
      WebPFree(image_info.data);

      //pixel_data_list[i].visual_output = 
      fan::string_read_data<uint32_t>(in, offset);
      //pixel_data_list[i].min_filter = 
      fan::string_read_data<uint32_t>(in, offset);
      //pixel_data_list[i].mag_filter = 
      fan::string_read_data<uint32_t>(in, offset);
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
  bool qti(const std::string& name, ti_t* ti) {
    bool ret = 1;
    iterate_loaded_images([&](auto& image, uint32_t pack_id) {
      if (ret == 0) {
        return;
      }
      if (image.image_name == name) {
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
  bool qti(uint64_t hash, ti_t* ti) {
    bool ret = 1;
    iterate_loaded_images([&](auto& image, uint32_t pack_id) {
      if (ret == 0) {
        return;
      }
      if (fan::get_hash(image.image_name.c_str()) == hash) {
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