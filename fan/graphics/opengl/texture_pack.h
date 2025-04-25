struct texturepack_t {

  using ti_t = fan::graphics::ti_t;

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
  void open_compiled(const std::string& filename, fan::graphics::image_load_properties_t lp);

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