struct texture_pack_t {

  using ti_t = fan::graphics::texture_pack::ti_t;

  struct texture_minor_t {
    std::string name;
    fan::vec2i position;
    fan::vec2i size;
  };

  inline static constexpr uint16_t MAX_TEXTURE_MINOR = 1024;
  struct single_texturepack_t {
    texture_minor_t texture_minor_list[MAX_TEXTURE_MINOR];
    std::vector<uint8_t> webpdata;
  };

  struct texture_minor_decoded_t {
    fan::graphics::texture_pack::unique_t unique_id;
    std::string name;
    fan::vec2i position;
    fan::vec2i size;
  };
  struct pixel_data_t {
    fan::graphics::image_t image;
  };
  struct single_texturepack_decoded_t {
    uint32_t minor_count;
    texture_minor_decoded_t texture_minor_list[MAX_TEXTURE_MINOR];
    uint32_t image_list_id;
  };

  std::vector<single_texturepack_decoded_t> texture_major_list;
  std::vector<pixel_data_t> image_list;
  fan::graphics::texture_pack::texture_unique_map_t unique_map;

  std::string file_path;

  pixel_data_t& get_pixel_data(fan::graphics::texture_pack::unique_t unique) {
    return image_list[unique_map[unique].major];
  }

  void open_compiled(const std::string& filename, const std::source_location& callers_path = std::source_location::current()) {
    fan::graphics::image_load_properties_t lp;
    lp.visual_output =fan::graphics::image_sampler_address_mode::clamp_to_edge;
    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;
    /*
    lp.min_filter = (decltype(lp.min_filter))min_filter;
    lp.mag_filter = (decltype(lp.mag_filter))mag_filter;
    */
    open_compiled(filename, lp, callers_path);
  }
  void open_compiled(const std::string& filename, fan::graphics::image_load_properties_t lp, const std::source_location& callers_path = std::source_location::current()) {
    texture_major_list.clear();
    image_list.clear();
    unique_map.Clear();

    file_path = filename;

    std::string in;
    fan::io::file::read(fan::io::file::find_relative_path(filename, callers_path), &in);

    std::size_t offset = 0;
    std::size_t pack_list_size = fan::string_read_data<std::size_t>(in, offset);


    image_list.resize(pack_list_size);
    texture_major_list.resize(pack_list_size);
    uint32_t unique_id = 0;
    for (std::size_t i = 0; i < pack_list_size; i++) {
      std::size_t texture_list_size = fan::string_read_data<std::size_t>(in, offset);
      texture_major_list[i].minor_count = texture_list_size;
      texture_major_list[i].image_list_id = i;
      for (std::size_t k = 0; k < texture_list_size; k++) {
        texture_pack_t::texture_minor_decoded_t texture;
        texture.name = fan::string_read_data<std::string>(in, offset);
        texture.position = fan::string_read_data<fan::vec2ui>(in, offset);
        texture.size = fan::string_read_data<fan::vec2ui>(in, offset);
        auto it = unique_map.NewNodeLast();
        unique_map[it].major = i;
        unique_map[it].minor = k;
        texture.unique_id = it;
        texture_major_list[i].texture_minor_list[k] = texture;
      }

      std::vector<uint8_t> pixel_data = fan::string_read_data<std::vector<uint8_t>>(in, offset);
      fan::webp::info_t image_info;
      if (fan::webp::decode(
        pixel_data.data(),
        pixel_data.size(),
        &image_info
        )) {
        fan::throw_error_impl();
      }
      image_info.type = fan::image::image_type_e::webp;
      image_info.channels = 4;
      image_list[i].image = fan::graphics::g_render_context_handle->image_load_info_props(
        fan::graphics::g_render_context_handle, 
        *(fan::image::info_t*)&image_info, 
        lp
      );
      fan::webp::free_image(image_info.data);

      //pixel_data_list[i].visual_output = 
      fan::string_read_data<uint32_t>(in, offset);
      //pixel_data_list[i].min_filter = 
      fan::string_read_data<uint32_t>(in, offset);
      //pixel_data_list[i].mag_filter = 
      fan::string_read_data<uint32_t>(in, offset);
    }
  }

  void iterate_loaded_images(auto lambda) {
    for (uint32_t i = 0; i < texture_major_list.size(); i++) {
      for (uint32_t j = 0; j < texture_major_list[i].minor_count; j++) {
        lambda(texture_major_list[i].texture_minor_list[j]);
      }
    }
  }

  operator bool() {
    return size();
  }

  std::size_t size() const {
    return texture_major_list.size();
  }

  texture_minor_decoded_t operator[](fan::graphics::texture_pack::unique_t unique_id) {
    if (unique_id.iic()) {
      return texture_minor_decoded_t{};
    }
    return texture_major_list[unique_map[unique_id].major].texture_minor_list[unique_map[unique_id].minor];
  }

  fan::graphics::texture_pack::unique_t operator[](const std::string& name) {
    texture_pack_t::ti_t ti;
    qti(name, &ti);
    return ti.unique_id;
  }

  // query texturepack image
  bool qti(const std::string& name, ti_t* ti) {
    bool ret = 1;
    iterate_loaded_images([&](auto& image) {
      if (ret == 0) {
        return;
      }
      if (image.name == name) {
        ti->unique_id = image.unique_id;
        ti->position = image.position;
        ti->size = image.size;
        ti->image = get_pixel_data(ti->unique_id).image;
        ret = 0;
        return;
      }
      });
    return ret;
  }
  bool qti(uint64_t hash, ti_t* ti) {
    bool ret = 1;
    iterate_loaded_images([&](auto& image) {
      if (ret == 0) {
        return;
      }
      if (fan::get_hash(image.name.c_str()) == hash) {
        ti->unique_id = image.unique_id;
        ti->position = image.position;
        ti->size = image.size;
        ti->image = get_pixel_data(ti->unique_id).image;
        ret = 0;
        return;
      }
      });
    return ret;
  }
};