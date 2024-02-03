struct sb_pfr_name {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::pixel_format_renderer;

  struct vi_t {
    fan::vec3 position = 0;
  private:
    f32_t pad;
  public:
    fan::vec2 size = 0;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
  private:
    f32_t pad2[2];
  public:
  };

   struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::textureid_t<0>,
      loco_t::textureid_t<1>,
      loco_t::textureid_t<2>,
      loco_t::textureid_t<3>,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    loco_t::image_t images[4];
    uint8_t format = fan::pixel_format::undefined;
    cid_t* cid;
    bool blending = false;
  };

  struct properties_t : vi_t, ri_t {

    /*todo cloned from context_key_t - make define*/
    using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::textureid_t<0>,
      loco_t::textureid_t<1>,
      loco_t::textureid_t<2>,
      loco_t::textureid_t<3>,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;

    using type_t = sb_pfr_name;

    loco_t::camera_t* camera = 0;
    fan::graphics::viewport_t* viewport = 0;
  };

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));

  #if defined(loco_opengl)
    #ifndef sb_shader_vertex_path
      #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/pixel_format_renderer.vs)
    #endif
    #ifndef sb_shader_fragment_path
      #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/nv12.fs)
    #endif
  #endif
  
  sb_pfr_name() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~sb_pfr_name() {
    sb_close();
  }

  #define sb_no_blending
  #include _FAN_PATH(graphics/shape_builder.h)

  void set_fragment(uint8_t format) {
    fan::string fragment_code;
    fan::string path;
    switch (format) {
      case fan::pixel_format::yuv420p: {
        path = STRINGIFY_DEFINE(_FAN_PATH(graphics/glsl/opengl/2D/objects/yuv420p.fs));
        break;
      }
      case fan::pixel_format::nv12: {
        path = STRINGIFY_DEFINE(_FAN_PATH(graphics/glsl/opengl/2D/objects/nv12.fs));
        break;
      }
      default: {
        fan::throw_error("invalid format");
      }
    }
    path.replace_all("<", "");
    path.replace_all(">", "");
    fan::io::file::read(path, &fragment_code);
    m_shader.set_fragment(
      fragment_code
    );
    m_shader.compile();
  }

  void push_back(loco_t::cid_nt_t& id, properties_t p) {
    get_key_value(uint16_t) = p.position.z;

    get_key_value(loco_t::textureid_t<0>) = &gloco->unloaded_image;
    get_key_value(loco_t::textureid_t<1>) = &gloco->unloaded_image;
    get_key_value(loco_t::textureid_t<2>) = &gloco->unloaded_image;
    get_key_value(loco_t::textureid_t<3>) = &gloco->unloaded_image;

   /* if (p.images[0].is_invalid()) {
      get_key_value(loco_t::textureid_t<0>) = &loco.;
      get_key_value(loco_t::textureid_t<1>) = &loco.;
      get_key_value(loco_t::textureid_t<2>) = &loco.;
      get_key_value(loco_t::textureid_t<3>) = &loco.;
    }
    else {
      get_key_value(loco_t::textureid_t<0>) = &p.images[0];
      get_key_value(loco_t::textureid_t<1>) = &p.images[1];
      get_key_value(loco_t::textureid_t<2>) = &p.images[2];
      get_key_value(loco_t::textureid_t<3>) = &p.images[3];
    }*/
    
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;
    sb_push_back(id, p);
    //auto* ri = &sb_get_ri(cid);
    //sb_set_key<context_key_t::key_t::get_index_with_type<loco_t::textureid_t<0>>()>(cid, &ri->images[0]);
    //ri = &sb_get_ri(cid);
    //sb_set_key<context_key_t::key_t::get_index_with_type<loco_t::textureid_t<1>>()>(cid, &ri->images[1]);
    //ri = &sb_get_ri(cid);
    //sb_set_key<context_key_t::key_t::get_index_with_type<loco_t::textureid_t<2>>()>(cid, &ri->images[2]);
    //ri = &sb_get_ri(cid);
    //sb_set_key<context_key_t::key_t::get_index_with_type<loco_t::textureid_t<3>>()>(cid, &ri->images[3]);
  }

  void erase(loco_t::cid_nt_t& id) {
    auto& ri = sb_get_ri(id);
    uint32_t TextureAmount = fan::pixel_format::get_texture_amount(ri.format);
    for (uint32_t i = 0; i < TextureAmount; ++i) {
      ri.images[i].unload();
    }
    sb_erase(id);
  }

  void draw(const redraw_key_t &redraw_key, loco_bdbt_NodeReference_t key_root) {
    sb_draw(key_root);
  }

  void reload(loco_t::cid_nt_t& id, void** data, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR) {
    reload(id, sb_get_ri(id).format, data, image_size, filter);
  }

  void reload(loco_t::cid_nt_t& id, uint8_t format, void** data, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR) {
    auto* ri = &sb_get_ri(id);
    auto image_count_new = fan::pixel_format::get_texture_amount(format);
    if (format != ri->format) {
      set_vertex(
        loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/pixel_format_renderer.vs))
      );
      set_fragment(format);
      m_shader.use();

      auto image_count_old = fan::pixel_format::get_texture_amount(ri->format);
      if (image_count_new < image_count_old) {
        // deallocate some texture
        for (uint32_t i = image_count_new; i < image_count_old; ++i) {
          ri->images[i].unload();
        }
      }
      else if (image_count_new > image_count_old) {
        // allocate more texture
        for (uint32_t i = image_count_old; i < image_count_new; ++i) {
          ri->images[i].create_texture();
          switch (i) {
          case 0: {
            sb_set_context_key<loco_t::textureid_t<0>>(id, &ri->images[0]);
            break;
          }
          case 1: {
            sb_set_context_key<loco_t::textureid_t<1>>(id, &ri->images[1]);
            break;
          }
          case 2: {
            sb_set_context_key<loco_t::textureid_t<2>>(id, &ri->images[2]);
            break;
          }
          case 3: {
            sb_set_context_key<loco_t::textureid_t<3>>(id, &ri->images[3]);
            break;
          }
          }
          ri = &sb_get_ri(id);
        }
      }
    }

      for (uint32_t i = 0; i < image_count_new; i++) {
        fan::webp::image_info_t image_info;
        image_info.data = data[i];
        image_info.size = fan::pixel_format::get_image_sizes(format, image_size)[i];
        auto lp = fan::pixel_format::get_image_properties<loco_t::image_t::load_properties_t>(format)[i];
        lp.min_filter = filter;
        lp.mag_filter = filter;
        if (1) {
          ri->images[i].reload_pixels(image_info, lp);
        }
      }

    ri->format = format;
  }

  void reload(loco_t::cid_nt_t& id, uint8_t format, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR) {
    void* data[4]{};
    reload(id, format, data, image_size, filter);
  }
};