struct sb_pfr_name {

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

   struct bm_properties_t {
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

  struct ri_t : bm_properties_t {
    loco_t::image_t images[4];
    uint8_t format = fan::pixel_format::undefined;
    cid_t* cid;
  };

  struct properties_t : vi_t, ri_t {

    loco_t::camera_t* camera = 0;
    fan::graphics::viewport_t* viewport = 0;
  };

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));

  #if defined(loco_opengl)
  #ifndef sb_shader_vertex_path
  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/pixel_format_renderer.vs)
  #endif
  #ifndef sb_shader_fragment_path
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/nv12.fs)
  #endif
  #endif

  #include _FAN_PATH(graphics/shape_builder.h)

  
  sb_pfr_name() {
    sb_open();
  }
  ~sb_pfr_name() {
    sb_close();
  }

  void set_fragment(uint8_t format) {
    switch (format) {
      case fan::pixel_format::yuv420p: {
        m_shader.set_fragment(gloco->get_context(), 
          #include _FAN_PATH(graphics/glsl/opengl/2D/objects/yuv420p.fs)
        );
        m_shader.compile(gloco->get_context());
        break;
      }
      case fan::pixel_format::nv12: {
        m_shader.set_fragment(gloco->get_context(), 
          #include _FAN_PATH(graphics/glsl/opengl/2D/objects/nv12.fs)
        );
        m_shader.compile(gloco->get_context());
        break;
      }
    }
  }

  void push_back(fan::graphics::cid_t* cid, properties_t p) {
    auto loco = get_loco();
    get_key_value(uint16_t) = p.position.z;

    get_key_value(loco_t::textureid_t<0>) = &loco->unloaded_image;
    get_key_value(loco_t::textureid_t<1>) = &loco->unloaded_image;
    get_key_value(loco_t::textureid_t<2>) = &loco->unloaded_image;
    get_key_value(loco_t::textureid_t<3>) = &loco->unloaded_image;

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
    sb_push_back(cid, p);
    //auto* ri = &sb_get_ri(cid);
    //sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<0>>()>(cid, &ri->images[0]);
    //ri = &sb_get_ri(cid);
    //sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<1>>()>(cid, &ri->images[1]);
    //ri = &sb_get_ri(cid);
    //sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<2>>()>(cid, &ri->images[2]);
    //ri = &sb_get_ri(cid);
    //sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<3>>()>(cid, &ri->images[3]);
  }

  void erase(fan::graphics::cid_t* cid) {
    auto& ri = sb_get_ri(cid);
    uint32_t TextureAmount = fan::pixel_format::get_texture_amount(ri.format);
    for (uint32_t i = 0; i < TextureAmount; ++i) {
      ri.images[i].unload(get_loco());
    }
    sb_erase(cid);
  }

  void draw(bool blending = false) {
    sb_draw(root);
  }

  void reload(fan::graphics::cid_t* cid, void** data, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR) {
    reload(cid, sb_get_ri(cid).format, data, image_size, filter);
  }

  void reload(fan::graphics::cid_t* cid, uint8_t format, void** data, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR) {
    auto* ri = &sb_get_ri(cid);
    auto image_count_new = fan::pixel_format::get_texture_amount(format);
    if (format != ri->format) {
      set_vertex(
        #include _FAN_PATH(graphics/glsl/opengl/2D/objects/pixel_format_renderer.vs)
      );
      set_fragment(format);
      m_shader.use(gloco->get_context());

      auto image_count_old = fan::pixel_format::get_texture_amount(ri->format);
      if (image_count_new < image_count_old) {
        // deallocate some texture
        for (uint32_t i = image_count_new; i < image_count_old; ++i) {
          ri->images[i].unload(get_loco());
        }
      }
      else if (image_count_new > image_count_old) {
        // allocate more texture
        for (uint32_t i = image_count_old; i < image_count_new; ++i) {
          ri->images[i].create_texture(get_loco());
          switch (i) {
          case 0: {
            sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<0>>()>(cid, &ri->images[0]);
            break;
          }
          case 1: {
            sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<1>>()>(cid, &ri->images[1]);
            break;
          }
          case 2: {
            sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<2>>()>(cid, &ri->images[2]);
            break;
          }
          case 3: {
            sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<3>>()>(cid, &ri->images[3]);
            break;
          }
          }
          ri = &sb_get_ri(cid);
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
          ri->images[i].reload_pixels(get_loco(), image_info, lp);
        }
      }

    ri->format = format;
  }

  void reload(fan::graphics::cid_t* cid, uint8_t format, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR) {
    void* data[4]{};
    reload(cid, format, data, image_size, filter);
  }
};