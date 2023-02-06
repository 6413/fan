struct sb_pfr_name {

  loco_t* get_loco() {
    loco_t* loco = OFFSETLESS(this, loco_t, sb_pfr_var_name);
    return loco;
  }

  struct cid_t {
    uint8_t format;
    uint16_t bm_id;
    uint16_t block_id;
    uint8_t instance_id;
  };

  #if defined(loco_yuv420p)
    #define sb_shape_var_name yuv420p
    #define sb_sprite_name yuv420p_t

    #define sb_get_loco \
      loco_t* get_loco() { \
        loco_t* loco = OFFSETLESS(this, loco_t, sb_pfr_var_name . sb_shape_var_name); \
        return loco; \
      }

    #include _FAN_PATH(graphics/opengl/2D/objects/yuv420p.h)
    yuv420p_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif

  #if defined(loco_nv12)
    #define sb_shape_var_name nv12
    #define sb_sprite_name nv12_t

    #define sb_get_loco \
      loco_t* get_loco() { \
        loco_t* loco = OFFSETLESS(this, loco_t, sb_pfr_var_name . sb_shape_var_name); \
        return loco; \
      }

    #include _FAN_PATH(graphics/opengl/2D/objects/nv12.h)
    nv12_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif

  struct properties_t {
    loco_t::matrices_t* matrices = 0;
    fan::graphics::viewport_t* viewport = 0;

    uint8_t pixel_format = fan::pixel_format::undefined;
    loco_t::image_t* images[4]{};

    fan::vec3 position = 0;
    fan::vec2 size = 0;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
  };

  void push_back(fan::graphics::cid_t* cid, const properties_t& p) {
    auto loco = get_loco();
    switch (p.pixel_format) {
    case fan::pixel_format::undefined: {
      fan::throw_error("undefined pixel format in push_back properties");
      break;
    }
      #if defined(loco_yuv420p)
      case fan::pixel_format::yuv420p: {
        yuv420p_t::properties_t sp;
        sp.matrices = p.matrices;
        sp.viewport = p.viewport;
        sp.position = p.position;
        sp.size = p.size;
        sp.tc_position = p.tc_position;
        sp.tc_size = p.tc_size;
        sp.y = p.images[0];
        sp.u = p.images[1];
        sp.v = p.images[2];
        yuv420p.push_back(cid, sp);
        break;
      }
      #endif
      #if defined(loco_nv12)
      case fan::pixel_format::nv12: {
        nv12_t::properties_t sp;
        sp.matrices = p.matrices;
        sp.viewport = p.viewport;
        sp.position = p.position;
        sp.size = p.size;
        sp.tc_position = p.tc_position;
        sp.tc_size = p.tc_size;
        sp.y = p.images[0];
        sp.vu = p.images[1];
        nv12.push_back(cid, sp);
        break;
      }
      #endif
      default: {
        fan::throw_error("pixel format has not been defined for loco. #define loco_*");
        break;
      }
    }
    ((cid_t*)cid)->format = p.pixel_format;
  }

  void reload(fan::graphics::cid_t* cid, void** data, const fan::vec2& image_size) {
    switch (((cid_t*)cid)->format) {
      #if defined(loco_yuv420p)
        case fan::pixel_format::yuv420p: {
          yuv420p.reload(cid, data, image_size);
          break;
        }
      #endif
      #if defined(loco_nv12)
        case fan::pixel_format::nv12: {
          nv12.reload(cid, data, image_size);
          break;
        }
      #endif
    }
  }

  void draw() {
    #if defined(loco_yuv420p)
      yuv420p.draw();
    #endif
    #if defined(loco_nv12)
      nv12.draw();
    #endif
  }

};