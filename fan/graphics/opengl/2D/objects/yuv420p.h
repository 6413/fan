struct sb_sprite_name {

  struct vi_t {
    fan::vec3 position = 0;
  private:
    f32_t pad;
  public:
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t angle = 0;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
  };

   struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::textureid_t<0>,
      loco_t::textureid_t<1>,
      loco_t::textureid_t<2>,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  #define sb_cid sb_pfr_name::cid_t

  struct ri_t {
    cid_t* cid;
  };

  struct properties_t : vi_t, ri_t {

    /*todo cloned from context_key_t - make define*/
    using parsed_masterpiece_t = fan::masterpiece_t<
      uint16_t,
      loco_t::textureid_t<0>,
      loco_t::textureid_t<1>,
      loco_t::textureid_t<2>,
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;

    loco_t::image_t* y = 0;
    loco_t::image_t* u = 0;
    loco_t::image_t* v = 0;

    loco_t::camera_t* camera = 0;
    fan::graphics::viewport_t* viewport = 0;

    properties_t() = default;
    properties_t(const vi_t& i) : vi_t(i) {}
    properties_t(const ri_t& i) : ri_t(i) {}

  private:
    void _load_yuv(loco_t* loco, void** data, const fan::vec2& image_size, uint32_t stride[3]) {
      loco_t::image_t::load_properties_t lp;
      lp.format = loco_t::image_t::format::r8_unorm;
      lp.internal_format = loco_t::image_t::format::r8_unorm; 
      lp.visual_output = loco_t::image_t::sampler_address_mode::clamp_to_edge;
      lp.filter = loco_t::image_t::filter::linear;
                                 
      fan::webp::image_info_t ii;

      ii.data = data[0];
      ii.size = image_size; 
      loco->sb_pfr_var_name.sb_shape_var_name.image[0].load(loco, ii, lp);
                                
      ii.data = data[1]; 
      ii.size = image_size / 2; 
      loco->sb_pfr_var_name.sb_shape_var_name.image[1].load(loco, ii, lp);
                                
      ii.data = data[2]; 
      loco->sb_pfr_var_name.sb_shape_var_name.image[2].load(loco, ii, lp);
    }
  public:

    void load_yuv(loco_t* loco, void* data, const fan::vec2& image_size) {
      uint32_t stride[3];
      stride[0] = image_size.x;
      stride[1] = image_size.x / 2;
      stride[2] = image_size.x / 2;
      load_yuv(loco, data, image_size, stride);
    }
    void load_yuv(loco_t* loco, void* data, const fan::vec2& image_size, uint32_t stride[3]) {
      void* datas[3];
      uint64_t offset = 0;
      datas[0] = data;
      datas[1] = (uint8_t*)data + (offset += image_size.multiply());
      datas[2] = (uint8_t*)data + (offset += image_size.multiply() / 4);
      load_yuv(loco, datas, image_size, stride);
    }

    void load_yuv(loco_t* loco, void** data, const fan::vec2& image_size) {
      uint32_t stride[3];
      stride[0] = image_size.x;
      stride[1] = image_size.x / 2;
      stride[2] = image_size.x / 2;
      load_yuv(loco, data, image_size, stride);
    }
    void load_yuv(loco_t* loco, void** data, const fan::vec2& image_size, uint32_t stride[3]) {

      _load_yuv(loco, data, image_size, stride);

      y = &loco->sb_pfr_var_name.sb_shape_var_name.image[0];
      u = &loco->sb_pfr_var_name.sb_shape_var_name.image[1];
      v = &loco->sb_pfr_var_name.sb_shape_var_name.image[2];
    }
  };

  void push_back(fan::graphics::cid_t* cid, properties_t& p) {

    get_key_value(uint16_t) = p.position.z;
    if (p.y == nullptr) {
      get_key_value(loco_t::textureid_t<0>) = &image[0];
      get_key_value(loco_t::textureid_t<1>) = &image[1];
      get_key_value(loco_t::textureid_t<2>) = &image[2];
    }
    else {
      get_key_value(loco_t::textureid_t<0>) = p.y;
      get_key_value(loco_t::textureid_t<1>) = p.u;
      get_key_value(loco_t::textureid_t<2>) = p.v;
      image[0] = *p.y;
      image[1] = *p.u;
      image[2] = *p.v;
    }
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    sb_push_back(cid, p);

  #if defined(loco_vulkan)
    auto loco = get_loco();
    auto& camera = loco->camera_list[p.camera];
    if (camera.camera_index.yuv420p == (decltype(camera.camera_index.yuv420p))-1) {
      camera.camera_index.yuv420p = m_camera_index++;
      m_shader.set_camera(loco, camera.camera_id, camera.camera_index.yuv420p);  
    }

  #endif

  }
  void erase(fan::graphics::cid_t* cid) {
    sb_erase(cid);
  }

  void draw(bool blending = false) {
    sb_draw(root);
  }

  void reload(fan::graphics::cid_t* cid, void** data, const fan::vec2& image_size) {
    auto loco = get_loco();
    
    loco_t::image_t::load_properties_t lp;
    lp.format = loco_t::image_t::format::r8_unorm;
    lp.internal_format = loco_t::image_t::format::r8_unorm; 
    lp.visual_output = loco_t::image_t::sampler_address_mode::clamp_to_edge;
    lp.filter = loco_t::image_t::filter::linear;
                                 
    fan::webp::image_info_t ii; 
                               
    ii.data = data[0];
    ii.size = image_size; 
    loco->sb_pfr_var_name.sb_shape_var_name.image[0].reload_pixels(loco, ii,lp);
                                
    ii.data = data[1]; 
    ii.size = image_size / 2; 
    loco->sb_pfr_var_name.sb_shape_var_name.image[1].reload_pixels(loco, ii, lp);
                                
    ii.data = data[2]; 
    loco->sb_pfr_var_name.sb_shape_var_name.image[2].reload_pixels(loco, ii, lp);
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));


  #if defined(loco_opengl)
    #ifndef sb_shader_vertex_path
      #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.vs)
    #endif
    #ifndef sb_shader_fragment_path
      #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/yuv420p.fs)
    #endif
  #endif

  #include _FAN_PATH(graphics/shape_builder.h)

  sb_sprite_name() : image{ 
    loco_t::image_t(get_loco()), 
    loco_t::image_t(get_loco()), 
    loco_t::image_t(get_loco()) 
  } {
    sb_open();
  }
  ~sb_sprite_name() {
    sb_close();
  }

  loco_t::image_t image[3];

  /*void set_camera(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::camera_list_NodeReference_t n) {
  auto block = sb_get_block(loco, cid);
  *block->p[cid->instance_id].key.get_value<0>() = n;
  }

  void set(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
  auto block = sb_get_block(loco, cid);
  *block->p[cid->instance_id].key.get_value<1>() = n;
  }*/

  #if defined(loco_vulkan)
    //uint32_t m_texture_index = 0;
    uint32_t m_camera_index = 0;
  #endif
};

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)