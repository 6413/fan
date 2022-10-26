struct sb_sprite_name {

  struct instance_t {
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

  #define hardcode0_t fan::opengl::textureid_t<0>
  #define hardcode0_n y
  #define hardcode1_t fan::opengl::textureid_t<1>
  #define hardcode1_n u
  #define hardcode2_t fan::opengl::textureid_t<2>
  #define hardcode2_n v

  #define hardcode3_t fan::opengl::matrices_list_NodeReference_t
  #define hardcode3_n matrices
  #define hardcode4_t fan::opengl::viewport_list_NodeReference_t
  #define hardcode4_n viewport
  #include _FAN_PATH(graphics/opengl/2D/objects/hardcode_open.h)

  struct instance_properties_t {
    struct key_t : parsed_masterpiece_t {}key;
    expand_get_functions
  };

  struct properties_t : instance_t, instance_properties_t {
    properties_t() = default;
    properties_t(const instance_t& i) : instance_t(i) {}
    properties_t(const instance_properties_t& p) : instance_properties_t(p) {}

      #define _load_yuv(_f) \
      fan::opengl::context_t* context = loco->get_context(); \
      fan::opengl::image_t::load_properties_t lp; \
      lp.format = fan::opengl::GL_RED; \
      lp.internal_format = fan::opengl::GL_RED; \
                                 \
      fan::webp::image_info_t ii; \
                                \
      ii.data = data[0]; \
      ii.size = image_size; \
      loco->sb_shape_var_name.image[0]._f(context, ii, lp); \
                                \
      ii.data = data[1]; \
      ii.size = image_size / 2; \
      loco->sb_shape_var_name.image[1]._f(context, ii, lp); \
                                \
      ii.data = data[2]; \
      loco->sb_shape_var_name.image[2]._f(context, ii, lp); 

    void load_yuv(loco_t* loco, void* data, const fan::vec2& image_size) {
      void* datas[3];
      uint64_t offset = 0;
      datas[0] = data;
      datas[1] = (uint8_t*)data + (offset += image_size.multiply());
      datas[2] = (uint8_t*)data + (offset += image_size.multiply() / 4);
      load_yuv(loco, datas, image_size);
    }
    // void*[3]
    void load_yuv(loco_t* loco, void** data, const fan::vec2& image_size) {
      _load_yuv(load);

      get_y() = &loco->sb_shape_var_name.image[0];
      get_u() = &loco->sb_shape_var_name.image[1];
      get_v() = &loco->sb_shape_var_name.image[2];
    }
  };

  void push_back(fan::opengl::cid_t* cid, properties_t& p) {
    sb_push_back(cid, p);
  }
  void erase(fan::opengl::cid_t* cid) {
    sb_erase(cid);
  }

  void draw() {
    sb_draw();
  }

  void reload_yuv(fan::opengl::cid_t* cid, void** data, const fan::vec2& image_size) {
    auto loco = get_loco();
    _load_yuv(reload_pixels);
    set_image_data(cid, &image[0], &image[1], &image[2]);
  }

  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(instance_t) / 4));
  #ifndef sb_shader_vertex_path
  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.vs)
  #endif
  #ifndef sb_shader_fragment_path
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/yuv420p.fs)
  #endif
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)
  #undef sb_shader_vertex_path
  #undef sb_shader_fragment_path

  sb_sprite_name() {
    sb_open();
  }
  ~sb_sprite_name() {
    sb_close();
  }

  void set_image_data(fan::opengl::cid_t* cid, 
    fan::opengl::textureid_t<0> y,
    fan::opengl::textureid_t<1> u,
    fan::opengl::textureid_t<2> v
  ) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(y)>()>(cid, y);
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(u)>()>(cid, u);
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(v)>()>(cid, v);
  }

  fan::opengl::image_t image[3];

  /*void set_matrices(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::matrices_list_NodeReference_t n) {
  auto block = sb_get_block(loco, cid);
  *block->p[cid->instance_id].key.get_value<0>() = n;
  }

  void set(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
  auto block = sb_get_block(loco, cid);
  *block->p[cid->instance_id].key.get_value<1>() = n;
  }*/
};

#undef _load_yuv
#undef sb_sprite_name

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)