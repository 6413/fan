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
  
  struct bm_properties_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::textureid_t<0>,
      loco_t::matrices_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };
  
  struct ri_t {
    fan::graphics::cid_t* cid;
  };

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  struct properties_t : vi_t, ri_t, bm_properties_t {

    make_key_value(loco_t::textureid_t<0>, image);
    make_key_value(loco_t::matrices_list_NodeReference_t, matrices);
    make_key_value(fan::graphics::viewport_list_NodeReference_t, viewport);

    properties_t() = default;
    properties_t(const vi_t& i) : vi_t(i) {}
    properties_t(const bm_properties_t& p) : bm_properties_t(p) {}
  };

  #undef make_key_value

  void push_back(fan::graphics::cid_t* cid, properties_t& p) {
    #if defined(loco_vulkan)
      auto loco = get_loco();
      VkDescriptorImageInfo imageInfo{};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      auto& img = loco->image_list[p.image];
      imageInfo.imageView = img.image_view;
      imageInfo.sampler = img.sampler;
      if (img.texture_index.sprite == (decltype(img.texture_index.sprite))-1) {
        img.texture_index.sprite = m_texture_index++;
        if (m_texture_index > fan::vulkan::max_textures) {
          fan::throw_error("too many textures max:" + fan::vulkan::max_textures);
        }
        m_ssbo.m_descriptor.m_properties[2].image_infos[img.texture_index.sprite] = imageInfo;
        m_ssbo.m_descriptor.update(loco->get_context(), 1, 2);
      }

      auto& matrices = loco->matrices_list[p.matrices];
      if (matrices.matrices_index.sprite == (decltype(matrices.matrices_index.sprite))-1) {
        matrices.matrices_index.sprite = m_matrices_index++;
        m_shader.set_matrices(loco, matrices.matrices_id, &loco->m_write_queue, matrices.matrices_index.sprite);
      }
    #endif
    sb_push_back(cid, p);
  }
  void erase(fan::graphics::cid_t* cid) {
    sb_erase(cid);
  }

  void draw() {
    sb_draw();
  }

  // can be bigger with vulkan
  static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(vi_t) / 4));
  #if defined(loco_opengl)
    #ifndef sb_shader_vertex_path
      #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.vs)
    #endif
    #ifndef sb_shader_fragment_path
      #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.fs)
    #endif
  #elif defined(loco_vulkan)
    #define vulkan_buffer_count 3
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/sprite.vert.spv)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/sprite.frag.spv)
  #endif

  #define vk_sb_ssbo
  #define vk_sb_vp
  #define vk_sb_image
  #include _FAN_PATH(graphics/shape_builder.h)
  
  sb_sprite_name() {
    sb_open();
  }
  ~sb_sprite_name() {
    sb_close();
  }

  /*loco_t::matrices_t* get_matrices(fan::graphics::cid_t* cid) {
    auto block = sb_get_block(cid);
    loco_t* loco = get_loco();
    return loco->matrices_list[*block->p[cid->instance_id].key.get_value<
      instance_properties_t::key_t::get_index_with_type<loco_t::matrices_list_NodeReference_t>()
    >()].matrices_id;
  }
  void set_matrices(fan::graphics::cid_t* cid, loco_t::matrices_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  fan::graphics::viewport_t* get_viewport(fan::graphics::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return loco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<
      instance_properties_t::key_t::get_index_with_type<fan::graphics::viewport_list_NodeReference_t>()
    >()].viewport_id;
  }
  void set_viewport(fan::graphics::cid_t* cid, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }*/

  /*

  void set(loco_t* loco, fan::opengl::cid_t* cid, fan::opengl::viewport_list_NodeReference_t n) {
    auto block = sb_get_block(loco, cid);
    *block->p[cid->instance_id].key.get_value<1>() = n;
  }*/

  //void set_image(fan::graphics::cid_t* cid, loco_t::image_t* n) {
  ////#if defined(loco_opengl)
  //  sb_set_key<instance_properties_t::key_t::get_index_with_type<loco_t::textureid_t<0>>()>(cid, n);
  ////#else
  ////  auto loco = get_loco();
  ////  auto& img = loco->image_list[n];
  ////  // maybe samplers etc need to update too
  ////  m_descriptor.m_properties[2].image_infos[img.texture_index.sprite].imageView = img.image_view;
  ////  m_descriptor.update(loco->get_context(), 1, 2, 1, img.texture_index.sprite);

  ////#endif
  //}
  //void set_viewport_value(fan::graphics::cid_t* cid, fan::vec2 p, fan::vec2 s) {
  //  loco_t* loco = get_loco();
  //  auto block = sb_get_block(cid);
  //  loco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<2>()].viewport_id->set(loco->get_context(), p, s, loco->get_window()->get_size());
  //  //sb_set_key(cid, &properties_t::image, n);
  //}

  #if defined(loco_vulkan)
    uint32_t m_texture_index = 0;
    uint32_t m_matrices_index = 0;
  #endif
};

#undef vulkan_buffer_count
#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_sprite_name