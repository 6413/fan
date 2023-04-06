struct sb_sprite_name {

  #ifndef sb_custom_shape_type
  static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::sprite;
  #else
  static constexpr typename loco_t::shape_type_t::_t shape_type = sb_custom_shape_type;
  #endif

  struct vi_t {
    loco_sprite_vi_t
  };

  struct bm_properties_t {
    loco_sprite_bm_properties_t
  };
  
  struct cid_t;

  struct ri_t : bm_properties_t {
    loco_sprite_ri_t
  };

  struct properties_t : vi_t, ri_t {
    using type_t = sb_sprite_name;
    loco_sprite_properties_t
  };

  void push_back(fan::graphics::cid_t* cid, properties_t p) {

    get_key_value(uint16_t) = p.position.z;
    get_key_value (loco_t::textureid_t<0>) = p.image;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

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

      auto& camera = loco->camera_list[p.camera];
      if (camera.camera_index.sprite == (decltype(camera.camera_index.sprite))-1) {
        camera.camera_index.sprite = m_camera_index++;
        m_shader.set_camera(loco, camera.camera_id, camera.camera_index.sprite);
      }
    #endif
    loco_bdbt_NodeReference_t key_root = root;
    {
      uint8_t key_blending = p.blending;
      loco_bdbt_Key_t<8> k;
      typename decltype(k)::KeySize_t ki;
      k.Query(&get_loco()->bdbt, &key_blending, &ki, &key_root);
      if (ki != 8) {
        auto sub_root = key_root;
        key_root = loco_bdbt_NewNode(&get_loco()->bdbt);
        k.InFrom(&get_loco()->bdbt, &key_blending, ki, sub_root, key_root);
      }
    }
    sb_push_back(cid, p, key_root);
  }
  void erase(fan::graphics::cid_t* cid) {
    loco_bdbt_NodeReference_t key_root = root;
    uint8_t key_blending = sb_get_ri(cid).blending;
    loco_bdbt_Key_t<8> k;
    typename decltype(k)::KeySize_t ki;
    k.Query(&get_loco()->bdbt, &key_blending , &ki, &key_root);
    sb_erase(cid, key_root);
  }

  void draw(bool blending) {
    loco_bdbt_NodeReference_t key_root = root;
    uint8_t key_blending = blending;
    loco_bdbt_Key_t<8> k;
    typename decltype(k)::KeySize_t ki;
    k.Query(&get_loco()->bdbt, &key_blending, &ki, &key_root);
    if (ki != 8) {
      return;
    }
    if (blending) {
      m_current_shader = &m_blending_shader;
    }
    else {
      m_current_shader = &m_shader;
    }
    sb_draw(key_root);
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
    #define vulkan_buffer_count 5
    #define sb_shader_vertex_path graphics/glsl/vulkan/2D/objects/sprite.vert
    #define sb_shader_fragment_path graphics/glsl/vulkan/2D/objects/sprite.frag
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

  /*loco_t::camera_t* get_camera(fan::graphics::cid_t* cid) {
    auto block = sb_get_block(cid);
    loco_t* loco = get_loco();
    return loco->camera_list[*block->p[cid->instance_id].key.get_value<
      instance_properties_t::key_t::get_index_with_type<loco_t::camera_list_NodeReference_t>()
    >()].camera_id;
  }
  void set_camera(fan::graphics::cid_t* cid, loco_t::camera_list_NodeReference_t n) {
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

  void set_image(fan::graphics::cid_t* cid, loco_t::textureid_t<0> n) {
  #if defined(loco_opengl)
    sb_set_key<bm_properties_t::key_t::get_index_with_type<loco_t::textureid_t<0>>()>(cid, n);
  #else
    auto loco = get_loco();
    auto& img = loco->image_list[n];
    // maybe samplers etc need to update too
    m_ssbo.m_descriptor.m_properties[2].image_infos[img.texture_index.sprite].imageView = img.image_view;
    m_ssbo.m_descriptor.update(loco->get_context(), 1, 2, 1, img.texture_index.sprite);

  #endif
  }

  properties_t get_properties(loco_t::cid_t* cid) {
    properties_t p = sb_get_properties(cid);
    p.camera = get_loco()->camera_list[*p.key.get_value<loco_t::camera_list_NodeReference_t>()].camera_id;
    p.viewport = get_loco()->get_context()->viewport_list[*p.key.get_value<fan::graphics::viewport_list_NodeReference_t>()].viewport_id;
    //loco_t::image_t image;
    //image.texture_reference = ;
    //*
    p.image = get_loco()->image_list[*p.key.get_value<loco_t::textureid_t<0>>()].image;
    return p;
  }

  //void set_viewport_value(fan::graphics::cid_t* cid, fan::vec2 p, fan::vec2 s) {
  //  loco_t* loco = get_loco();
  //  auto block = sb_get_block(cid);
  //  loco->get_context()->viewport_list[*block->p[cid->instance_id].key.get_value<2>()].viewport_id->set(loco->get_context(), p, s, loco->get_window()->get_size());
  //  //sb_set_key(cid, &properties_t::image, n);
  //}

  #if defined(loco_vulkan)
    uint32_t m_texture_index = 0;
    uint32_t m_camera_index = 0;
  #endif
};

#undef vulkan_buffer_count
#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_sprite_name