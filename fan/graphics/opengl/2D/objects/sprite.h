struct sb_sprite_name {

  #ifndef sb_custom_shape_type
  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::sprite;
  #else
  static constexpr typename loco_t::shape_type_t shape_type = sb_custom_shape_type;
  #endif

  struct vi_t {
    loco_t::position3_t position = 0;
    f32_t parallax_factor = 0;
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 angle = fan::vec3(0);
    int flags = 0;
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;
  };

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t< 
      loco_t::textureid_t<0>, 
      loco_t::camera_list_NodeReference_t, 
      fan::graphics::viewport_list_NodeReference_t 
    >; 
    struct key_t : parsed_masterpiece_t {}key;
  };
  
  struct cid_t;

  struct ri_t {
    bool blending = false;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = sb_sprite_name;
    
    loco_t::image_t* image = &gloco->default_texture;
    loco_t::camera_t* camera = &gloco->default_camera->camera;
    fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport;
    #if defined(loco_opengl)
    bool load_tp(loco_t::texturepack_t::ti_t* ti) {
      auto& im = *ti->image;
      image = &im;
      tc_position = ti->position / im.size;
      tc_size = ti->size / im.size;
      return 0;
    }
    #endif
  };
  void push_back(loco_t::cid_nt_t& id, properties_t p) {

    get_key_value (loco_t::textureid_t<0>) = p.image;
    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;
    sb_push_back(id, p);
    #if defined(loco_vulkan)
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    auto& img = gloco->image_list[p.image];
    imageInfo.imageView = img.image_view;
    imageInfo.sampler = img.sampler;
    if (img.texture_index.sprite == (decltype(img.texture_index.sprite))-1) {
      img.texture_index.sprite = m_texture_index++;
      if (m_texture_index > fan::vulkan::max_textures) {
        fan::throw_error("too many textures max:" + fan::vulkan::max_textures);
      }
      m_ssbo.m_descriptor.m_properties[2].image_infos[img.texture_index.sprite] = imageInfo;
      m_ssbo.m_descriptor.update(gloco->get_context(), 1, 2);
    }

    auto& camera = gloco->camera_list[p.camera];
    if (camera.camera_index.sprite == (decltype(camera.camera_index.sprite))-1) {
      camera.camera_index.sprite = m_camera_index++;
      m_shader.set_camera(camera.camera_id, camera.camera_index.sprite);
    }
    #endif
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  void draw(const redraw_key_t &redraw_key, loco_bdbt_NodeReference_t key_root) {
    if (redraw_key.blending) {
      //m_current_shader = &m_blending_shader;
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
      #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.vs)
    #endif
    #ifndef sb_shader_fragment_path
      #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.fs)
    #endif
  #elif defined(loco_vulkan)
  #define vulkan_buffer_count 5
  #ifndef sb_shader_fragment_path
    #define sb_shader_vertex_path graphics/glsl/vulkan/2D/objects/sprite.vert
  #endif
  #ifndef sb_shader_fragment_path
    #define sb_shader_fragment_path graphics/glsl/vulkan/2D/objects/sprite.frag
  #endif
  #define vk_sb_ssbo
  #define vk_sb_vp
  #define vk_sb_image
  #endif
  
  sb_sprite_name() {
    #if defined(loco_opengl)
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
    #elif defined(loco_vulkan)
    sb_open();
    #endif
  }
  ~sb_sprite_name() {
    sb_close();
  }

  #include _FAN_PATH(graphics/shape_builder.h)

  void set_image(loco_t::cid_nt_t& id, loco_t::textureid_t<0> n) {
  #if defined(loco_opengl)
    sb_set_context_key<loco_t::textureid_t<0>>(id, n);
  #else
    auto& img = gloco->image_list[n];
    // maybe samplers etc need to update too
    m_ssbo.m_descriptor.m_properties[2].image_infos[img.texture_index.sprite].imageView = img.image_view;
    m_ssbo.m_descriptor.update(gloco->get_context(), 1, 2, 1, img.texture_index.sprite);

  #endif
  }

  #if defined(loco_opengl)

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);
    return p; 
  }

  bool load_tp(loco_t::shape_t& id, loco_t::texturepack_t::ti_t* ti) { 
    auto& im = *ti->image;

    set_image(id, &im);
    
    set(id, &vi_t::tc_position, ti->position / im.size);
    set(id, &vi_t::tc_size, ti->size / im.size);

    return 0;
  }
  #endif
  #if defined(loco_vulkan)
  uint32_t m_texture_index = 0;
  uint32_t m_camera_index = 0;
  #endif

};
#undef sb_sprite_name