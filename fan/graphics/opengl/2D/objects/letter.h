struct letter_t {

  static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::letter;

  struct vi_t {
    loco_t::position3_t position; 
    f32_t outline_size;
    fan::vec2 size;
    fan::vec2 tc_position;
    fan::color color = fan::colors::white;
    fan::color outline_color;
    fan::vec2 tc_size;
    f32_t angle=0; 
    f32_t pad[1];
  };

  static constexpr uint32_t max_instance_size = fan::min(256ull, 4096 / (sizeof(vi_t) / 4));

  struct context_key_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::camera_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct cid_t;

  struct ri_t {
    f32_t font_size;
    uint32_t letter_id;
    bool blending = true;
  };

  struct properties_t : vi_t, ri_t, context_key_t {
    using type_t = letter_t;
    loco_t::camera_t* camera = &gloco->default_camera->camera;
    loco_t::viewport_t* viewport = &gloco->default_camera->viewport;
  };

  void push_back(loco_t::cid_nt_t& id, properties_t& p) {

    get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;

    fan::font::character_info_t si = gloco->font.info.get_letter_info(p.letter_id, p.font_size);
    p.tc_position = si.glyph.position / gloco->font.image.size;
    p.tc_size.x = si.glyph.size.x / gloco->font.image.size.x;
    p.tc_size.y = si.glyph.size.y / gloco->font.image.size.y;

    p.size = si.metrics.size / 2;
    #if defined(loco_vulkan)
      VkDescriptorImageInfo imageInfo{};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      auto& img = loco->image_list[loco->font.image.texture_reference];
      imageInfo.imageView = img.image_view;
      imageInfo.sampler = img.sampler;
      if (img.texture_index.letter == (decltype(img.texture_index.letter))-1) {
        img.texture_index.letter = m_texture_index++;
        if (m_texture_index > fan::vulkan::max_textures) {
          fan::throw_error("too many textures max:" + fan::vulkan::max_textures);
        }
       m_ssbo.m_descriptor.m_properties[2].image_infos[img.texture_index.letter] = imageInfo;
        m_ssbo.m_descriptor.update(loco->get_context(), 1, 2);
      }

      auto& camera = loco->camera_list[p.camera];
      if (camera.camera_index.letter == (decltype(camera.camera_index.letter))-1) {
        camera.camera_index.letter = m_camera_index++;
        m_shader.set_camera(loco, camera.camera_id, camera.camera_index.letter);
      }
    #endif

    sb_push_back(id, p);
  }
  void erase(loco_t::cid_nt_t& id) {
    sb_erase(id);
  }

  void draw(const redraw_key_t &redraw_key, loco_bdbt_NodeReference_t key_root) {
    if (redraw_key.blending) {
      m_current_shader = &m_blending_shader;
    }
    else {
      m_current_shader = &m_shader;
    }
    gloco->process_block_properties_element<0>(this, &gloco->font.image);
    sb_draw(key_root);
  }

   #if defined(loco_opengl)
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/letter.vs)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/letter.fs)
  #endif

  letter_t() {
    sb_open(sb_shader_vertex_path, sb_shader_fragment_path);
  }
  ~letter_t() {
    sb_close();
  }

  #include _FAN_PATH(graphics/shape_builder.h)

  void set_camera(loco_t::cid_nt_t& id, loco_t::camera_list_NodeReference_t n) {
    sb_set_context_key<decltype(n)>(id, n);
  }

  void set_viewport(loco_t::cid_nt_t& id, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_context_key<decltype(n)>(id, n);
  }

  properties_t get_properties(loco_t::cid_nt_t& id) {
    properties_t p = sb_get_properties(id);    
    return p;
  }

  #if defined(loco_vulkan)
    uint32_t m_texture_index = 0;
    uint32_t m_camera_index = 0;
  #endif
};