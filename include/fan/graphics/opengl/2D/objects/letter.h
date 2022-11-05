struct letter_t {

  struct vi_t {
    fan::vec3 position;
    f32_t outline_size;
    fan::vec2 size;
    fan::vec2 tc_position;
    fan::color color = fan::colors::white;
    fan::color outline_color;
    fan::vec2 tc_size;
  private:
    f32_t pad[2];
  };

  static constexpr uint32_t max_instance_size = fan::min(256ull, 4096 / (sizeof(vi_t) / 4));

  struct bm_properties_t {
    using parsed_masterpiece_t = fan::masterpiece_t<
      loco_t::matrices_list_NodeReference_t,
      fan::graphics::viewport_list_NodeReference_t
    >;
    struct key_t : parsed_masterpiece_t {}key;
  };

  struct ri_t {
    fan::graphics::cid_t* cid;
    f32_t font_size;
    uint16_t letter_id;
  };

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  struct properties_t : vi_t, ri_t, bm_properties_t {

    make_key_value(loco_t::matrices_list_NodeReference_t, matrices);
    make_key_value(fan::graphics::viewport_list_NodeReference_t, viewport);

    properties_t() = default;
    properties_t(const vi_t& i) : vi_t(i) {}
    properties_t(const bm_properties_t& p) : bm_properties_t(p) {}
  };

  #undef make_key_value

  void push_back(fan::graphics::cid_t* cid, properties_t& p) {
    loco_t* loco = get_loco();
    fan::font::character_info_t si = loco->font.info.get_letter_info(p.letter_id, p.font_size);

    p.tc_position = si.glyph.position / loco->font.image.size;
    p.tc_size.x = si.glyph.size.x / loco->font.image.size.x;
    p.tc_size.y = si.glyph.size.y / loco->font.image.size.y;

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

      auto& matrices = loco->matrices_list[p.matrices];
      if (matrices.matrices_index.letter == (decltype(matrices.matrices_index.letter))-1) {
        matrices.matrices_index.letter = m_matrices_index++;
        m_shader.set_matrices(loco, matrices.matrices_id, &loco->m_write_queue, matrices.matrices_index.letter);
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

   #if defined(loco_opengl)
    #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/letter.vs)
    #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/letter.fs)
  #elif defined(loco_vulkan)
    #define vulkan_buffer_count 3
    #define sb_shader_vertex_path _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/letter.vert.spv)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/vulkan/2D/objects/letter.frag.spv)
  #endif

  #define vk_sb_ssbo
  #define vk_sb_vp
  #define vk_sb_image
  #include _FAN_PATH(graphics/shape_builder.h)

  letter_t() {
    sb_open();
  }
  ~letter_t() {
    sb_close();
  }

  void set_matrices(fan::graphics::cid_t* cid, loco_t::matrices_list_NodeReference_t n) {
    sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  void set_viewport(fan::graphics::cid_t* cid, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_key<bm_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  #if defined(loco_vulkan)
    uint32_t m_texture_index = 0;
    uint32_t m_matrices_index = 0;
  #endif
};

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)