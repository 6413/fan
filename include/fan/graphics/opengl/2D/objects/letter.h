struct letter_t {

  struct instance_t {
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

  static constexpr uint32_t max_instance_size = fan::min(256ull, 4096 / (sizeof(instance_t) / 4));

  #define hardcode0_t loco_t::matrices_list_NodeReference_t
  #define hardcode0_n matrices
  #define hardcode1_t fan::graphics::viewport_list_NodeReference_t
  #define hardcode1_n viewport
  #include _FAN_PATH(graphics/opengl/2D/objects/hardcode_open.h)

  struct instance_properties_t {
    struct key_t : parsed_masterpiece_t {}key;
    expand_get_functions

    f32_t font_size;
    uint16_t letter_id;
  };

  struct properties_t : instance_t, instance_properties_t {
    properties_t() = default;
    properties_t(const instance_t& i) : instance_t(i) {}
    properties_t(const instance_properties_t& p) : instance_properties_t(p) {}
  };

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
        m_descriptor.m_properties[2].image_infos[img.texture_index.letter] = imageInfo;
      }
      m_descriptor.update(loco->get_context());

      auto& matrices = loco->matrices_list[p.get_matrices()];
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
  #include _FAN_PATH(graphics/shape_builder.h)

  letter_t() {
    #if defined(loco_vulkan)

      auto loco = get_loco();
      auto context = loco->get_context();

      std::array<fan::vulkan::write_descriptor_set_t, vulkan_buffer_count> ds_properties{};

      ds_properties[0].binding = 0;
      ds_properties[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      ds_properties[0].flags = VK_SHADER_STAGE_VERTEX_BIT;
      ds_properties[0].range = VK_WHOLE_SIZE;
      ds_properties[0].common = &m_ssbo.common;
      ds_properties[0].dst_binding = 0;

      ds_properties[1].binding = 1;
      ds_properties[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      ds_properties[1].flags = VK_SHADER_STAGE_VERTEX_BIT;
      ds_properties[1].common = &m_shader.projection_view_block.common;
      ds_properties[1].range = sizeof(fan::mat4) * 2;
      ds_properties[1].dst_binding = 1;

      VkDescriptorImageInfo imageInfo{};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = loco->unloaded_image.get(loco).image_view;
      imageInfo.sampler = loco->unloaded_image.get(loco).sampler;

      ds_properties[2].use_image = 1;
      ds_properties[2].binding = 2;
      ds_properties[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      ds_properties[2].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
      for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
        ds_properties[2].image_infos[i] = imageInfo;
      }
      ds_properties[2].dst_binding = 2; // ?

    #endif
    sb_open(
      #if defined(loco_vulkan)
        ds_properties
      #endif
    );
  }
  ~letter_t() {
    sb_close();
  }

  void set_matrices(fan::graphics::cid_t* cid, loco_t::matrices_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  void set_viewport(fan::graphics::cid_t* cid, fan::graphics::viewport_list_NodeReference_t n) {
    sb_set_key<instance_properties_t::key_t::get_index_with_type<decltype(n)>()>(cid, n);
  }

  #if defined(loco_vulkan)
    uint32_t m_texture_index = 0;
    uint32_t m_matrices_index = 0;
  #endif
};

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)