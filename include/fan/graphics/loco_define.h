#pragma once

loco_t::loco_t(loco_t::properties_t p) 
    #ifdef loco_window
    :
    window(fan::vec2(800, 800)),
    #endif
    #if defined(loco_context)
    context(
      #if defined(loco_window)
      get_window()
      #endif
    )
    #endif
    #if defined(loco_window)
    , unloaded_image(this, fan::webp::image_info_t{ (void*)pixel_data, 1 })
    #endif
  {
    #if defined(loco_window)

   // set_vsync(p.vsync);

    get_window()->add_buttons_callback([this](const mouse_buttons_cb_data_t& d) {
      fan::vec2 window_size = get_window()->get_size();
    feed_mouse_button(d.button, d.state, get_mouse_position());
      });

    get_window()->add_keys_callback([&](const keyboard_keys_cb_data_t& d) {
      feed_keyboard(d.key, d.state);
      });

    get_window()->add_mouse_move_callback([&](const mouse_move_cb_data_t& d) {
      feed_mouse_move(get_mouse_position());
      });

    get_window()->add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
      feed_text(d.character);
      });
    #endif
    #if defined(loco_opengl)
    fan::print("RENDERER BACKEND: OPENGL");
    #elif defined(loco_vulkan)
    fan::print("RENDERER BACKEND: VULKAN");
    #endif

    #if defined(loco_letter)
    font.open(this, loco_font);
    #endif

    #if defined(loco_post_process)
    fan::opengl::core::renderbuffer_t::properties_t rp;
    rp.size = get_window()->get_size();
    if (post_process.open(rp)) {
      fan::throw_error("failed to initialize frame buffer");
    }
    #endif

  #if defined(loco_opengl)
    loco_t::image_t::load_properties_t lp;
    lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;
  #if defined(loco_framebuffer)
    m_framebuffer.open(get_context());
    m_framebuffer.bind(get_context());
  #endif
  #endif

  #if defined(loco_opengl)

  #if defined(loco_framebuffer)

	  fan::webp::image_info_t ii;
	  ii.data = nullptr;
    ii.size = get_window()->get_size();

    lp.internal_format = fan::opengl::GL_RGBA;
    lp.format = fan::opengl::GL_RGBA;
    lp.min_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
    lp.mag_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
    lp.type = fan::opengl::GL_FLOAT;

    color_buffers[0].load(this, ii, lp);
    get_context()->opengl.call(get_context()->opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

    color_buffers[0].bind_texture(this);
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      *color_buffers[0].get_texture(this),
      fan::opengl::GL_COLOR_ATTACHMENT0
    );

    lp.internal_format = fan::opengl::GL_RGBA16F;
    lp.format = fan::opengl::GL_RGBA;

    color_buffers[1].load(this, ii, lp);

    color_buffers[1].bind_texture(this);
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      *color_buffers[1].get_texture(this),
      fan::opengl::GL_COLOR_ATTACHMENT1
    );

    get_context()->opengl.call(get_context()->opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

    get_window()->add_resize_callback([this](const auto& d) {
      loco_t::image_t::load_properties_t lp;
      lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;

      fan::webp::image_info_t ii;
	    ii.data = nullptr;
      ii.size = get_window()->get_size();

      lp.internal_format = fan::opengl::GL_RGBA;
      lp.format = fan::opengl::GL_RGBA;
      lp.type = fan::opengl::GL_FLOAT;
      lp.min_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
      lp.mag_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;

      color_buffers[0].reload_pixels(this, ii, lp);

      color_buffers[0].bind_texture(this);
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        *color_buffers[0].get_texture(this),
        fan::opengl::GL_COLOR_ATTACHMENT0
      );

      get_context()->opengl.call(get_context()->opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

      color_buffers[1].reload_pixels(this, ii, lp);

      color_buffers[1].bind_texture(this);
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        *color_buffers[1].get_texture(this),
        fan::opengl::GL_COLOR_ATTACHMENT1
      );

      get_context()->opengl.call(get_context()->opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

      fan::opengl::core::renderbuffer_t::properties_t rp;
      m_framebuffer.bind(get_context());
      rp.size = ii.size;
      rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
      m_rbo.set_storage(get_context(), rp);
    });

    fan::opengl::core::renderbuffer_t::properties_t rp;
    m_framebuffer.bind(get_context());
    rp.size = ii.size;
    rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
    m_rbo.open(get_context());
    m_rbo.set_storage(get_context(), rp);
    rp.internalformat = fan::opengl::GL_DEPTH_ATTACHMENT;
    m_rbo.bind_to_renderbuffer(get_context(), rp);

    unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

    for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }

    get_context()->opengl.call(get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);
    // finally check if framebuffer is complete
    if (!m_framebuffer.ready(get_context())) {
      fan::throw_error("framebuffer not ready");
    }

    m_framebuffer.unbind(get_context());

    m_fbo_final_shader.open(get_context());
    m_fbo_final_shader.set_vertex(
      get_context(),
      #include _FAN_PATH(graphics/glsl/opengl/2D/effects/loco_fbo.vs)
    );
    m_fbo_final_shader.set_fragment(
      get_context(),
      #include _FAN_PATH(graphics/glsl/opengl/2D/effects/loco_fbo.fs)
    );
    m_fbo_final_shader.compile(get_context());
  #endif
  #endif

  #if defined(loco_vulkan) && defined(loco_window)
    fan::vulkan::pipeline_t::properties_t pipeline_p;

    auto context = get_context();

    render_fullscreen_shader.open(context, &m_write_queue);
    render_fullscreen_shader.set_vertex(
      context, 
      "graphics/glsl/vulkan/2D/objects/loco_fbo.vert", 
      #include _FAN_PATH(graphics/glsl/vulkan/2D/objects/loco_fbo.vert))
    );
    render_fullscreen_shader.set_fragment(
      context, 
      "graphics/glsl/vulkan/2D/objects/loco_fbo.frag", 
      #include _FAN_PATH(graphics/glsl/vulkan/2D/objects/loco_fbo.frag))
    );
    VkDescriptorSetLayout layouts[] = {
    #if defined(loco_line)
      line.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_rectangle)
      rectangle.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_sprite)
      sprite.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_letter)
      letter.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_button)
      button.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_text_box)
      text_box.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_yuv420p)
      yuv420p.m_ssbo.m_descriptor.m_layout,
    #endif
    };
    pipeline_p.descriptor_layout_count = 1;
    pipeline_p.descriptor_layout = layouts;
    pipeline_p.shader = &render_fullscreen_shader;
    pipeline_p.push_constants_size = sizeof(loco_t::push_constants_t);
    pipeline_p.subpass = 1;
    VkDescriptorImageInfo imageInfo{};

    VkPipelineColorBlendAttachmentState color_blend_attachment[1]{};
    color_blend_attachment[0].colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT
		;
    color_blend_attachment[0].blendEnable = VK_TRUE;
    color_blend_attachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment[0].colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment[0].alphaBlendOp = VK_BLEND_OP_ADD;
    pipeline_p.color_blend_attachment_count = std::size(color_blend_attachment);
    pipeline_p.color_blend_attachment = color_blend_attachment;
    pipeline_p.enable_depth_test = false;
    context->render_fullscreen_pl.open(context, pipeline_p);
  #endif

    default_texture.create_missing_texture(this);

    #if defined(loco_rectangle)
      *types.get_value<rectangle_t*>() = &rectangle;
    #endif
    #if defined(loco_sprite)
      *types.get_value<sprite_t*>() = &sprite;
    #endif
    #if defined(loco_button)
      *types.get_value<button_t*>() = &button;
    #endif
    #if defined(loco_text)
      *types.get_value<text_t*>() = &text;
    #endif
    #if defined(loco_light)
      *types.get_value<light_t*>() = &light;
    #endif
    #if defined(loco_t_id_t_types)
      #if !defined(loco_t_id_t_ptrs)
        #error loco_t_id_t_ptrs not defined
      #else
      std::apply([&](const auto&... args) {
        ((*types.get_value<std::remove_const_t<std::remove_reference_t<decltype(args)>>>() = args), ...);
      }, std::tuple<loco_t_id_t_types>{ loco_t_id_t_ptrs });
      #endif
    #endif
  }



fan_create_get_set_define_extra(fan::vec3, position,  
  if (get_position().z != data.z) {
    get_loco()->shape_set_depth(*this, data.z);
  }
, ;);
fan_create_set_define_custom(fan::vec2, position, 
  get_loco()->shape_set_position(*this, fan::vec3(data, get_position().z));
);
fan_create_get_set_define(fan::vec2, size);
fan_create_get_set_define(fan::color, color);
fan_create_get_set_define(f32_t, angle);
fan_create_get_set_define(fan::string, text);
fan_create_get_set_define(fan::vec2, rotation_point);
fan_create_get_set_define(f32_t, font_size);

fan_create_set_define(f32_t, depth);
                   
fan_create_set_define(loco_t::camera_list_NodeReference_t, camera);
fan_create_set_define(fan::graphics::viewport_list_NodeReference_t, viewport);

//

fan_build_get_set_define(fan::vec3, position);
fan_build_get_set_define(fan::vec2, size);
fan_build_get_set_define(fan::color, color);
fan_build_get_set_define(f32_t, angle);
fan_build_get_set_define(fan::vec2, rotation_point);

make_global_function_define(erase,
  if constexpr (has_erase_v<shape_t, loco_t::cid_t*>) {
    (*shape)->erase(cid);
  },
  loco_t::cid_t* cid
);

fan_build_get_set_generic_define(f32_t, font_size);
fan_build_get_set_generic_define(loco_t::camera_list_NodeReference_t, camera);
fan_build_get_set_generic_define(fan::graphics::viewport_list_NodeReference_t, viewport);

fan_build_get_set_generic_define(fan::string, text);

fan_has_function_concept(sb_set_depth);

make_global_function_define(set_depth,
  if constexpr (has_set_depth_v<shape_t, loco_t::cid_t*, f32_t>) { 
    (*shape)->set_depth(cid, data); 
  } 
  else if constexpr (has_sb_set_depth_v<shape_t, loco_t::cid_t*, f32_t>) { 
    (*shape)->sb_set_depth(cid, data); 
  }, 
  loco_t::cid_t* cid, 
  const auto& data 
);


#define make_shape_id_define(name) \
  loco_t::name ## _id_t::name ## _id_t(const properties_t& p) { \
    (loco_access)->name.push_back(*this, *(loco_t::name ## _t::properties_t*)&p); \
  } \
   \
  loco_t::name ## _id_t& loco_t::name ## _id_t::operator[](const properties_t& p) { \
    (loco_access)->name.push_back(*this, *(loco_t::name ## _t::properties_t*)&p); \
    return *this; \
  } \
   \
  loco_t::name ## _id_t::~name##_id_t() { \
    (loco_access)->name.erase(*this); \
  }


#if defined(loco_vfi)
make_shape_id_define(vfi);
#endif

loco_t::cid_nr_t::cid_nr_t(const cid_nr_t& nr) {
  init();
  (loco_access)->cid_list[*this].cid.shape_type = (loco_access)->cid_list[nr].cid.shape_type;
}

loco_t::cid_nr_t::cid_nr_t(cid_nr_t&& nr) {
  NRI = nr.NRI;
  nr.invalidate_soft();
}

loco_t::cid_nr_t& loco_t::cid_nr_t::operator=(const cid_nr_t& id) {
  if (this != &id) {
    init();
    (loco_access)->cid_list[*this].cid.shape_type = (loco_access)->cid_list[id].cid.shape_type;
  }
  return *this;
}

loco_t::cid_nr_t& loco_t::cid_nr_t::operator=(cid_nr_t&& id) {
  if (this != &id) {
    if (!is_invalid()) {
      invalidate();
    }
    NRI = id.NRI;

    id.invalidate_soft();
  }
  return *this;
}


void loco_t::cid_nr_t::init() {
  *(base_t*)this = (loco_access)->cid_list.NewNodeLast();
}

bool loco_t::cid_nr_t::is_invalid() {
  return cid_list_inric(*this);
}

void loco_t::cid_nr_t::invalidate_soft() {
  *(base_t*)this = (loco_access)->cid_list.gnric();
}

void loco_t::cid_nr_t::invalidate() {
  if (is_invalid()) {
    return;
  }
  (loco_access)->cid_list.unlrec(*this);
  *(base_t*)this = (loco_access)->cid_list.gnric();
}

loco_t::id_t::id_t(const auto& properties) {
  cid.init();
  (loco_access)->push_shape(*this, properties);
}

inline loco_t::id_t::id_t(const id_t& id) : cid(id.cid) {
  (loco_access)->shape_get_properties(*(id_t*)&id, [&](const auto& properties) {
    (loco_access)->push_shape(*this, properties);
  });
}
inline loco_t::id_t::id_t(id_t&& id) : cid(std::move(id.cid)) {
  id.cid.invalidate();
}

loco_t::id_t::~id_t() {
  //fan::print((uint32_t)cid.NRI);
  erase();
  cid.invalidate();
}

loco_t::id_t& loco_t::id_t::operator=(const id_t& id) {
  if (this != &id) {
    (loco_access)->shape_get_properties(*(id_t*)&id, [&](const auto& properties) {
      cid.init();
      (loco_access)->push_shape(*this, properties);
    });
  }
  return *this;
}

loco_t::id_t& loco_t::id_t::operator=(id_t&& id) {
  if (this != &id) {
    if (!cid.is_invalid()) {
      erase();
    }
    cid = std::move(id.cid);

    id.cid.invalidate();
  }
  return *this;
}


void loco_t::id_t::erase() {
  if (cid.is_invalid()) {
    return;
  }
  (loco_access)->shape_erase(*this);
  cid.invalidate();
}

loco_t::id_t::operator fan::opengl::cid_t *(){
  return &(loco_access)->cid_list[cid].cid;
}

loco_t* loco_t::id_t::get_loco() {
  return loco_access;
}

template <typename T>
void loco_t::push_shape(cid_t* cid, T properties) {
  if constexpr(!std::is_same_v<std::nullptr_t, T>){
    (*types.get_value<typename T::type_t*>())->push_back(cid, properties);
  }
}

void loco_t::shape_get_properties(loco_t::cid_t* cid, auto lambda) {
  types.iterate([&]<typename T>(auto shape_index, T shape) {
    using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
    if (shape_t::shape_type == cid->shape_type) {
      if constexpr (has_get_properties_v<shape_t, loco_t::cid_t*>) {
          lambda((*shape)->get_properties(cid));
      }
      else if constexpr (has_sb_get_properties_v<shape_t, loco_t::cid_t*>) {
          lambda((*shape)->sb_get_properties(cid));
      }
    }
  }); 
}


#undef loco_access

#undef loco_rectangle_vi_t