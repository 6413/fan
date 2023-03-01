#ifndef sb_get_loco
#define sb_get_loco \
  loco_t* get_loco() { \
    loco_t* loco = OFFSETLESS(this, loco_t, sb_shape_var_name); \
    return loco; \
  }
#endif

sb_get_loco

#ifndef sb_vertex_count
#define sb_vertex_count 6
#endif


protected:

using ssbo_t = fan::vulkan::core::ssbo_t<vi_t, ri_t, max_instance_size, vulkan_buffer_count>;

loco_bdbt_NodeReference_t root;

#define BLL_set_CPP_ConstructDestruct
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix shape_bm
#define BLL_set_BaseLibrary 1
#define BLL_set_Link 0
#define BLL_set_type_node uint16_t
#define BLL_set_NodeData \
  ssbo_t::nr_t first_ssbo_nr; \
  ssbo_t::nr_t last_ssbo_nr; \
  bm_properties_t bm_properties; \
  uint32_t total_instances;
#include _FAN_PATH(BLL/BLL.h)

shape_bm_t bm_list;

public:

struct cid_t {
  union {
    struct {
			shape_bm_NodeReference_t bm_id;
			ssbo_t::nr_t block_id;
			uint8_t instance_id;
    };
    uint64_t filler;
  };
};

void sb_open() {
  loco_t* loco = get_loco();
  auto context = loco->get_context();
  root = loco_bdbt_NewNode(&loco->bdbt);

  m_shader.open(context, &loco->m_write_queue);
  m_shader.set_vertex(context,
    STRINGIFY(sb_shader_vertex_path),
    #include _FAN_PATH(sb_shader_vertex_path)
  );
  m_shader.set_fragment(context, 
    STRINGIFY(sb_shader_vertex_path),
    #include _FAN_PATH(sb_shader_fragment_path)
  );

  m_ssbo.open(context);

  #include _FAN_PATH(graphics/shape_open_settings.h)

  m_ssbo.open_descriptors(context, loco->descriptor_pool.m_descriptor_pool, ds_properties);
  m_ssbo.m_descriptor.update(context, 2, ds_offset);
  ds_properties[1].buffer = m_shader.projection_view_block.common.memory[context->currentFrame].buffer;
  m_ssbo.m_descriptor.m_properties[1] = ds_properties[1];
  m_ssbo.m_descriptor.update(context, 1, 1, 0, 0);

  // only for rectangle
  #if defined(loco_wboit) && defined(vk_shape_wboit)
    m_ssbo.m_descriptor.update(context, 2, 2);
  #endif

  fan::vulkan::pipeline_t::properties_t p;

#if defined (loco_wboit)
  VkPipelineColorBlendAttachmentState color_blend_attachment[2]{};
	color_blend_attachment[0].colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT |
		VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT |
		VK_COLOR_COMPONENT_A_BIT
	;
	color_blend_attachment[0].blendEnable = VK_TRUE;
	color_blend_attachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
	color_blend_attachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
	color_blend_attachment[0].colorBlendOp = VK_BLEND_OP_ADD;
	color_blend_attachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	color_blend_attachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	color_blend_attachment[0].alphaBlendOp = VK_BLEND_OP_ADD;

	color_blend_attachment[1].colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT
	;

	color_blend_attachment[1].blendEnable = VK_TRUE;
	color_blend_attachment[1].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
	color_blend_attachment[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
	color_blend_attachment[1].colorBlendOp = VK_BLEND_OP_ADD;
	color_blend_attachment[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	color_blend_attachment[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
	color_blend_attachment[1].alphaBlendOp = VK_BLEND_OP_ADD;
  p.depth_test_compare_op = VK_COMPARE_OP_ALWAYS;
#else

  VkPipelineColorBlendAttachmentState color_blend_attachment[2]{};
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

  color_blend_attachment[1] = color_blend_attachment[0];

#endif

  p.color_blend_attachment_count = std::size(color_blend_attachment);
  p.color_blend_attachment = color_blend_attachment;

  p.descriptor_layout_count = 1;
  p.descriptor_layout = &m_ssbo.m_descriptor.m_layout;
  p.shader = &m_shader;
  p.push_constants_size = sizeof(loco_t::push_constants_t);
#if defined(sb_vertex_count) && sb_vertex_count == 2
  p.shape_type = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
#endif
  m_pipeline.open(context, p);
}

template <typename T>
void reset_camera(T x) {
  T::m_camera_index = 0
}

template <typename T>
void reset_texture(T x) {
  T::m_texture_index = 0
}

void sb_close() {
  loco_t* loco = get_loco();
  auto context = loco->get_context();

  m_shader.close(context, &loco->m_write_queue);

  m_ssbo.close(context, &loco->m_write_queue);

  if constexpr(fan::_has_camera_id_t<decltype(*this)>::value) {
    reset_camera(this);
  }
  if constexpr(fan::_has_texture_id_t<decltype(*this)>::value) {
    reset_texture(this);
  }

  //uint32_t index = fan::ofof<loco_t>() - offsetof(loco_t, )
  //for (uint8_t i = 0; i < loco->camera_list.Usage(); ++i) {
  //  ((uint8_t *)&loco->camera_list[*(loco_t::camera_list_NodeReference_t*)&i].camera_index)[index]
  //}

  //vkDestroyDescriptorSetLayout(loco->get_context()->device, descriptorSetLayout, nullptr);

  //for (uint32_t i = 0; i < blocks.size(); i++) {
  //  blocks[i].uniform_buffer.close(loco->get_context());
  //}
}

struct block_t;

// STRUCT MANUAL PADDING IS REQUIRED (4 byte)
void sb_push_back(fan::vulkan::cid_t* fcid, auto p) {
  auto cid = (cid_t*)fcid;
  loco_t* loco = get_loco();

  #if defined (loco_line)
  if constexpr (std::is_same<decltype(p), loco_t::line_t::properties_t>::value) {
    p.src.z -= loco_t::camera_t::znearfar - 1;
    p.dst.z -= loco_t::camera_t::znearfar - 1;
  }
  else {
    p.position.z -= loco_t::camera_t::znearfar - 1;
  }
  #else
    p.position.z -= loco_t::camera_t::znearfar - 1;
  #endif

  p.depth = p.position.z;

  loco_bdbt_NodeReference_t nr = root;
  shape_bm_NodeReference_t& bmID = *(shape_bm_NodeReference_t*)&nr;
  loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k;
  typename decltype(k)::KeySize_t ki;
  k.Query(&loco->bdbt, &p.key, &ki, &nr);
  
  if (ki != sizeof(bm_properties_t::key_t) * 8) {
    auto lnr = bm_list.NewNode();
    auto ln = &bm_list[lnr];
    ln->first_ssbo_nr = m_ssbo.add(loco->get_context(), &loco->m_write_queue);
    m_ssbo.instance_list.LinkAsLast(ln->first_ssbo_nr);
    ln->last_ssbo_nr = ln->first_ssbo_nr;
    ln->bm_properties = *(bm_properties_t*)&p;
    ln->total_instances = 0;
    k.InFrom(&loco->bdbt, &p.key, ki, nr, lnr.NRI);

    nr = lnr.NRI;
  }
  else if (bm_list[bmID].total_instances % max_instance_size == 0) {
    auto new_ssbo_nr = m_ssbo.add(loco->get_context(), &loco->m_write_queue);
    m_ssbo.instance_list.linkNext(bm_list[bmID].last_ssbo_nr, new_ssbo_nr);
    bm_list[bmID].last_ssbo_nr = new_ssbo_nr;
  }

  auto bm = &bm_list[bmID];

  auto ssbo_nr = bm->last_ssbo_nr;

  const auto instance_id = bm->total_instances % max_instance_size;

  ri_t& ri = m_ssbo.instance_list.get_ri(ssbo_nr, instance_id);
  m_ssbo.copy_instance(loco->get_context(), &loco->m_write_queue, ssbo_nr, instance_id, (vi_t*)&p);

  ri.cid = cid;
  cid->bm_id = *((shape_bm_NodeReference_t*)&nr);
  cid->block_id = bm_list[bmID].last_ssbo_nr;
  cid->instance_id = instance_id;

  // do we need it
  //block->p[instance_id] = *(instance_properties_t*)&p;

  bm_list[bmID].total_instances++;
}
void sb_erase(fan::graphics::cid_t* fcid) {
  auto cid = (cid_t*)fcid;
  loco_t* loco = get_loco();

  auto bm_id = cid->bm_id;
  auto bm = &bm_list[bm_id];

  auto block_id = cid->block_id;
  //auto block_node = blocks.GetNodeByReference(*(bll_block_NodeReference_t*)&cid->block_id);
  //auto block = &block_node->data.block;

  auto& last_block_id = bm->last_ssbo_nr;
  //auto* last_block_node = blocks.GetNodeByReference(last_block_id);
  //block_t* last_block = &last_block_node->data.block;
  uint32_t last_instance_id = (bm->total_instances - 1) % max_instance_size;
  static int x = 0;
  if (block_id == last_block_id && cid->instance_id == last_instance_id) {
    bm->total_instances--;
    if (bm->total_instances % max_instance_size == 0) {
			auto prev_block_id = m_ssbo.instance_list.GetNodeByReference(
				block_id,
				m_ssbo.multiple_type_link_index
			)->PrevNodeReference;
      m_ssbo.instance_list.unlrec(block_id);
      if (bm->first_ssbo_nr == bm->last_ssbo_nr) {
				loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k;
				typename decltype(k)::KeySize_t ki;
				k.Remove(&loco->bdbt, &bm->bm_properties.key, root);
				bm_list.Recycle(bm_id);
      }
      else {
        bm->last_ssbo_nr = prev_block_id;
      }
    }
    return;
  }

  m_ssbo.copy_instance(
    loco->get_context(),
    &loco->m_write_queue,
    last_block_id,
    last_instance_id,
    cid->block_id,
    cid->instance_id
  );

  if (bm->total_instances % max_instance_size == 1) {
    auto prev_block_id = m_ssbo.instance_list.GetNodeByReference(
			last_block_id,
			m_ssbo.multiple_type_link_index
		)->PrevNodeReference;
    m_ssbo.instance_list.unlrec(last_block_id);

    bm->last_ssbo_nr = prev_block_id;
  }

  bm->total_instances--;

  auto& ri = m_ssbo.instance_list.get_ri(cid->block_id, cid->instance_id);
  ri.cid->block_id = block_id;
  ri.cid->instance_id = cid->instance_id;
}

vi_t& sb_get_vi(fan::graphics::cid_t* fcid) {
  auto cid = (cid_t*)fcid;
  return m_ssbo.instance_list.get_vi(ssbo_t::nr_t{cid->block_id}, cid->instance_id);
}
template <typename T>
void sb_set_vi(fan::graphics::cid_t* cid, auto T::* member, auto value) {
  sb_get_vi(cid).*member = value;
}

ri_t& sb_get_ri(fan::graphics::cid_t* fcid) {
  auto cid = (cid_t*)fcid;
  return m_ssbo.instance_list.get_ri(ssbo_t::nr_t{cid->block_id}, cid->instance_id);
}
template <typename T>
void sb_set_ri(fan::graphics::cid_t* fcid, auto T::* member, auto value) {
  sb_get_ri(fcid).*member = value;
}

//auto sb_get_block(fan::graphics::cid_t* cid) {
//  auto& block_node = m_ssbo.instance_list.get blocks[*(bll_block_NodeReference_t*)&cid->block_id];
//  return &block_node.block;
//}

template <typename T, typename T2>
auto get(fan::graphics::cid_t *cid, T T2::*member) {
  loco_t* loco = get_loco();

#if defined(loco_line)
  if constexpr (std::is_same_v<T2, loco_t::line_t::vi_t>) {
    if constexpr (std::is_same_v<decltype(member), decltype(&T2::src)> ||
                  std::is_same_v<decltype(member), decltype(&T2::dst)>) {
      return sb_get_vi(cid).*member + fan::vec3(0, 0, loco_t::camera_t::znearfar - 1);
    }
  }
  else {
#endif
    if constexpr (std::is_same_v<decltype(member), decltype(&T2::position)>) {
      return sb_get_vi(cid).*member + fan::vec3(0, 0, loco_t::camera_t::znearfar - 1);
    }
#if defined(loco_line)
  }
#endif
  return sb_get_vi(cid).*member;
}
template <typename T, typename T2>
void set(fan::graphics::cid_t *fcid, T T2::*member, const auto& value) {
  loco_t* loco = get_loco();
  auto cid = (cid_t*)fcid;
  
  #define one_line \
    m_ssbo.copy_instance( \
      loco->get_context(), \
      &loco->m_write_queue, \
      cid->block_id, \
      cid->instance_id, \
      member, \
      fan::vec3(value) - fan::vec3(0, 0, loco_t::camera_t::znearfar - 1) \
    );

  #if defined(loco_line)
  if constexpr (std::is_same_v<T2, loco_t::line_t::vi_t>) {
    if constexpr (std::is_same_v<decltype(member), decltype(&T2::src)> ||
                  std::is_same_v<decltype(member), decltype(&T2::dst)>) {
      one_line
    }
    else {
      m_ssbo.copy_instance(loco->get_context(), &loco->m_write_queue, cid->block_id, cid->instance_id, member, value);
    }
  }
  if constexpr (!std::is_same_v<T2, loco_t::line_t::vi_t>) {
  #endif
    if constexpr (std::is_same_v<decltype(member), decltype(&T2::position)>) {
      one_line
    }
    else {
      m_ssbo.copy_instance(loco->get_context(), &loco->m_write_queue, cid->block_id, cid->instance_id, member, value);
    }
#if defined(loco_line)
  }
#endif
  #undef one_line
}

template <typename T = void>
loco_t::camera_t* get_camera(fan::graphics::cid_t* cid) requires fan::has_camera_t<properties_t> {
  auto ri = sb_get_ri(cid);
  loco_t* loco = get_loco();
  return loco->camera_list[*ri.key.get_value<
    bm_properties_t::key_t::get_index_with_type<loco_t::camera_list_NodeReference_t>()
  >()].camera_id;
}
template <typename T = void>
fan::graphics::viewport_t* get_viewport(fan::graphics::cid_t* cid) requires fan::has_viewport_t<properties_t> {
  auto ri = sb_get_ri(cid);
  loco_t* loco = get_loco();
  auto idx = *ri.key.get_value<
    bm_properties_t::key_t::get_index_with_type<fan::graphics::viewport_list_NodeReference_t>()
  >();
  return loco->get_context()->viewport_list[idx].viewport_id;
}

//void set_vertex(const fan::string& str) {
//  loco_t* loco = get_loco();
//  m_shader.set_vertex(loco->get_context(), str);
//}
//void set_fragment(const fan::string& str) {
//  loco_t* loco = get_loco();
//  m_shader.set_fragment(loco->get_context(), str);
//}

template <uint32_t depth = 0>
void traverse_draw(auto nr) {
  loco_t* loco = get_loco();
  auto context = loco->get_context();
  if constexpr (depth == bm_properties_t::key_t::count + 1) {
    auto bmn = bm_list.GetNodeByReference(*(shape_bm_NodeReference_t*)&nr);
    auto bnr = bmn->data.first_ssbo_nr;

    while (1) {
      auto node = m_ssbo.instance_list.GetNodeByReference(bnr, 1);

      loco->m_write_queue.process(context);

      uint32_t count = max_instance_size;
      if (bnr == bmn->data.last_ssbo_nr) {
        count = (bmn->data.total_instances - 1) % max_instance_size + 1;
      }

      context->draw(
        sb_vertex_count,
        count,
        (uint32_t)bnr.NRI * max_instance_size,
        m_pipeline,
        1,
        &m_ssbo.m_descriptor.m_descriptor_set[context->currentFrame]
      );

      if (bnr == bmn->data.last_ssbo_nr) {
        break;
      }
      bnr = node->NextNodeReference;
    }
  }
  else {
    //loco_bdbt_Key_t<sizeof(typename instance_properties_t::key_t::get_type<depth>::type) * 8> k;
    typename loco_bdbt_Key_t<sizeof(typename bm_properties_t::key_t::get_type<depth>::type) * 8>::Traverse_t kt;
    kt.init(nr);
    typename bm_properties_t::key_t::get_type<depth>::type o;
    #if fan_use_uninitialized == 0
    memset(&o, 0, sizeof(o));
    #endif
    while (kt.Traverse(&loco->bdbt, &o)) {
      loco->process_block_properties_element(this, o);
      traverse_draw<depth + 1>(kt.Output);
    }
  }
}

void sb_draw(uint32_t draw_mode = 0) {
  traverse_draw(root);
}

template <typename T>
static void line_move_z(T& x) {
  x.src.z += loco_t::camera_t::znearfar - 1;
  x.dst.z += loco_t::camera_t::znearfar - 1;
}

template <typename T>
static void else_move_z(T& x) {
  x.position.z += loco_t::camera_t::znearfar - 1;
}

template <uint32_t i>
void sb_set_key(fan::graphics::cid_t* fcid, auto value) {
  auto cid = (cid_t*)fcid;
  loco_t* loco = get_loco();
  properties_t p;
  *(vi_t*)&p = m_ssbo.instance_list.get_vi(ssbo_t::nr_t{cid->block_id}, cid->instance_id);
  *(ri_t*)&p = m_ssbo.instance_list.get_ri(ssbo_t::nr_t{cid->block_id}, cid->instance_id);
  *p.key.get_value<i>() = value;
  #if defined (loco_line)
  if constexpr (std::is_same_v<decltype(p), loco_t::line_t::properties_t>) {
    line_move_z(p);
  }
  else {
    else_move_z(p);
  }
  #else
    else_move_z(p);
  #endif
  sb_erase(fcid);
  sb_push_back(fcid, p);
}

ssbo_t m_ssbo;
fan::graphics::shader_t m_shader;

fan::vulkan::pipeline_t m_pipeline;

#undef vk_sb_ssbo
#undef vk_sb_vp
#undef vk_sb_image

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_vertex_count