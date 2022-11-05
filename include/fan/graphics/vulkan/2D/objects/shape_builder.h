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

void sb_open() {
  loco_t* loco = get_loco();
  auto context = loco->get_context();
  root = loco_bdbt_NewNode(&loco->bdbt);

  m_shader.open(context, &loco->m_write_queue);
  m_shader.set_vertex(context, sb_shader_vertex_path);
  m_shader.set_fragment(context, sb_shader_fragment_path);

  m_ssbo.open(context);

  #include _FAN_PATH(graphics/shape_open_settings.h)

  m_ssbo.open_descriptors(context, loco->descriptor_pool.m_descriptor_pool, ds_properties);
  ds_properties[1].buffer = m_shader.projection_view_block.common.memory[context->currentFrame].buffer;
  m_ssbo.m_descriptor.m_properties[1] = ds_properties[1];
  m_ssbo.m_descriptor.update(context, 1, 1, 0, 0);

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

#endif

  p.color_blend_attachment_count = std::size(color_blend_attachment);
  p.color_blend_attachment = color_blend_attachment;

  p.descriptor_layout_count = 1;
  p.descriptor_layout = &m_ssbo.m_descriptor.m_layout;
  p.shader = &m_shader;
  p.push_constants_size = sizeof(loco_t::push_constants_t);
  m_pipeline.open(context, p);
}
void sb_close() {
  loco_t* loco = get_loco();
  auto context = loco->get_context();

  assert(0);
  //loco_bdbt_close(&loco->bdbt);

  m_shader.close(context, &loco->m_write_queue);

  m_ssbo.close(context, &loco->m_write_queue);

  assert(0);
  //vkDestroyDescriptorSetLayout(loco->get_context()->device, descriptorSetLayout, nullptr);

  //for (uint32_t i = 0; i < blocks.size(); i++) {
  //  blocks[i].uniform_buffer.close(loco->get_context());
  //}
}

struct block_t;

// STRUCT MANUAL PADDING IS REQUIRED (4 byte)
void sb_push_back(fan::graphics::cid_t* cid, properties_t& p) {
  loco_t* loco = get_loco();

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

  cid->bm_id = ((shape_bm_NodeReference_t*)&nr)->NRI;
  cid->block_id = bm_list[bmID].last_ssbo_nr.NRI;
  cid->instance_id = instance_id;

  // do we need it
  //block->p[instance_id] = *(instance_properties_t*)&p;

  bm_list[bmID].total_instances++;
}
void sb_erase(fan::graphics::cid_t* cid) {
  //loco_t* loco = get_loco();

  //auto bm_id = *(shape_bm_NodeReference_t*)&cid->bm_id;
  //auto bm_node = bm_list.GetNodeByReference(bm_id);

  //auto block_id = *(bll_block_NodeReference_t*)&cid->block_id;
  //auto block_node = blocks.GetNodeByReference(*(bll_block_NodeReference_t*)&cid->block_id);
  //auto block = &block_node->data.block;

  //auto& last_block_id = bm_node->data.last_block;
  //auto* last_block_node = blocks.GetNodeByReference(last_block_id);
  //block_t* last_block = &last_block_node->data.block;
  //uint32_t last_instance_id = (bm_node->data.total_instances - 1) % max_instance_size;

  ////// erase descriptorset from block
  ////assert(0);
  //if (block_id == last_block_id && cid->instance_id == max_instance_size - 1) {
  //  block->ssbo_index -= 1;
  //  if (block->ssbo_index == 0) {
  //    auto lpnr = block_node->PrevNodeReference;

  //    block->close(loco);
  //    blocks.Unlink(block_id);
  //    blocks.Recycle(block_id);
  //    if (last_block_id == bm_node->data.first_block) {
  //      loco_bdbt_Key_t<sizeof(instance_properties_t::key_t) * 8> k;
  //      typename decltype(k)::KeySize_t ki;
  //      k.Remove(&loco->bdbt, &bm_node->data.instance_properties.key, root);
  //      bm_list.Recycle(bm_id);
  //    }
  //    else {
  //      last_block_id = lpnr;
  //    }
  //  }
  //  return;
  //}
  //instance_t* last_instance_data = m_ssbo.get_instance(loco->get_context(), last_instance_id);
  //const uint32_t instance_id = (bm_list[bm_id].total_instances - 1) % max_instance_size;

  //m_ssbo.copy_instance(
  //  loco->get_context(),
  //  &loco->m_write_queue,
  //  instance_id + cid->instance_id,
  //  last_instance_data
  //);

  //last_block->ssbo_index -= 1;

  //block->p[instance_id] = last_block->p[last_instance_id];

  //block->cid[instance_id] = last_block->cid[last_instance_id];
  //block->cid[instance_id]->block_id = block_id.NRI;
  //block->cid[instance_id]->instance_id = instance_id;

  //if (last_block->ssbo_index == 0) {
  //  auto lpnr = last_block_node->PrevNodeReference;

  //  last_block->close(loco);
  //  blocks.Unlink(last_block_id);
  //  blocks.Recycle(last_block_id);

  //  bm_node->data.last_block = lpnr;
  //}
}

//block_t* sb_get_block(fan::graphics::cid_t* cid) {
//  auto& block_node = blocks[*(bll_block_NodeReference_t*)&cid->block_id];
//  return &block_node.block;
//}

//auto get(fan::graphics::cid_t *cid, auto instance_t::*member) {
//  loco_t* loco = get_loco();
//  auto block = sb_get_block(cid);
//  return m_ssbo.get_instance(loco->get_context(), block->ssbo_index + cid->instance_id)->*member;
//}
//template <typename T, typename T2>
//void set(fan::graphics::cid_t *cid, T instance_t::*member, const T2& value) {
//  loco_t* loco = get_loco();
//  auto block = sb_get_block(cid);
//  m_ssbo.edit_instance(loco->get_context(), &loco->m_write_queue, block->ssbo_index + cid->instance_id, member, value);
//}

void set_vertex(const fan::string& str) {
  loco_t* loco = get_loco();
  m_shader.set_vertex(loco->get_context(), str);
}
void set_fragment(const fan::string& str) {
  loco_t* loco = get_loco();
  m_shader.set_fragment(loco->get_context(), str);
}

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

      context->draw(
        sb_vertex_count,
        bnr == bmn->data.last_ssbo_nr ? bmn->data.total_instances % (max_instance_size + 1) : max_instance_size,
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

template <uint32_t i>
void sb_set_key(fan::graphics::cid_t* cid, auto value) {
  loco_t* loco = get_loco();
  auto block = sb_get_block(cid);
  properties_t p;
  *(instance_t*)&p = *m_ssbo.get_instance(loco->get_context(), block->ssbo_index + cid->instance_id % max_instance_size);
  *(bm_propeties_t*)&p = block->p[cid->instance_id];
  *p.key.get_value<i>() = value;
  sb_erase(cid);
  sb_push_back(cid, p);
}

using ssbo_t = fan::vulkan::core::ssbo_t<vi_t, ri_t, max_instance_size, vulkan_buffer_count>;

ssbo_t m_ssbo;
fan::graphics::shader_t m_shader;

fan::vulkan::pipeline_t m_pipeline;

protected:

  loco_bdbt_NodeReference_t root;

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix shape_bm
  #define BLL_set_BaseLibrary 1
  #define BLL_set_Link 0
  #define BLL_set_type_node uint16_t
  #define BLL_set_node_data \
    ssbo_t::nr_t first_ssbo_nr; \
    ssbo_t::nr_t last_ssbo_nr; \
    bm_properties_t bm_properties; \
    uint32_t total_instances;
  #include _FAN_PATH(BLL/BLL.h)

  shape_bm_t bm_list;

public:

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_vertex_count