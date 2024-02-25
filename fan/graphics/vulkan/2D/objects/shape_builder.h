#ifndef sb_depth_var
#define sb_depth_var position
#endif

#ifndef sb_vertex_count
#define sb_vertex_count 6
#endif

#ifndef sb_has_own_key_root
#define sb_has_own_key_root 0
#endif
#ifndef sb_ignore_3_key
#define sb_ignore_3_key 0
#endif

#if sb_has_own_key_root == 1
loco_bdbt_NodeReference_t key_root;
#endif

using key_t = fan::masterpiece_t <
  #if sb_ignore_3_key == 0
  loco_t::redraw_key_t,
  uint16_t,
  loco_t::shape_type_t,
  #endif
  context_key_t
>;
static constexpr bool key_equality_assert = fan::assert_equality_v<sizeof(key_t), (
  sizeof(context_key_t) +
  sizeof(loco_t::shape_type_t) * (sb_ignore_3_key == 0) +
  sizeof(uint16_t) * (sb_ignore_3_key == 0) +
  sizeof(loco_t::redraw_key_t) * (sb_ignore_3_key == 0)
  )>;

using push_key_t = fan::masterpiece_t <
  #if sb_ignore_3_key == 0
  loco_t::make_push_key_t<loco_t::redraw_key_t>,
  loco_t::make_push_key_t<uint16_t, true>,
  loco_t::make_push_key_t<loco_t::shape_type_t>,
  #endif
  loco_t::make_push_key_t<context_key_t>
>;

struct block_element_t {
  key_t key;
  uint8_t vi[sizeof(vi_t)];
  ri_t ri;
};


protected:

using ssbo_t = fan::vulkan::core::ssbo_t<vi_t, ri_t, max_instance_size, vulkan_buffer_count>;

loco_bdbt_NodeReference_t root;

#define BLL_set_CPP_ConstructDestruct
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix shape_bm
#include _FAN_PATH(fan_bll_preset.h)
#define BLL_set_Link 0
#define BLL_set_type_node uint16_t
#define BLL_set_NodeData \
  ssbo_t::nr_t first_ssbo_nr; \
  ssbo_t::nr_t last_ssbo_nr; \
  key_t key; \
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
  auto& context = gloco->get_context();
  root = loco_bdbt_NewNode(&gloco->bdbt);

  m_shader.open(context, &gloco->m_write_queue);
  m_shader.set_vertex(context,
    STRINGIFY_DEFINE(sb_shader_vertex_path),
    #include _FAN_PATH(sb_shader_vertex_path)
  );
  m_shader.set_fragment(context, 
    STRINGIFY_DEFINE(sb_shader_vertex_path),
    #include _FAN_PATH(sb_shader_fragment_path)
  );

  m_ssbo.open(context);

  #include _FAN_PATH(graphics/shape_open_settings.h)

  m_ssbo.open_descriptors(context, gloco->descriptor_pool.m_descriptor_pool, ds_properties);
  m_ssbo.m_descriptor.update(context, 2, ds_offset);
  ds_properties[1].buffer = m_shader.get_shader().projection_view_block.common.memory[context.currentFrame].buffer;
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
  //T::m_camera_index = 0
}

template <typename T>
void reset_texture(T x) {
  //T::m_texture_index = 0
}

void sb_close() {
  
  auto& context = gloco->get_context();

  m_shader.close(context, &gloco->m_write_queue);

  m_ssbo.close(context, &gloco->m_write_queue);

  if constexpr(fan::_has_camera_id_t<decltype(*this)>::value) {
    reset_camera(this);
  }
  if constexpr(fan::_has_texture_id_t<decltype(*this)>::value) {
    reset_texture(this);
  }

  //uint32_t index = fan::ofof<loco_t>() - offsetof(loco_t, )
  //for (uint8_t i = 0; i < gloco->camera_list.Usage(); ++i) {
  //  ((uint8_t *)&gloco->camera_list[*(loco_t::camera_list_NodeReference_t*)&i].camera_index)[index]
  //}

  //vkDestroyDescriptorSetLayout(gloco->get_context().device, descriptorSetLayout, nullptr);

  //for (uint32_t i = 0; i < blocks.size(); i++) {
  //  blocks[i].camera_indexuniform_buffer.close(gloco->get_context());
  //}
}

struct block_t;

void suck_block_element(loco_t::cid_nt_t& id, auto&&... suckers) {

  auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
  auto& bm_node = bm_list[bm_id];

  auto block_id = *(ssbo_t::nr_t*)&id->block_id;
  auto instance_id = id->instance_id;

  if constexpr (sizeof...(suckers)) {

    block_element_t* block_element = fan::get_variadic_element<0>(suckers...);

    vi_t* current_vi = &m_ssbo.instance_list.get_vi(block_id, instance_id);
    ri_t* current_ri = &m_ssbo.instance_list.get_ri(block_id, instance_id);

    block_element->key = bm_node.key;

    std::memcpy(
      &block_element->vi,
      current_vi,
      sizeof(fan::get_variadic_element<0>(suckers...)->vi)
    );
    std::memcpy(
      &block_element->ri,
      current_ri,
      sizeof(fan::get_variadic_element<0>(suckers...)->ri)
    );
  }

  auto last_block_id = bm_node.last_ssbo_nr;
  uint32_t last_instance_id = (bm_node.total_instances - 1) % max_instance_size;
  if (block_id == last_block_id && instance_id == last_instance_id) {
    bm_node.total_instances--;
    if (bm_node.total_instances % max_instance_size == 0) {
      auto prev_block_id = m_ssbo.instance_list.GetNodeByReference(
        block_id,
        m_ssbo.multiple_type_link_index
      )->PrevNodeReference;
      m_ssbo.instance_list.unlrec(block_id);
      if (bm_node.first_ssbo_nr == bm_node.last_ssbo_nr) {
        loco_bdbt_Key_t<sizeof(context_key_t::key_t) * 8> k;
        typename decltype(k)::KeySize_t ki;
        k.r(&gloco->bdbt, &bm_node.key, root);
        bm_list.Recycle(bm_id);
      }
      else {
        bm_node.last_ssbo_nr = prev_block_id;
      }
    }
    return;
  }
  else {
    m_ssbo.copy_instance(
      gloco->get_context(),
      &gloco->m_write_queue,
      *(ssbo_t::nr_t*)&last_block_id,
      *(ssbo_t::instance_id_t*)&last_instance_id,
      *(ssbo_t::nr_t*)&block_id,
      *(ssbo_t::instance_id_t*)&instance_id
    );

    if (bm_node.total_instances % max_instance_size == 1) {
      auto prev_block_id = m_ssbo.instance_list.GetNodeByReference(
        last_block_id,
        m_ssbo.multiple_type_link_index
      )->PrevNodeReference;
      m_ssbo.instance_list.unlrec(last_block_id);

      bm_node.last_ssbo_nr = prev_block_id;
    }

    bm_node.total_instances--;
  }
}

void unsuck_block_element(
  loco_t::cid_nt_t& id,
  block_element_t& block_element
) {
  shape_bm_NodeReference_t bm_id;
  uint16_t block_id;

  push_key_t key;
  key.iterate([&](auto i, const auto& data) {
    *data = { .data = *block_element.key.template get_value<i.value>() };
    });

  get_block_id_from_push(bm_id, block_id, key);
  auto& bm = bm_list[bm_id];

  auto ssbo_nr = m_ssbo.add(gloco->get_context(), &gloco->m_write_queue);
  const auto instance_id = bm.total_instances % max_instance_size;
  {
    m_ssbo.copy_instance(gloco->get_context(), &gloco->m_write_queue, ssbo_nr, instance_id, (vi_t*)block_element.vi);
    ri_t& ri = m_ssbo.instance_list.get_ri(ssbo_nr, instance_id);
    ri = std::move(std::move(block_element.ri));
  }

  id->bm_id = *(uint16_t*)&bm_id;
  id->block_id = *(uint16_t*)&ssbo_nr;
  id->instance_id = instance_id;
  id->shape_type = (std::underlying_type<decltype(shape_type)>::type)shape_type;

  // do we need it
  //block->p[instance_id] = *(instance_properties_t*)&p;

  bm.total_instances++;

  //const uint32_t instance_id = block->uniform_buffer.size() - 1;

//  block->id[instance_id] = id;

  //block->ri[instance_id] = std::move(block_element.ri);
}

shape_bm_NodeReference_t push_new_bm(auto& key) {
  auto lnr = bm_list.NewNode();
  auto ln = &bm_list[lnr];
  ln->first_ssbo_nr = m_ssbo.add(gloco->get_context(), &gloco->m_write_queue);
  m_ssbo.instance_list.LinkAsLast(ln->first_ssbo_nr);// do we need this
  ln->last_ssbo_nr = ln->first_ssbo_nr;
  ln->total_instances = 0;
  //ln->last_block = ln->first_block;
  ln->key.iterate([&](auto i, const auto& data) {
    *data = key.template get_value<i.value>()->data;
  });
  return lnr;
}

constexpr void iterate_keys(uint16_t max_depth, uint16_t depth, auto& key, auto& bm_id, auto old_o) {
  if (depth == max_depth) {
    return;
  }
  if (depth != max_depth - 1) {
    auto o = loco_bdbt_NewNode(&gloco->bdbt);
    key.get_value(depth, [&](const auto& data) {
      data->k.a(&gloco->bdbt, &data->data, 0, old_o, o);
      });
    iterate_keys(max_depth, depth + 1, key, bm_id, o);
  }
  else {
    bm_id = push_new_bm(key);
    key.get_value(depth, [&](const auto& data) {
      data->k.a(&gloco->bdbt, &data->data, 0, old_o, bm_id.NRI);
      });
  }
};

void sb_erase_key_from(loco_t::cid_nt_t& id) {
  auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
  auto bm_node = bm_list.GetNodeByReference(bm_id);

  loco_bdbt_NodeReference_t key_nr =
    #if sb_has_own_key_root == 1
    key_root
    #else
    gloco->root
    #endif
    ;

  fan::masterpiece_t <
    #if sb_ignore_3_key == 0
    loco_t::make_erase_key_t<loco_t::redraw_key_t>,
    loco_t::make_erase_key_t<uint16_t, true>,
    loco_t::make_erase_key_t<loco_t::shape_type_t>,
    #endif
    loco_t::make_erase_key_t<context_key_t::key_t>
  > key{
  #if sb_ignore_3_key == 0
    loco_t::make_erase_key_t<loco_t::redraw_key_t>{.data = {.blending = sb_get_ri(id).blending}},
    loco_t::make_erase_key_t<uint16_t, true>{.data = (uint16_t)sb_get_vi(id).sb_depth_var.z},
    {.data = shape_type },
  #endif
    {.data = bm_node->data.key.get_value<context_key_t>()->key}
  };

  key.iterate([&]<typename T>(const auto & i, const T & data) {
    data->key_nr = key_nr;
    typename std::remove_pointer_t<T>::key_t k;
    k.q(&gloco->bdbt, &data->data, &data->key_size, &key_nr);
    #if fan_debug >= 2
    if (data->key_size != sizeof(data->data) * 8) {
      __abort();
    }
    #endif
  });

  key.reverse_iterate_ret([&]<typename T>(auto i, const T & data) -> int {
    typename std::remove_pointer_t<T>::key_t k;
    k.r(&gloco->bdbt, &data->data, data->key_nr);

    if (loco_bdbt_inrhc(&gloco->bdbt, data->key_nr) == true) {
      return 1;
    }
    if constexpr (i.value != 0) { // if its not last to iterate
      loco_bdbt_Recycle(&gloco->bdbt, data->key_nr);
    }
    return 0;
  });
}

void get_block_id_from_push(
  shape_bm_NodeReference_t& bm_id,
  uint16_t& block_id,
  auto& key
) {
  using key_t = std::remove_reference_t<decltype(key)>;

  auto key_nr =
    #if sb_has_own_key_root == 1
    key_root
    #else
    gloco->root
    #endif
    ;

  for (uint32_t i = 0; i < key.size(); ++i) {
    bool do_break = false;
    key.get_value(i, [&](const auto& data) {
      typename decltype(data->k)::KeySize_t ki;
      data->k.q(&gloco->bdbt, &data->data, &ki, &key_nr);
      if (i == key_t::size() - 1) {
        if (ki != sizeof(decltype(data->data)) * 8) {
          bm_id = push_new_bm(key);
          data->k.a(&gloco->bdbt, &data->data, ki, key_nr, bm_id.NRI);
        }
        else {
          bm_id.NRI = key_nr;
        }
      }
      else {
        if (ki != sizeof(decltype(data->data)) * 8) {
          auto o0 = loco_bdbt_NewNode(&gloco->bdbt);
          data->k.a(&gloco->bdbt, &data->data, ki, key_nr, o0);
          iterate_keys(key_t::size(), i + 1, key, bm_id, o0);
          do_break = true;
        }
      }
      });
    if (do_break) {
      break;
    }
  }

  auto& bm = bm_list[bm_id];
  block_id = *(uint16_t*)&bm.last_ssbo_nr;

  if (bm.total_instances && bm.total_instances % max_instance_size == 0) {
    auto new_ssbo_nr = m_ssbo.add(gloco->get_context(), &gloco->m_write_queue);
    m_ssbo.instance_list.linkNext(bm_list[bm_id].last_ssbo_nr, new_ssbo_nr);
    bm_list[bm_id].last_ssbo_nr = new_ssbo_nr;
  }
}

// STRUCT MANUAL PADDING IS REQUIRED (4 byte)
void sb_push_back(loco_t::cid_nt_t& id, auto p) {
    // TODO?
  //p.depth = p.position.z;

  push_key_t key{
  #if sb_ignore_3_key == 0
    loco_t::make_push_key_t<loco_t::redraw_key_t>{.data = {.blending = p.blending}},
    loco_t::make_push_key_t<uint16_t, true>{.data = (uint16_t)p.sb_depth_var.z},
    {.data = shape_type },
  #endif
    {.data = p.key}
  };

  shape_bm_NodeReference_t bm_id;
  uint16_t block_id;
  get_block_id_from_push(bm_id, block_id, key);

  auto bm = &bm_list[bm_id];

  auto ssbo_nr = block_id;

  const auto instance_id = bm->total_instances % max_instance_size;

  ri_t& ri = m_ssbo.instance_list.get_ri(*(ssbo_t::nr_t*)&ssbo_nr, instance_id);
  ri = std::move(*dynamic_cast<ri_t*>(&p));
  m_ssbo.copy_instance(gloco->get_context(), &gloco->m_write_queue, *(ssbo_t::nr_t*)&ssbo_nr, instance_id, (vi_t*)&p);

  id->bm_id = *(uint16_t*)&bm_id;
  id->block_id = ssbo_nr;
  id->instance_id = instance_id;
  id->shape_type = (std::underlying_type<decltype(shape_type)>::type)shape_type;

  // do we need it
  //block->p[instance_id] = *(instance_properties_t*)&p;

  bm->total_instances++;
}
void sb_erase(loco_t::cid_nt_t& id) {  
  suck_block_element(id);
}

vi_t& sb_get_vi(loco_t::cid_nt_t& id) {
  return m_ssbo.instance_list.get_vi(*(ssbo_t::nr_t*)&id->block_id, id->instance_id);
}
template <typename T>
void sb_set_vi(loco_t::cid_nt_t& id, auto T::* member, auto value) {
  sb_get_vi(id).*member = value;
}

ri_t& sb_get_ri(loco_t::cid_nt_t& id) {
  return m_ssbo.instance_list.get_ri(*(ssbo_t::nr_t*)&id->block_id, id->instance_id);
}
template <typename T>
void sb_set_ri(loco_t::cid_nt_t& id, auto T::* member, auto value) {
  sb_get_ri(id).*member = value;
}


//key_t& get_bm_key(loco_t::cid_nt_t& id) {
//  auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
//  auto bm_node = bm_list.GetNodeByReference(bm_id);
//  return bm_node->data.key;
//}
//
//context_key_t& get_context_key(loco_t::cid_nt_t& id) {
//  return *get_bm_key(id).get_value<context_key_t>();
//}
//auto sb_get_block(fan::graphics::cid_t* cid) {
//  auto& block_node = m_ssbo.instance_list.get blocks[*(bll_block_NodeReference_t*)&cid->block_id];
//  return &block_node.block;
//}

template <typename T, typename T2>
auto get(loco_t::cid_nt_t& id, T T2::*member) {
  return sb_get_vi(id).*member;
}
template <typename T, typename T2, typename T3>
void set(loco_t::cid_nt_t& id, T T2::* member, const T3& value) {
  if constexpr (std::is_same_v<T, loco_t::position3_t>) {
    if constexpr (std::is_same_v<T3, fan::vec3>)
      if (value.z != get(id, member).z) {
        sb_set_depth(id, value.z);
      }
  }
  m_ssbo.copy_instance(gloco->get_context(), &gloco->m_write_queue, *(ssbo_t::nr_t*)&id->block_id, id->instance_id, member, value);
}

key_t& get_bm_key(loco_t::cid_nt_t& id) {
  auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
  auto bm_node = bm_list.GetNodeByReference(bm_id);
  return bm_node->data.key;
}

context_key_t& get_context_key(loco_t::cid_nt_t& id) {
  return *get_bm_key(id).get_value<context_key_t>();
}

template <typename T = void>
loco_t::camera_t* get_camera(loco_t::cid_nt_t& id) requires fan::has_camera_t<properties_t> {
  return gloco->camera_list[*get_context_key(id).key.get_value<loco_t::camera_list_NodeReference_t>()].camera_id;
}
template <typename T = void>
void set_camera(loco_t::cid_nt_t& id, loco_t::camera_t* camera) requires fan::has_camera_t<properties_t> {
  sb_set_context_key<loco_t::camera_list_NodeReference_t>(id, camera);
}

template <typename T = void>
loco_t::viewport_t* get_viewport(loco_t::cid_nt_t& id) requires fan::has_viewport_t<properties_t> {
  return gloco->get_context().viewport_list[*get_context_key(id).key.get_value<fan::graphics::viewport_list_NodeReference_t>()].viewport_id;
}
template <typename T = void>
void set_viewport(loco_t::cid_nt_t& id, loco_t::viewport_t* viewport) requires fan::has_viewport_t<properties_t> {
  sb_set_context_key<fan::graphics::viewport_list_NodeReference_t>(id, viewport);
}

template <typename T = void>
loco_t::image_t* get_image(loco_t::cid_nt_t& id) requires fan::has_image_t<properties_t> {
  properties_t p;
  loco_t::image_t* ptr = nullptr;
  [&id, &ptr] <typename T2>(T2 & p, auto * This) mutable {
    if constexpr (fan::has_image_t<T2>) {
      auto nr = This->get_context_key(id).key.template get_value<loco_t::textureid_t<0>>();
      if constexpr (std::is_same_v< std::remove_reference_t<decltype(*nr)>, loco_t::textureid_t<0>>) {
        ptr = gloco->image_list[*(loco_t::textureid_t<0>*)nr].image;
      }
    }
  }(p, this);
  return ptr;
}

template <uint32_t depth = 0>
void traverse_draw(auto nr) {
  
  auto& context = gloco->get_context();
  if constexpr (depth == context_key_t::key_t::count + 1) {
    auto bmn = bm_list.GetNodeByReference(*(shape_bm_NodeReference_t*)&nr);
    auto bnr = bmn->data.first_ssbo_nr;

    while (1) {
      auto node = m_ssbo.instance_list.GetNodeByReference(bnr, 1);

      gloco->m_write_queue.process(context);

      uint32_t count = max_instance_size;
      if (bnr == bmn->data.last_ssbo_nr) {
        count = (bmn->data.total_instances - 1) % max_instance_size + 1;
      }

      context.bindless_draw(
        sb_vertex_count,
        count,
        (uint32_t)bnr.NRI * max_instance_size
      );

      if (bnr == bmn->data.last_ssbo_nr) {
        break;
      }
      bnr = node->NextNodeReference;
    }
  }
  else {
    //loco_bdbt_Key_t<sizeof(typename instance_properties_t::key_t::get_type<depth>::type) * 8> k;
    typename loco_bdbt_Key_t<sizeof(typename context_key_t::key_t::get_type<depth>::type) * 8>::Traverse_t kt;
    kt.i(nr);
    typename context_key_t::key_t::get_type<depth>::type o;
    while (kt.t(&gloco->bdbt, &o)) {
      gloco->process_block_properties_element(this, o);
      traverse_draw<depth + 1>(kt.Output);
    }
  }
}

void sb_draw(loco_bdbt_NodeReference_t key_nr, uint32_t draw_mode = 0) {
  gloco->get_context().bind_draw(
    m_pipeline,
    1,
    &m_ssbo.m_descriptor.m_descriptor_set[gloco->get_context().currentFrame]
  );
  traverse_draw(key_nr);
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

properties_t sb_get_properties(loco_t::cid_nt_t& id) {
  //auto block = sb_get_block(id);

  auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;

  auto block_id = *(ssbo_t::nr_t*)&id->block_id;

  properties_t p;
  *(context_key_t*)&p = get_context_key(id);
  *(vi_t*)&p = m_ssbo.instance_list.get_vi(block_id, id->instance_id);
  *(ri_t*)&p = m_ssbo.instance_list.get_ri(block_id, id->instance_id);

  [&id] <typename T>(T & p, auto * This) {
    if constexpr (fan::has_camera_t<T>) {
      p.camera = This->get_camera(id);
    }
  }(p, this);

  [&id] <typename T>(T & p, auto * This) {
    if constexpr (fan::has_viewport_t<T>) {
      p.viewport = This->get_viewport(id);
    }
  }(p, this);

  [&id] <typename T>(T & p, auto * This) {
    if constexpr (fan::has_image_t<T>) {
      p.image = This->get_image(id);
    }
  }(p, this);
  return p;
}

template <typename T>
void sb_set_context_key(loco_t::cid_nt_t& id, auto value) {
  block_element_t block_element;
  suck_block_element(id, &block_element);
  *block_element.key.get_value<context_key_t>()->key.get_value<T>() = value;
  unsuck_block_element(id, block_element);
}

void sb_set_depth(loco_t::cid_nt_t& id, f32_t depth) {
  #if sb_ignore_3_key == 0
  block_element_t block_element;
  suck_block_element(id, &block_element);
  ((vi_t*)block_element.vi)->sb_depth_var.z = depth;
  *block_element.key.get_value<1>() = (uint16_t)depth;
  unsuck_block_element(id, block_element);
  #endif
}

ssbo_t m_ssbo;
loco_t::shader_t m_shader;
loco_t::shader_t* m_current_shader = &m_shader;

fan::vulkan::pipeline_t m_pipeline;

#undef vk_sb_ssbo
#undef vk_sb_vp
#undef vk_sb_image

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_vertex_count