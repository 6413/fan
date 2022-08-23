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
  root = loco_bdbt_NewNode(&loco->bdbt);
  bll_block_open(&blocks);
  shape_bm_open(&bm_list);
  

  m_shader.open(loco->get_context());
  m_shader.set_vertex(
    loco->get_context(),
    #include sb_shader_vertex_path
  );
  m_shader.set_fragment(
    loco->get_context(),
    #include sb_shader_fragment_path
  );
  m_shader.compile(loco->get_context());
}
void sb_close() {
  loco_t* loco = get_loco();

  bll_block_close(&blocks);
  shape_bm_close(&bm_list);

  assert(0);
  //loco_bdbt_close(&loco->bdbt);

  m_shader.close(loco->get_context());

  //for (uint32_t i = 0; i < blocks.size(); i++) {
  //  blocks[i].uniform_buffer.close(loco->get_context());
  //}
}

struct block_t;

block_t* sb_push_back(fan::opengl::cid_t* cid, properties_t& p) {
  loco_t* loco = get_loco();
 
  loco_bdbt_NodeReference_t nr = root;
  loco_bdbt_Key_t<sizeof(instance_properties_t::key_t) * 8> k;
  typename decltype(k)::KeySize_t ki;
  k.Query(&loco->bdbt, &p.instance_properties.key, &ki, &nr);
  if (ki != sizeof(instance_properties_t::key_t) * 8) {
    auto lnr = shape_bm_NewNode(&bm_list);
    auto ln = shape_bm_GetNodeByReference(&bm_list, lnr);
    ln->data.first_block = bll_block_NewNodeLast(&blocks);
    bll_block_GetNodeByReference(&blocks, ln->data.first_block)->data.block.open(loco, this);
    ln->data.last_block = ln->data.first_block;
    ln->data.instance_properties = p.instance_properties;
    k.InFrom(&loco->bdbt, &p.instance_properties.key, ki, nr, lnr.NRI);
    nr = lnr.NRI;
  }

  instance_t it = p;

  shape_bm_Node_t* bmn = shape_bm_GetNodeByReference(&bm_list, *(shape_bm_NodeReference_t*)&nr);
  block_t* last_block = &bll_block_GetNodeByReference(&blocks, bmn->data.last_block)->data.block;

  if (last_block->uniform_buffer.size() == max_instance_size) {
    auto nnr = bll_block_NewNode(&blocks);
    bll_block_linkNext(&blocks, bmn->data.last_block, nnr);
    bmn->data.last_block = nnr;
    last_block = &bll_block_GetNodeByReference(&blocks, bmn->data.last_block)->data.block;
    last_block->open(loco, this);
  }

  block_t* block = last_block;

  block->uniform_buffer.push_ram_instance(loco->get_context(), it);

  const uint32_t instance_id = block->uniform_buffer.size() - 1;

  block->cid[instance_id] = cid;

  block->uniform_buffer.common.edit(
    loco->get_context(),
    &loco->m_write_queue,
    instance_id * sizeof(instance_t),
    instance_id * sizeof(instance_t) + sizeof(instance_t)
  );

  cid->bm_id = ((shape_bm_NodeReference_t*)&nr)->NRI;
  cid->block_id = bmn->data.last_block.NRI;
  cid->instance_id = instance_id;

  block->p[instance_id] = p.instance_properties;
  return block;
}
void sb_erase(fan::opengl::cid_t* cid) {
  loco_t* loco = get_loco();
  auto bm_id = *(shape_bm_NodeReference_t*)&cid->bm_id;
  auto bm_node = shape_bm_GetNodeByReference(&bm_list, bm_id);

  auto block_id = *(bll_block_NodeReference_t*)&cid->block_id;
  auto block_node = bll_block_GetNodeByReference(&blocks, *(bll_block_NodeReference_t*)&cid->block_id);
  auto block = &block_node->data.block;

  auto& last_block_id = bm_node->data.last_block;
  auto* last_block_node = bll_block_GetNodeByReference(&blocks, last_block_id);
  block_t* last_block = &last_block_node->data.block;
  uint32_t last_instance_id = last_block->uniform_buffer.size() - 1;

  if (bll_block_IsNodeReferenceEqual(block_id, last_block_id) && cid->instance_id == block->uniform_buffer.size() - 1) {
    block->uniform_buffer.common.m_size -= sizeof(instance_t);
    if (block->uniform_buffer.size() == 0) {
      auto lpnr = block_node->PrevNodeReference;
      block->close(loco);
      bll_block_Unlink(&blocks, block_id);
      bll_block_Recycle(&blocks, block_id);
      if (bll_block_IsNodeReferenceEqual(last_block_id, bm_node->data.first_block)) {
        loco_bdbt_Key_t<sizeof(instance_properties_t::key_t) * 8> k;
        typename decltype(k)::KeySize_t ki;
        k.Remove(&loco->bdbt, &bm_node->data.instance_properties.key, root);
        shape_bm_Recycle(&bm_list, bm_id);
      }
      else {
        //fan::print("here");
        last_block_id = lpnr;
      }
    }
  //  fan::print(shape_bm_usage(&bm_list));
    return;
  }

  instance_t* last_instance_data = last_block->uniform_buffer.get_instance(loco->get_context(), last_instance_id);

  block->uniform_buffer.copy_instance(
    loco->get_context(),
    cid->instance_id,
    last_instance_data
  );

  last_block->uniform_buffer.common.m_size -= sizeof(instance_t);

  block->p[cid->instance_id] = last_block->p[last_instance_id];

  block->cid[cid->instance_id] = last_block->cid[last_instance_id];
  block->cid[cid->instance_id]->block_id = block_id.NRI;
  block->cid[cid->instance_id]->instance_id = cid->instance_id;

  if (last_block->uniform_buffer.size() == 0) {
    auto lpnr = last_block_node->PrevNodeReference;

    last_block->close(loco);
    bll_block_Unlink(&blocks, last_block_id);
    bll_block_Recycle(&blocks, last_block_id);

    bm_node->data.last_block = lpnr;
  }

  block->uniform_buffer.common.edit(
    loco->get_context(),
    &loco->m_write_queue,
    cid->instance_id * sizeof(instance_t),
    cid->instance_id * sizeof(instance_t) + sizeof(instance_t)
  );
}

block_t* sb_get_block(fan::opengl::cid_t* cid) {
  loco_t* loco = get_loco();
  auto block_id = *(bll_block_NodeReference_t*)&cid->block_id;
  auto block_node = bll_block_GetNodeByReference(&blocks, *(bll_block_NodeReference_t*)&cid->block_id);
  return &block_node->data.block;
}

template <typename T>
T get(fan::opengl::cid_t *cid, T instance_t::*member) {
  loco_t* loco = get_loco();
  auto block = sb_get_block(cid);
  return block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id)->*member;
}
template <typename T, typename T2>
void set(fan::opengl::cid_t *cid, T instance_t::*member, const T2& value) {
  loco_t* loco = get_loco();
  auto block = sb_get_block(cid);
  block->uniform_buffer.edit_instance(loco->get_context(), cid->instance_id, member, value);
  block->uniform_buffer.common.edit(
    loco->get_context(),
    &loco->m_write_queue,
    cid->instance_id * sizeof(instance_t) + fan::ofof<instance_t, T>(member),
    cid->instance_id * sizeof(instance_t) + fan::ofof<instance_t, T>(member) + sizeof(T)
  );
}

void set_vertex(const std::string& str) {
  loco_t* loco = get_loco();
  m_shader.set_vertex(loco->get_context(), str);
}
void set_fragment(const std::string& str) {
  loco_t* loco = get_loco();
  m_shader.set_fragment(loco->get_context(), str);
}
void compile() {
  loco_t* loco = get_loco();
  m_shader.compile(loco->get_context());
}

template <uint32_t depth = 0>
void traverse_draw(auto nr, uint32_t draw_mode) {
  loco_t* loco = get_loco();
  if constexpr(depth == instance_properties_t::key_t::count + 1) {
    auto bmn = shape_bm_GetNodeByReference(&bm_list, *(shape_bm_NodeReference_t*)&nr);
    auto bnr = bmn->data.first_block;

    while(1) {
      auto node = bll_block_GetNodeByReference(&blocks, bnr);
      node->data.block.uniform_buffer.bind_buffer_range(
        loco->get_context(), 
        node->data.block.uniform_buffer.size()
      );

      node->data.block.uniform_buffer.draw(
        loco->get_context(),
        0 * sb_vertex_count,
        node->data.block.uniform_buffer.size() * sb_vertex_count,
        draw_mode
      );
      if (bll_block_IsNodeReferenceEqual(bnr, bmn->data.last_block)) {
        break;
      }
      bnr = node->NextNodeReference;
    }
  }
  else {
    loco_bdbt_Key_t<sizeof(typename instance_properties_t::key_t::get_type<depth>::type) * 8> k;
    typename decltype(k)::Traverse_t kt;
    kt.init(nr);
    typename instance_properties_t::key_t::get_type<depth>::type o;
    while(kt.Traverse(&loco->bdbt, &o)) {
      loco->process_block_properties_element(this, o);
      traverse_draw<depth + 1>(kt.Output, draw_mode);
    }
  }
}

void sb_draw(uint32_t draw_mode = fan::opengl::GL_TRIANGLES) {
  loco_t* loco = get_loco();
  m_shader.use(loco->get_context());
  traverse_draw(root, draw_mode);
}

void sb_set_key(fan::opengl::cid_t* cid, auto properties_t::*member, auto value) {
  loco_t* loco = get_loco();
  auto block = sb_get_block(cid);
  properties_t p;
  *(instance_t*)&p = *block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id);
  p.instance_properties = block->p[cid->instance_id];
  p.*member = value;
  sb_erase(cid);
  sb_push_back(cid, p);
}

fan::shader_t m_shader;

struct block_t {
  void open(loco_t* loco, auto* shape) {
    uniform_buffer.open(loco->get_context());
    uniform_buffer.init_uniform_block(loco->get_context(), shape->m_shader.id, "instance_t");
  }
  void close(loco_t* loco) {
    uniform_buffer.close(loco->get_context(), &loco->m_write_queue);
  }

  fan::opengl::core::uniform_block_t<instance_t, max_instance_size> uniform_buffer;
  fan::opengl::cid_t* cid[max_instance_size];
  instance_properties_t p[max_instance_size];
};

protected:

loco_bdbt_NodeReference_t root;

#define BLL_set_prefix bll_block
#define BLL_set_BaseLibrary 1
#define BLL_set_Link 1
#define BLL_set_StoreFormat 1
#define BLL_set_StoreFormat1_alloc_open malloc
#define BLL_set_StoreFormat1_alloc_close free
#define BLL_set_type_node uint16_t
#define BLL_set_node_data \
    block_t block;
#include _FAN_PATH(BLL/BLL.h)

bll_block_t blocks;

#define BLL_set_prefix shape_bm
#define BLL_set_BaseLibrary 1
#define BLL_set_Link 0
#define BLL_set_type_node uint16_t
#define BLL_set_node_data \
    bll_block_NodeReference_t first_block; \
    bll_block_NodeReference_t last_block; \
    instance_properties_t instance_properties;
#include _FAN_PATH(BLL/BLL.h)

shape_bm_t bm_list;

public:

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_vertex_count