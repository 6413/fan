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

struct block_t {
  void open(loco_t* loco, auto* shape) {
    uniform_buffer.open(loco->get_context());
    uniform_buffer.init_uniform_block(loco->get_context(), shape->m_shader.id, "instance_t");
  }
  void close(loco_t* loco) {
    uniform_buffer.close(loco->get_context(), &loco->m_write_queue);
  }

  fan::opengl::core::uniform_block_t<vi_t, max_instance_size> uniform_buffer;
  fan::opengl::cid_t* cid[max_instance_size];
  ri_t p[max_instance_size];
};

protected:

loco_bdbt_NodeReference_t root;

#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix bll_block
#define BLL_set_BaseLibrary 1
#define BLL_set_Link 1
#define BLL_set_StoreFormat 1
//#define BLL_set_StoreFormat1_alloc_open malloc
//#define BLL_set_StoreFormat1_alloc_close free
#define BLL_set_type_node uint16_t
#define BLL_set_NodeData \
    block_t block;
#include _FAN_PATH(BLL/BLL.h)

bll_block_t blocks;

#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix shape_bm
#define BLL_set_BaseLibrary 1
#define BLL_set_Link 0
#define BLL_set_type_node uint16_t
#define BLL_set_NodeData \
    bll_block_NodeReference_t first_block; \
    bll_block_NodeReference_t last_block; \
    bm_properties_t instance_properties;
#include _FAN_PATH(BLL/BLL.h)

shape_bm_t bm_list;

public:

struct cid_t {
  union {
    struct {
			shape_bm_NodeReference_t bm_id;
			bll_block_NodeReference_t block_id;
			uint8_t instance_id;
    };
    uint64_t filler;
  };
};

void sb_open() {
  loco_t* loco = get_loco();
  root = loco_bdbt_NewNode(&loco->bdbt);
  blocks.Open();
  bm_list.Open();
  
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

  blocks.Close();
  bm_list.Close();

  //assert(0);
  //loco_bdbt_close(&loco->bdbt);

  m_shader.close(loco->get_context());

  //for (uint32_t i = 0; i < blocks.size(); i++) {
  //  blocks[i].uniform_buffer.close(loco->get_context());
  //}
}

struct block_t;

// STRUCT MANUAL PADDING IS REQUIRED (32 BIT)
block_t* sb_push_back(fan::opengl::cid_t* cid, auto& p) {

  #if defined (loco_line)
  if constexpr (std::is_same<std::remove_reference_t<decltype(p)>, loco_t::line_t::properties_t>::value) {
    p.depth = p.src.z;
  }
  else {
    p.depth = p.position.z;
  }
  #else
    p.depth = p.position.z;
  #endif
  loco_t* loco = get_loco();
  loco_bdbt_NodeReference_t nr = root;
  loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k;
  typename decltype(k)::KeySize_t ki;
  //printf("%08x\n", *(uint32_t*)&p.key);
  k.Query(&loco->bdbt, &p.key, &ki, &nr);

  if (ki != sizeof(bm_properties_t::key_t) * 8) {
    auto lnr = bm_list.NewNode();
    auto ln = &bm_list[lnr];
    ln->first_block = blocks.NewNodeLast();
    blocks[ln->first_block].block.open(loco, this);
    ln->last_block = ln->first_block;
    ln->instance_properties = *(bm_properties_t*)&p;
    k.InFrom(&loco->bdbt, &p.key, ki, nr, lnr.NRI);
    nr = lnr.NRI;
  }

  vi_t it = p;
  shape_bm_Node_t* bmn = bm_list.GetNodeByReference(*(shape_bm_NodeReference_t*)&nr);
  block_t* last_block = &blocks[bmn->data.last_block].block;

  if (last_block->uniform_buffer.size() == max_instance_size) {
    auto nnr = blocks.NewNode();
    blocks.linkNext(bmn->data.last_block, nnr);
    bmn->data.last_block = nnr;
    last_block = &blocks[bmn->data.last_block].block;
    last_block->open(loco, this);
  }
  block_t* block = last_block;
  block->uniform_buffer.push_ram_instance(loco->get_context(), it);

  const uint32_t instance_id = block->uniform_buffer.size() - 1;

  block->cid[instance_id] = cid;

  block->uniform_buffer.common.edit(
    loco->get_context(),
    &loco->m_write_queue,
    instance_id * sizeof(vi_t),
    instance_id * sizeof(vi_t) + sizeof(vi_t)
  );

  cid->bm_id = ((shape_bm_NodeReference_t*)&nr)->NRI;
  cid->block_id = bmn->data.last_block.NRI;
  cid->instance_id = instance_id;

  block->p[instance_id] = *(ri_t*)&p;
  return block;
}
void sb_erase(fan::opengl::cid_t* cid) {
  loco_t* loco = get_loco();
  auto bm_id = *(shape_bm_NodeReference_t*)&cid->bm_id;
  auto bm_node = bm_list.GetNodeByReference(bm_id);

  auto block_id = *(bll_block_NodeReference_t*)&cid->block_id;
  auto block_node = blocks.GetNodeByReference(*(bll_block_NodeReference_t*)&cid->block_id);
  auto block = &block_node->data.block;

  auto& last_block_id = bm_node->data.last_block;
  auto* last_block_node = blocks.GetNodeByReference(last_block_id);
  block_t* last_block = &last_block_node->data.block;
  uint32_t last_instance_id = last_block->uniform_buffer.size() - 1;

  if (block_id == last_block_id && cid->instance_id == block->uniform_buffer.size() - 1) {
    block->uniform_buffer.m_size -= sizeof(vi_t);
    if (block->uniform_buffer.size() == 0) {
      auto lpnr = block_node->PrevNodeReference;
      block->close(loco);
      blocks.Unlink(block_id);
      blocks.Recycle(block_id);
      if (last_block_id == bm_node->data.first_block) {
        loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k;
        typename decltype(k)::KeySize_t ki;
        k.Remove(&loco->bdbt, &bm_node->data.instance_properties.key, root);
        bm_list.Recycle(bm_id);
      }
      else {
        //fan::print("here");
        last_block_id = lpnr;
      }
    }
  //  fan::print(shape_bm_usage(&bm_list));
    return;
  }

  vi_t* last_instance_data = last_block->uniform_buffer.get_instance(loco->get_context(), last_instance_id);

  block->uniform_buffer.copy_instance(
    loco->get_context(),
    &loco->m_write_queue,
    cid->instance_id,
    last_instance_data
  );

  last_block->uniform_buffer.m_size -= sizeof(vi_t);

  block->p[cid->instance_id] = last_block->p[last_instance_id];

  block->cid[cid->instance_id] = last_block->cid[last_instance_id];
  block->cid[cid->instance_id]->block_id = block_id.NRI;
  block->cid[cid->instance_id]->instance_id = cid->instance_id;

  if (last_block->uniform_buffer.size() == 0) {
    auto lpnr = last_block_node->PrevNodeReference;

    last_block->close(loco);
    blocks.Unlink(last_block_id);
    blocks.Recycle(last_block_id);

    bm_node->data.last_block = lpnr;
  }
}

block_t* sb_get_block(fan::opengl::cid_t* cid) {
  auto& block_node = blocks[*(bll_block_NodeReference_t*)&cid->block_id];
  return &block_node.block;
}

//ri_t& sb_get_ri() {
//  return 
//}

template <typename T, typename T2>
T get(fan::opengl::cid_t *cid, T T2::*member) {
  loco_t* loco = get_loco();
  auto block = sb_get_block(cid);
#if defined(loco_line)
  if constexpr (std::is_same_v<T2, loco_t::line_t::vi_t>) {
    if constexpr (std::is_same_v<decltype(member), decltype(&T2::src)> ||
                std::is_same_v<decltype(member), decltype(&T2::dst)>) {
      return block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id)->*member + fan::vec3(0, 0, loco_t::matrices_t::znearfar / 2 - 1);
    }
  }
  else {
#endif
    if constexpr (std::is_same_v<decltype(member), decltype(&T2::position)>) {
      return block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id)->*member + fan::vec3(0, 0, loco_t::matrices_t::znearfar / 2 - 1);
    }
#if defined(loco_line)
  }
#endif
  return block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id)->*member;
}
template <typename T, typename T2>
void set(fan::opengl::cid_t *cid, T T2::*member, const auto& value) {
  loco_t* loco = get_loco();
  auto block = sb_get_block(cid);

  #define one_line \
    block->uniform_buffer.edit_instance( \
      loco->get_context(),  \
      &loco->m_write_queue,  \
      cid->instance_id,  \
      member,  \
      fan::vec3(value) - fan::vec3(0, 0, loco_t::matrices_t::znearfar / 2 - 1) \
    );
#if defined(loco_line)
if constexpr (std::is_same_v<T2, loco_t::line_t::vi_t>) {
  if constexpr (std::is_same_v<decltype(member), decltype(&T2::src)> ||
    std::is_same_v<decltype(member), decltype(&T2::dst)>) {
    one_line
  }
  else {
    block->uniform_buffer.edit_instance(loco->get_context(), &loco->m_write_queue, cid->instance_id, member, value);
  }
}
if constexpr (!std::is_same_v<T2, loco_t::line_t::vi_t>) {
  #endif
  if constexpr (std::is_same_v<decltype(member), decltype(&T2::position)>) {
    one_line
  }
  else {
    block->uniform_buffer.edit_instance(loco->get_context(), &loco->m_write_queue, cid->instance_id, member, value);
  }
  #if defined(loco_line)
}
#endif

#undef one_line

block->uniform_buffer.common.edit(
  loco->get_context(),
  &loco->m_write_queue,
  cid->instance_id * sizeof(vi_t) + fan::ofof(member),
  cid->instance_id * sizeof(vi_t) + fan::ofof(member) + sizeof(T)
);
}

template <typename T = void>
loco_t::matrices_t* get_matrices(fan::graphics::cid_t* cid) requires fan::has_matrices_t<properties_t> {
  auto ri = sb_get_ri(cid);
  loco_t* loco = get_loco();
  return loco->matrices_list[*ri.key.get_value<
    bm_properties_t::key_t::get_index_with_type<loco_t::matrices_list_NodeReference_t>()
  >()].matrices_id;
}

template <typename T = void>
fan::graphics::viewport_t* get_viewport(fan::graphics::cid_t* cid) requires fan::has_viewport_t<properties_t> {
  auto ri = sb_get_ri(cid);
  loco_t* loco = get_loco();
  return loco->get_context()->viewport_list[*ri.key.get_value<
    bm_properties_t::key_t::get_index_with_type<fan::graphics::viewport_list_NodeReference_t>()
  >()].viewport_id;
}

void set_vertex(const fan::string& str) {
  loco_t* loco = get_loco();
  m_shader.set_vertex(loco->get_context(), str);
}
void set_fragment(const fan::string& str) {
  loco_t* loco = get_loco();
  m_shader.set_fragment(loco->get_context(), str);
}
void compile() {
  loco_t* loco = get_loco();
  m_shader.compile(loco->get_context());
}

static inline std::vector<fan::function_t<void()>> draw_queue_helper;
static inline uint16_t zdepth = 0;

template <uint32_t depth = 0>
void traverse_draw(auto nr, uint32_t draw_mode) {
  loco_t* loco = get_loco();
  if constexpr (depth == bm_properties_t::key_t::count + 1) {
    auto bmn = bm_list.GetNodeByReference(*(shape_bm_NodeReference_t*)&nr);
    auto bnr = bmn->data.first_block;

    draw_queue_helper.push_back([this, loco, draw_mode, bmn, bnr]() mutable {
        while (1) {
          auto node = blocks.GetNodeByReference(bnr);
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
          if (bnr == bmn->data.last_block) {
            break;
          }
          bnr = node->NextNodeReference;
        }
      });

    loco->m_draw_queue.insert(loco_t::draw_t{
      // * 5 to dont collide with zdepth
      (uint64_t)zdepth,
      std::vector<fan::function_t<void()>>(draw_queue_helper.begin(), draw_queue_helper.end())
    });
    draw_queue_helper.clear();
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
      // update zdepth here if changes
      if constexpr (std::is_same_v<decltype(o), uint16_t>) {
        zdepth = o;
      }
      draw_queue_helper.push_back([this, loco, o, kt, draw_mode]() {
        loco->process_block_properties_element(this, o);
      });
      traverse_draw<depth + 1>(kt.Output, draw_mode);
    }
  }
}

void sb_draw(uint32_t draw_mode = fan::opengl::GL_TRIANGLES) {
  draw_queue_helper.clear();
  loco_t* loco = get_loco();
  m_shader.use(loco->get_context());
  traverse_draw(root, draw_mode);
}

template <typename T>
static void line_move_z(T& x) {
  x.src.z += loco_t::matrices_t::znearfar / 2 - 1;
  x.dst.z += loco_t::matrices_t::znearfar / 2 - 1;
}

template <typename T>
static void else_move_z(T& x) {
  x.position.z += loco_t::matrices_t::znearfar / 2 - 1;
}

template <uint32_t i>
void sb_set_key(fan::opengl::cid_t* cid, auto value) {
  loco_t* loco = get_loco();
  auto block = sb_get_block(cid);
  properties_t p;
  *(vi_t*)&p = *block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id);
  *(ri_t*)&p = block->p[cid->instance_id];
  *p.key.get_value<i>() = value;
  #if defined(loco_line)
  if constexpr (std::is_same_v<decltype(p), loco_t::line_t::properties_t>) {
    line_move_z(p);
  }
  else {
  #endif
    else_move_z(p);
#if defined(loco_line)
  }
#endif
  sb_erase(cid);
  sb_push_back(cid, p);
}

fan::opengl::shader_t m_shader;

vi_t& sb_get_vi(fan::graphics::cid_t* cid) {
  auto loco = get_loco();
  return *sb_get_block(cid)->uniform_buffer.get_instance(loco->get_context(), cid->instance_id);
}
template <typename T>
void sb_set_vi(fan::graphics::cid_t* cid, auto T::* member, auto value) {
  auto loco = get_loco();
  auto& instance = sb_get_vi(cid);
  instance.*member = value;
  sb_get_block(cid)->uniform_buffer.edit_instance(loco->get_context(), &loco->m_write_queue, cid->instance_id, member, value);
  //sb_get_block(cid)->uniform_buffer.copy_instance(loco->get_context(), &instance);
}

ri_t& sb_get_ri(fan::graphics::cid_t* cid) {
  auto loco = get_loco();
  return sb_get_block(cid)->p[cid->instance_id];
}
template <typename T>
void sb_set_ri(fan::graphics::cid_t* cid, auto T::* member, auto value) {
  sb_get_ri(cid).*member = value;
}

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path
#undef sb_vertex_count