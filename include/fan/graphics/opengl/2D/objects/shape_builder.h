void sb_open(loco_t* loco) {

  root = loco_bdbt_NewNode(&loco->bdbt);

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
void sb_close(loco_t* loco) {
  assert(0);
  //loco_bdbt_close(&loco->bdbt);

  m_shader.close(loco->get_context());

  //for (uint32_t i = 0; i < blocks.size(); i++) {
  //  blocks[i].uniform_buffer.close(loco->get_context());
  //}
}

void sb_push_back(loco_t* loco, fan::hector_t<block_t>& blocks, fan::opengl::cid_t* cid, const properties_t& p) {
  instance_t it = p;

  uint32_t block_id = blocks.size() - 1;

  if (block_id == (uint32_t)-1 || blocks[block_id].uniform_buffer.size() == max_instance_size) {
    blocks.push_back({});
    block_id++;
    blocks[block_id].uniform_buffer.open(loco->get_context());
    blocks[block_id].uniform_buffer.init_uniform_block(loco->get_context(), m_shader.id, "instance_t");
  }

  blocks[block_id].uniform_buffer.push_ram_instance(loco->get_context(), it);

  const uint32_t instance_id = blocks[block_id].uniform_buffer.size() - 1;

  blocks[block_id].cid[instance_id] = cid;

  blocks[block_id].uniform_buffer.common.edit(
    loco->get_context(),
    &loco->m_write_queue,
    instance_id * sizeof(instance_t),
    instance_id * sizeof(instance_t) + sizeof(instance_t)
  );

  cid->id = block_id * max_instance_size + instance_id;

  blocks[block_id].p[instance_id] = p.block_properties;
}
void erase(loco_t* loco, fan::opengl::cid_t* cid) {
 /* id_t id(this, cid);
  #if fan_debug >= fan_debug_medium
  if (block_id >= blocks.size()) {
    fan::throw_error("invalid access");
  }
  if (instance_id >= blocks[block_id].uniform_buffer.size()) {
    fan::throw_error("invalid access");
  }
  #endif

  if (block_id == blocks.size() - 1 && instance_id == blocks.ge()->uniform_buffer.size() - 1) {
    blocks[block_id].uniform_buffer.common.m_size -= blocks[block_id].uniform_buffer.common.buffer_bytes_size;
    if (blocks[block_id].uniform_buffer.size() == 0) {
      blocks[block_id].uniform_buffer.close(loco->get_context());
      blocks.m_size -= 1;
    }
    return;
  }

  uint32_t last_block_id = blocks.size() - 1;
  uint32_t last_instance_id = blocks[last_block_id].uniform_buffer.size() - 1;

  instance_t* last_instance_data = blocks[last_block_id].uniform_buffer.get_instance(loco->get_context(), last_instance_id);

  blocks[block_id].uniform_buffer.edit_ram_instance(
    loco->get_context(),
    instance_id,
    last_instance_data,
    0,
    sizeof(instance_t)
  );

  blocks[last_block_id].uniform_buffer.common.m_size -= sizeof(instance_t);

  blocks[block_id].p[instance_id] = blocks[last_block_id].p[last_instance_id];

  blocks[block_id].cid[instance_id] = blocks[last_block_id].cid[last_instance_id];
  blocks[block_id].cid[instance_id]->id = block_id * max_instance_size + instance_id;

  if (blocks[last_block_id].uniform_buffer.size() == 0) {
    blocks[last_block_id].uniform_buffer.close(loco->get_context());
    blocks.m_size -= 1;
  }

  blocks[block_id].uniform_buffer.common.edit(
    loco->get_context(),
    instance_id * sizeof(instance_t),
    instance_id * sizeof(instance_t) + sizeof(instance_t)
  );*/
}

template <typename T>
T get(loco_t* loco, fan::opengl::cid_t *cid, T instance_t::*member) {
  id_t id(this, cid);
  return id.block->uniform_buffer.get_instance(loco->get_context(), id.instance_id)->*member;
}
template <typename T, typename T2>
void set(loco_t* loco, fan::opengl::cid_t *cid, T instance_t::*member, const T2& value) {
  id_t id(this, cid);
  id.block->uniform_buffer.edit_ram_instance(loco->get_context(), id.instance_id, (T*)&value, fan::ofof<instance_t, T>(member), sizeof(T));
  id.block->uniform_buffer.common.edit(
    context,
    id.instance_id * sizeof(instance_t) + fan::ofof<instance_t, T>(member),
    id.instance_id * sizeof(instance_t) + fan::ofof<instance_t, T>(member) + sizeof(T)
  );
}

void set_vertex(loco_t* loco, const std::string& str) {
  m_shader.set_vertex(loco->get_context(), str);
}
void set_fragment(loco_t* loco, const std::string& str) {
  m_shader.set_fragment(loco->get_context(), str);
}
void compile(loco_t* loco) {
  m_shader.compile(loco->get_context());
}

fan::shader_t m_shader;

struct block_t {
  fan::opengl::core::uniform_block_t<instance_t, max_instance_size> uniform_buffer;
  fan::opengl::cid_t* cid[max_instance_size];
  block_properties_t p[max_instance_size];
};

loco_bdbt_NodeReference_t root;

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path