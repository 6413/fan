struct id_t{
  id_t(fan::opengl::cid_t* cid) {
    block = cid->id / max_instance_size;
    instance = cid->id % max_instance_size;
  }
  uint32_t block;
  uint32_t instance;
};

void open(fan::opengl::context_t* context) {

  m_shader.open(context);
  m_shader.set_vertex(
    context,
    #include sb_shader_vertex_path
  );
  m_shader.set_fragment(
    context,
    #include sb_shader_fragment_path
  );
  m_shader.compile(context);

  m_draw_node_reference = fan::uninitialized;

  blocks.open();
}
void close(fan::opengl::context_t* context) {
  m_shader.close(context);
  for (uint32_t i = 0; i < blocks.size(); i++) {
    blocks[i].uniform_buffer.close(context);
  }
}

void push_back(fan::opengl::context_t* context, fan::opengl::cid_t* cid, const properties_t& p) {
  instance_t it = p;

  uint32_t block_id = blocks.size() - 1;

  if (block_id == (uint32_t)-1 || blocks[block_id].uniform_buffer.size() == max_instance_size) {
    blocks.push_back({});
    block_id++;
    blocks[block_id].uniform_buffer.open(context);
    blocks[block_id].uniform_buffer.init_uniform_block(context, m_shader.id, "instance_t");
  }

  blocks[block_id].uniform_buffer.push_ram_instance(context, it);

  const uint32_t instance_id = blocks[block_id].uniform_buffer.size() - 1;

  blocks[block_id].cid[instance_id] = cid;

  blocks[block_id].uniform_buffer.common.edit(
    context,
    instance_id * sizeof(instance_t),
    instance_id * sizeof(instance_t) + sizeof(instance_t)
  );

  cid->id = block_id * max_instance_size + instance_id;

  blocks[block_id].p[instance_id] = p.block_properties;
}
void erase(fan::opengl::context_t* context, fan::opengl::cid_t* cid) {

  uint32_t block_id = cid->id / max_instance_size;
  uint32_t instance_id = cid->id % max_instance_size;

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
      blocks[block_id].uniform_buffer.close(context);
      blocks.m_size -= 1;
    }
    return;
  }

  uint32_t last_block_id = blocks.size() - 1;
  uint32_t last_instance_id = blocks[last_block_id].uniform_buffer.size() - 1;

  instance_t* last_instance_data = blocks[last_block_id].uniform_buffer.get_instance(context, last_instance_id);

  blocks[block_id].uniform_buffer.edit_ram_instance(
    context,
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
    blocks[last_block_id].uniform_buffer.close(context);
    blocks.m_size -= 1;
  }

  blocks[block_id].uniform_buffer.common.edit(
    context,
    instance_id * sizeof(instance_t),
    instance_id * sizeof(instance_t) + sizeof(instance_t)
  );
}

template <typename T>
T get(fan::opengl::context_t* context, const id_t& id, T instance_t::*member) {
  return blocks[id.block].uniform_buffer.get_instance(context, id.instance)->*member;
}
template <typename T, typename T2>
void set(fan::opengl::context_t* context, const id_t& id, T instance_t::*member, const T2& value) {
  blocks[id.block].uniform_buffer.edit_ram_instance(context, id.instance, (T*)&value, fan::ofof<instance_t, T>(member), sizeof(T));
  blocks[id.block].uniform_buffer.common.edit(
    context,
    id.instance * sizeof(instance_t) + fan::ofof<instance_t, T>(member),
    id.instance * sizeof(instance_t) + fan::ofof<instance_t, T>(member) + sizeof(T)
  );
}

void enable_draw(fan::opengl::context_t* context) {

  #if fan_debug >= fan_debug_low
  if (m_draw_node_reference != fan::uninitialized) {
    fan::throw_error("trying to call enable_draw twice");
  }
  #endif

  m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
}

void disable_draw(fan::opengl::context_t* context) {
  #if fan_debug >= fan_debug_low
  if (m_draw_node_reference == fan::uninitialized) {
    fan::throw_error("trying to disable unenabled draw call");
  }
  #endif
  context->disable_draw(m_draw_node_reference);
}

void set_vertex(fan::opengl::context_t* context, const std::string& str) {
  m_shader.set_vertex(context, str);
}
void set_fragment(fan::opengl::context_t* context, const std::string& str) {
  m_shader.set_fragment(context, str);
}
void compile(fan::opengl::context_t* context) {
  m_shader.compile(context);
}

fan::shader_t m_shader;

struct block_t {
  fan::opengl::core::uniform_block_t<instance_t, max_instance_size> uniform_buffer;
  fan::opengl::cid_t* cid[max_instance_size];
  block_properties_t p[max_instance_size];
};
uint32_t m_draw_node_reference;

fan::hector_t<block_t> blocks;

#undef sb_shader_vertex_path
#undef sb_shader_fragment_path