#ifndef sb_depth_var
  #define sb_depth_var position
#endif

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

  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix shape_bm
  #define BLL_set_BaseLibrary 1
  #define BLL_set_IsNodeRecycled 0
  #define BLL_set_Link 0
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeData \
    bll_block_NodeReference_t first_block; \
    bll_block_NodeReference_t last_block; \
    bm_properties_t instance_properties;
  #include _FAN_PATH(BLL/BLL.h)

  shape_bm_t bm_list;

  fan::window_t::resize_callback_NodeReference_t resize_nr;

  #pragma pack(push, 1)
  struct key_pack_t {
    loco_t::redraw_key_t redraw_key;
    uint16_t depth;
    loco_t::shape_type_t::_t shape_type;
    bm_properties_t::key_t shape_key;
  };
  #pragma pack(pop)

public:

  struct cid_t {
    union {
      struct
        #ifdef sb_cid
        : sb_cid
        #endif
      {
        #ifndef sb_cid
        shape_bm_NodeReference_t bm_id;
        bll_block_NodeReference_t block_id;
        uint8_t instance_id;
        #endif
      };
      uint64_t filler;
    };
  };

  void sb_open() {
    loco_t* loco = get_loco();
    blocks.Open();
    bm_list.Open();

    m_shader.open(loco->get_context());
    m_shader.set_vertex(
      loco->get_context(),
      #include sb_shader_vertex_path
    );
    m_shader.set_fragment(
      loco->get_context(),
      #ifndef sb_shader_fragment_string
        #include sb_shader_fragment_path
      #else
        sb_shader_fragment_string
      #endif
    );
    m_shader.compile(loco->get_context());

    m_blending_shader.open(loco->get_context());
    m_blending_shader.set_vertex(
      loco->get_context(),
      #include sb_shader_vertex_path
    );
    fan::string str = 
      #ifndef sb_shader_fragment_string
        #include sb_shader_fragment_path
      #else
        sb_shader_fragment_string
      #endif
    ;

    auto found = str.find("discard;");
    if (found != fan::string::npos) {
      str.erase(found, std::string_view("discard;").size());
    }
    m_blending_shader.set_fragment(
      loco->get_context(),
      str
    );
    m_blending_shader.compile(loco->get_context());

    m_current_shader = &m_shader;

    m_shader.use(loco->get_context());
    m_shader.set_vec2(loco->get_context(), "window_size", loco->get_window()->get_size());
    m_blending_shader.use(loco->get_context());
    m_blending_shader.set_vec2(loco->get_context(), "window_size", loco->get_window()->get_size());

    m_shader.use(loco->get_context());
    m_shader.set_vec3(loco->get_context(), loco_t::lighting_t::ambient_name, loco->lighting.ambient);
    m_blending_shader.use(loco->get_context());
    m_blending_shader.set_vec3(loco->get_context(), loco_t::lighting_t::ambient_name, loco->lighting.ambient);

    resize_nr = loco->get_window()->add_resize_callback([this, loco](const auto& d) {
      m_shader.use(loco->get_context());
      m_shader.set_vec2(loco->get_context(), "window_size", loco->get_window()->get_size());
      m_blending_shader.use(loco->get_context());
      m_blending_shader.set_vec2(loco->get_context(), "window_size", loco->get_window()->get_size());
    });
  }
  void sb_close() {
    loco_t* loco = get_loco();

    loco->get_window()->remove_resize_callback(resize_nr);

    blocks.Close();
    bm_list.Close();

    //assert(0);
    //loco_bdbt_close(&loco->bdbt);

    m_shader.close(loco->get_context());
    m_blending_shader.close(loco->get_context());

    //for (uint32_t i = 0; i < blocks.size(); i++) {
    //  blocks[i].uniform_buffer.close(loco->get_context());
    //}
  }

  struct block_t;

  shape_bm_NodeReference_t push_new_bm(properties_t& p) {
    auto lnr = bm_list.NewNode();
    auto ln = &bm_list[lnr];
    ln->first_block = blocks.NewNodeLast();
    blocks[ln->first_block].block.open(get_loco(), this);
    ln->last_block = ln->first_block;
    ln->instance_properties = *(bm_properties_t*)&p;
    return lnr;
  }

  // STRUCT MANUAL PADDING IS REQUIRED (32 BIT)
  block_t* sb_push_back(fan::opengl::cid_t* cid, properties_t& p) {

    loco_t* loco = get_loco();

    shape_bm_NodeReference_t bm_id;

    // todo compress 
    do{
      loco_bdbt_NodeReference_t nr = loco->root;

      loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k3;
      typename decltype(k3)::KeySize_t ki3;

      loco_bdbt_Key_t<sizeof(shape_type_t::_t) * 8> k2;
      typename decltype(k2)::KeySize_t ki2;

      uint16_t depth = p.position.z;
      loco_bdbt_Key_t<sizeof(uint16_t) * 8, true> k1;
      typename decltype(k1)::KeySize_t ki1;

      redraw_key_t redraw_key;
      redraw_key.blending = p.blending;
      loco_bdbt_Key_t<sizeof(redraw_key_t) * 8> k0;
      typename decltype(k0)::KeySize_t ki0;

      k0.q(&loco->bdbt, &redraw_key, &ki0, &nr);
      if (ki0 != sizeof(redraw_key_t) * 8) {
        auto o0 = loco_bdbt_NewNode(&loco->bdbt);
        k0.a(&loco->bdbt, &redraw_key, ki0, nr, o0);

        auto o1 = loco_bdbt_NewNode(&loco->bdbt);
        k1.a(&loco->bdbt, &depth, 0, o0, o1);

        auto o2 = loco_bdbt_NewNode(&loco->bdbt);
        k2.a(&loco->bdbt, &shape_type, 0, o1, o2);

        bm_id = push_new_bm(p);
        k3.a(&loco->bdbt, &p.key, 0, o2, bm_id.NRI);

        break;
      }

      k1.q(&loco->bdbt, &depth, &ki1, &nr);
      if (ki1 != sizeof(uint16_t) * 8) {
        auto o1 = loco_bdbt_NewNode(&loco->bdbt);
        k1.a(&loco->bdbt, &depth, ki1, nr, o1);

        auto o2 = loco_bdbt_NewNode(&loco->bdbt);
        k2.a(&loco->bdbt, &shape_type, 0, o1, o2);

        bm_id = push_new_bm(p);
        k3.a(&loco->bdbt, &p.key, 0, o2, bm_id.NRI);

        break;
      }

      k2.q(&loco->bdbt, &shape_type, &ki2, &nr);
      if (ki2 != sizeof(shape_type_t::_t)) {
        auto o2 = loco_bdbt_NewNode(&loco->bdbt);
        k2.a(&loco->bdbt, &shape_type, ki2, nr, o2);

        bm_id = push_new_bm(p);
        k3.a(&loco->bdbt, &p.key, 0, o2, bm_id.NRI);

        break;
      }

      {
        k3.q(&loco->bdbt, &p.key, &ki3, &nr);

        if (ki3 != sizeof(bm_properties_t::key_t) * 8) {
          bm_id = push_new_bm(p);
          k3.a(&loco->bdbt, &p.key, ki3, nr, bm_id.NRI);
        }
        else {
          bm_id = *(shape_bm_NodeReference_t*)&nr;
        }
      }

    }while (0);

    vi_t it = p;
    shape_bm_Node_t* bmn = bm_list.GetNodeByReference(bm_id);
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

    cid->bm_id = bm_id.NRI;
    cid->block_id = bmn->data.last_block.NRI;
    cid->instance_id = instance_id;
    cid->shape_type = -1;

    loco->types.iterate([&]<typename T>(auto shape_index, T shape) {
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
      if constexpr (std::is_same_v<shape_t, std::remove_reference_t<decltype(*this)>>) {
        cid->shape_type = shape_t::shape_type;
      }
    });

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
        if (last_block_id == bm_node->data.first_block) {
          loco_bdbt_NodeReference_t key_root = loco->root;

          loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k3;
          typename decltype(k3)::KeySize_t ki3;

          loco_bdbt_Key_t<sizeof(shape_type_t::_t) * 8> k2;
          typename decltype(k2)::KeySize_t ki2;

          uint16_t depth = p.position.z;
          loco_bdbt_Key_t<sizeof(uint16_t) * 8, true> k1;
          typename decltype(k1)::KeySize_t ki1;

          redraw_key_t redraw_key;
          redraw_key.blending = p.blending;
          loco_bdbt_Key_t<sizeof(redraw_key_t) * 8> k0;
          typename decltype(k0)::KeySize_t ki0;

          auto key_root0 = key_root;
          k0.q(&loco->bdbt, &redraw_key, &ki0, &key_root);
          if (ki0 != sizeof(redraw_key_t) * 8) {
            #if fan_debug >= 2
              __abort();
            #endif
          }
          k0.r(&loco->bdbt, &redraw_key, key_root);

          auto key_root1 = key_root;
          k1.q(&loco->bdbt, &depth, &ki1, &key_root);
          if (ki1 != sizeof(uint16_t) * 8) {
            #if fan_debug >= 2
              __abort();
            #endif
          }
          k1.r(&loco->bdbt, &depth, key_root);

          auto key_root2 = key_root;
          k2.q(&loco->bdbt, &shape_type, &ki2, &key_root);
          if (ki2 != sizeof(shape_type_t::_t)) {
            #if fan_debug >= 2
              __abort();
            #endif
          }
          k2.r(&loco->bdbt, &shape_type, key_root);

          auto key_root3 = key_root;
          k3.q(&loco->bdbt, &p.key, &ki3, &nr);
          if (ki3 != sizeof(bm_properties_t::key_t) * 8) {
            #if fan_debug >= 2
              __abort();
            #endif
          }
          k3.r(&loco->bdbt, &p.key, key_root);

          bm_list.Recycle(bm_id);
        }
        else {
          //fan::print("here");
          last_block_id = lpnr;
        }
        block->close(loco);
        blocks.Unlink(block_id);
        blocks.Recycle(block_id);
      }
      cid->bm_id = 0;
      cid->block_id = 0;
      cid->instance_id = 0;
      cid->instance_id = -1;
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
    cid->bm_id = 0;
    cid->block_id = 0;
    cid->instance_id = 0;
  }

  block_t* sb_get_block(fan::opengl::cid_t* fcid) {
    cid_t* cid = (cid_t*)fcid;
    auto& block_node = blocks[*(bll_block_NodeReference_t*)&cid->block_id];
    return &block_node.block;
  }

  //ri_t& sb_get_ri() {
  //  return 
  //}

  template <typename T, typename T2>
  T get(fan::opengl::cid_t* cid, T T2::* member) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    return block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id)->*member;
  }
  template <typename T, typename T2>
  void set(fan::opengl::cid_t* cid, T T2::* member, const auto& value) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);

    block->uniform_buffer.edit_instance(loco->get_context(), &loco->m_write_queue, cid->instance_id, member, value);
    block->uniform_buffer.common.edit(
      loco->get_context(),
      &loco->m_write_queue,
      cid->instance_id * sizeof(vi_t) + fan::ofof(member),
      cid->instance_id * sizeof(vi_t) + fan::ofof(member) + sizeof(T)
    );
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
    return loco->get_context()->viewport_list[*ri.key.get_value<
      bm_properties_t::key_t::get_index_with_type<fan::graphics::viewport_list_NodeReference_t>()
    >()].viewport_id;
  }

  void set_vertex(const fan::string& str) {
    loco_t* loco = get_loco();
    m_current_shader->set_vertex(loco->get_context(), str);
  }
  void set_fragment(const fan::string& str) {
    loco_t* loco = get_loco();
    m_current_shader->set_fragment(loco->get_context(), str);
  }
  void compile() {
    loco_t* loco = get_loco();
    m_current_shader->compile(loco->get_context());
  }

  static inline std::vector<fan::function_t<void()>> draw_queue_helper;

  template <uint32_t depth = 0>
  void traverse_draw(loco_bdbt_NodeReference_t nr, uint32_t draw_mode) {
    loco_t* loco = get_loco();
    if constexpr (depth == bm_properties_t::key_t::count + 1) {
      auto bmn = bm_list.GetNodeByReference(*(shape_bm_NodeReference_t*)&nr);
      auto bnr = bmn->data.first_block;

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
    }
    else {
      typename loco_bdbt_Key_t<sizeof(typename bm_properties_t::key_t::get_type<depth>::type) * 8>::Traverse_t kt;
      kt.i(nr);
      typename bm_properties_t::key_t::get_type<depth>::type o;
      while (kt.t(&loco->bdbt, &o)) {
        loco->process_block_properties_element(this, o);
        traverse_draw<depth + 1>(kt.Output, draw_mode);
      }
    }
  }

  void sb_draw(loco_bdbt_NodeReference_t key_root, uint32_t draw_mode = fan::opengl::GL_TRIANGLES) {
    loco_t* loco = get_loco();
    m_current_shader->use(loco->get_context());
    m_current_shader->set_vec3(loco->get_context(), loco_t::lighting_t::ambient_name, loco->lighting.ambient);
    m_current_shader->set_int(loco->get_context(), "_t00", 0);
    m_current_shader->set_int(loco->get_context(), "_t01", 1);
    traverse_draw(key_root, draw_mode);
  }

  properties_t sb_get_properties(fan::opengl::cid_t* cid) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    properties_t p;
    auto& something = bm_list[*(shape_bm_NodeReference_t*)&cid->bm_id];
    *(bm_properties_t*)&p = something.instance_properties;
    *(vi_t*)&p = *block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id);
    *(ri_t*)&p = block->p[cid->instance_id];
    return p;
  }

  template <uint32_t i>
  void sb_set_key(fan::opengl::cid_t* cid, auto value) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    properties_t p;
    *(vi_t*)&p = *block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id);
    *(ri_t*)&p = block->p[cid->instance_id];
    *p.key.get_value<i>() = value;

    sb_erase(cid);

    sb_push_back(cid, p);
  }

  void sb_set_depth(fan::opengl::cid_t* cid, f32_t depth) {
    loco_t* loco = get_loco();
    auto block = sb_get_block(cid);
    properties_t p;
    *(vi_t*)&p = *block->uniform_buffer.get_instance(loco->get_context(), cid->instance_id);
    *(ri_t*)&p = block->p[cid->instance_id];
    p.sb_depth_var.z = depth;
    sb_erase(cid);
    sb_push_back(cid, p);
  }

  fan::opengl::shader_t m_shader;
  fan::opengl::shader_t m_blending_shader;
  fan::opengl::shader_t* m_current_shader = nullptr;

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
  #undef sb_get_loco
  #undef sb_shader_fragment_string
  #undef sb_depth_var