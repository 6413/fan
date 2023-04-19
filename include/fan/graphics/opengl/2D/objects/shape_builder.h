#ifndef sb_depth_var
  #define sb_depth_var position
#endif

#ifndef sb_vertex_count
#define sb_vertex_count 6
#endif

struct block_t {
  void open(loco_t* loco, auto* shape) {
    uniform_buffer.open(gloco->get_context());
    uniform_buffer.init_uniform_block(gloco->get_context(), shape->m_shader.id, "instance_t");
  }
  void close(loco_t* loco) {
    uniform_buffer.close(gloco->get_context(), &gloco->m_write_queue);
  }

  fan::opengl::core::uniform_block_t<vi_t, max_instance_size> uniform_buffer;
  loco_t::cid_nt_t id[max_instance_size];
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
    blocks.Open();
    bm_list.Open();

    m_shader.open(gloco->get_context());
    m_shader.set_vertex(
      gloco->get_context(),
      #include sb_shader_vertex_path
    );
    m_shader.set_fragment(
      gloco->get_context(),
      #ifndef sb_shader_fragment_string
        #include sb_shader_fragment_path
      #else
        sb_shader_fragment_string
      #endif
    );
    m_shader.compile(gloco->get_context());

    m_blending_shader.open(gloco->get_context());
    m_blending_shader.set_vertex(
      gloco->get_context(),
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
      gloco->get_context(),
      str
    );
    m_blending_shader.compile(gloco->get_context());

    m_current_shader = &m_shader;

    m_shader.use(gloco->get_context());
    m_shader.set_vec2(gloco->get_context(), "window_size", gloco->get_window()->get_size());
    m_blending_shader.use(gloco->get_context());
    m_blending_shader.set_vec2(gloco->get_context(), "window_size", gloco->get_window()->get_size());

    m_shader.use(gloco->get_context());
    m_shader.set_vec3(gloco->get_context(), loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
    m_blending_shader.use(gloco->get_context());
    m_blending_shader.set_vec3(gloco->get_context(), loco_t::lighting_t::ambient_name, gloco->lighting.ambient);

    resize_nr = gloco->get_window()->add_resize_callback([this](const auto& d) {
      m_shader.use(gloco->get_context());
      m_shader.set_vec2(gloco->get_context(), "window_size", gloco->get_window()->get_size());
      m_blending_shader.use(gloco->get_context());
      m_blending_shader.set_vec2(gloco->get_context(), "window_size", gloco->get_window()->get_size());
    });
  }
  void sb_close() {

    gloco->get_window()->remove_resize_callback(resize_nr);

    blocks.Close();
    bm_list.Close();

    //assert(0);
    //loco_bdbt_close(&gloco->bdbt);

    m_shader.close(gloco->get_context());
    m_blending_shader.close(gloco->get_context());

    //for (uint32_t i = 0; i < blocks.size(); i++) {
    //  blocks[i].uniform_buffer.close(gloco->get_context());
    //}
  }

  struct block_t;

  shape_bm_NodeReference_t push_new_bm(properties_t& p) {
    auto lnr = bm_list.NewNode();
    auto ln = &bm_list[lnr];
    ln->first_block = blocks.NewNodeLast();
    blocks[ln->first_block].block.open(gloco, this);
    ln->last_block = ln->first_block;
    ln->instance_properties = *(bm_properties_t*)&p;
    return lnr;
  }

  // STRUCT MANUAL PADDING IS REQUIRED (32 BIT)
  block_t* sb_push_back(loco_t::cid_nt_t& id, properties_t& p) {

    shape_bm_NodeReference_t bm_id;

    // todo compress 
    do{
      loco_bdbt_NodeReference_t nr = gloco->root;

      loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k3;
      typename decltype(k3)::KeySize_t ki3;

      loco_t::shape_type_t::_t st = shape_type;
      loco_bdbt_Key_t<sizeof(loco_t::shape_type_t::_t) * 8> k2;
      typename decltype(k2)::KeySize_t ki2;

      uint16_t depth = p.sb_depth_var.z;
      loco_bdbt_Key_t<sizeof(uint16_t) * 8, true> k1;
      typename decltype(k1)::KeySize_t ki1;

      loco_t::redraw_key_t redraw_key;
      redraw_key.blending = p.blending;
      loco_bdbt_Key_t<sizeof(loco_t::redraw_key_t) * 8> k0;
      typename decltype(k0)::KeySize_t ki0;

      k0.q(&gloco->bdbt, &redraw_key, &ki0, &nr);
      if (ki0 != sizeof(loco_t::redraw_key_t) * 8) {
        auto o0 = loco_bdbt_NewNode(&gloco->bdbt);
        k0.a(&gloco->bdbt, &redraw_key, ki0, nr, o0);

        auto o1 = loco_bdbt_NewNode(&gloco->bdbt);
        k1.a(&gloco->bdbt, &depth, 0, o0, o1);

        auto o2 = loco_bdbt_NewNode(&gloco->bdbt);
        k2.a(&gloco->bdbt, &st, 0, o1, o2);

        bm_id = push_new_bm(p);
        k3.a(&gloco->bdbt, &p.key, 0, o2, bm_id.NRI);

        break;
      }

      k1.q(&gloco->bdbt, &depth, &ki1, &nr);
      if (ki1 != sizeof(uint16_t) * 8) {
        auto o1 = loco_bdbt_NewNode(&gloco->bdbt);
        k1.a(&gloco->bdbt, &depth, ki1, nr, o1);

        auto o2 = loco_bdbt_NewNode(&gloco->bdbt);
        k2.a(&gloco->bdbt, &st, 0, o1, o2);

        bm_id = push_new_bm(p);
        k3.a(&gloco->bdbt, &p.key, 0, o2, bm_id.NRI);

        break;
      }

      k2.q(&gloco->bdbt, &st, &ki2, &nr);
      if (ki2 != sizeof(loco_t::shape_type_t::_t) * 8) {
        auto o2 = loco_bdbt_NewNode(&gloco->bdbt);
        k2.a(&gloco->bdbt, &st, ki2, nr, o2);

        bm_id = push_new_bm(p);
        k3.a(&gloco->bdbt, &p.key, 0, o2, bm_id.NRI);

        break;
      }

      {
        k3.q(&gloco->bdbt, &p.key, &ki3, &nr);

        if (ki3 != sizeof(bm_properties_t::key_t) * 8) {
          bm_id = push_new_bm(p);
          k3.a(&gloco->bdbt, &p.key, ki3, nr, bm_id.NRI);
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
      last_block->open(gloco, this);
    }
    block_t* block = last_block;
    block->uniform_buffer.push_ram_instance(gloco->get_context(), it);

    const uint32_t instance_id = block->uniform_buffer.size() - 1;

    block->id[instance_id] = id;

    block->uniform_buffer.common.edit(
      gloco->get_context(),
      &gloco->m_write_queue,
      instance_id * sizeof(vi_t),
      instance_id * sizeof(vi_t) + sizeof(vi_t)
    );


    id->bm_id = bm_id.NRI;
    id->block_id = bmn->data.last_block.NRI;
    id->instance_id = instance_id;
    id->shape_type = -1;

    gloco->types.iterate([&]<typename T>(auto shape_index, T shape) {
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
      if constexpr (std::is_same_v<shape_t, std::remove_reference_t<decltype(*this)>>) {
        id->shape_type = shape_t::shape_type;
      }
    });

    block->p[instance_id] = *(ri_t*)&p;
    return block;
  }
  void sb_erase(loco_t::cid_nt_t& id) {
    auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
    auto bm_node = bm_list.GetNodeByReference(bm_id);

    auto block_id = *(bll_block_NodeReference_t*)&id->block_id;
    auto block_node = blocks.GetNodeByReference(*(bll_block_NodeReference_t*)&id->block_id);
    auto block = &block_node->data.block;

    auto& last_block_id = bm_node->data.last_block;
    auto* last_block_node = blocks.GetNodeByReference(last_block_id);
    block_t* last_block = &last_block_node->data.block;
    uint32_t last_instance_id = last_block->uniform_buffer.size() - 1;

    if (block_id == last_block_id && id->instance_id == block->uniform_buffer.size() - 1) {
      block->uniform_buffer.m_size -= sizeof(vi_t);
      if (block->uniform_buffer.size() == 0) {
        auto lpnr = block_node->PrevNodeReference;
        if (last_block_id == bm_node->data.first_block) {
          loco_bdbt_NodeReference_t key_root = gloco->root;

          loco_bdbt_Key_t<sizeof(bm_properties_t::key_t) * 8> k3;
          typename decltype(k3)::KeySize_t ki3;

          loco_bdbt_Key_t<sizeof(loco_t::shape_type_t::_t) * 8> k2;
          typename decltype(k2)::KeySize_t ki2;

          uint16_t depth = sb_get_vi(id).sb_depth_var.z;
          loco_bdbt_Key_t<sizeof(uint16_t) * 8, true> k1;
          typename decltype(k1)::KeySize_t ki1;

          loco_t::redraw_key_t redraw_key;
          redraw_key.blending = sb_get_ri(id).blending;
          loco_bdbt_Key_t<sizeof(loco_t::redraw_key_t) * 8> k0;
          typename decltype(k0)::KeySize_t ki0;

          auto key_root0 = key_root;
          k0.q(&gloco->bdbt, &redraw_key, &ki0, &key_root);
          #if fan_debug >= 2
            if (ki0 != sizeof(loco_t::redraw_key_t) * 8) {
              __abort();
            }
          #endif

          auto key_root1 = key_root;
          k1.q(&gloco->bdbt, &depth, &ki1, &key_root);
          #if fan_debug >= 2
            if (ki1 != sizeof(uint16_t) * 8) {
              __abort();
            }
          #endif

          auto key_root2 = key_root;
          k2.q(&gloco->bdbt, &shape_type, &ki2, &key_root);
          #if fan_debug >= 2
            if (ki2 != sizeof(loco_t::shape_type_t::_t) * 8) {
              __abort();
            }
          #endif

          auto key_root3 = key_root;
          k3.q(&gloco->bdbt, &bm_node->data.instance_properties.key, &ki3, &key_root);
          #if fan_debug >= 2
            if (ki3 != sizeof(bm_properties_t::key_t) * 8) {
              __abort();
            }
          #endif

          k3.r(&gloco->bdbt, &bm_node->data.instance_properties.key, key_root3);
          if(loco_bdbt_inrhc(&gloco->bdbt, key_root3) == false){
            loco_bdbt_Recycle(&gloco->bdbt, key_root3);
            k2.r(&gloco->bdbt, (void *)&shape_type, key_root2);
            if(loco_bdbt_inrhc(&gloco->bdbt, key_root2) == false){
              loco_bdbt_Recycle(&gloco->bdbt, key_root2);
              k1.r(&gloco->bdbt, &depth, key_root1);
              if(loco_bdbt_inrhc(&gloco->bdbt, key_root1) == false){
                loco_bdbt_Recycle(&gloco->bdbt, key_root1);
                k0.r(&gloco->bdbt, &redraw_key, key_root0);
              }
            }
          }

          bm_list.Recycle(bm_id);
        }
        else {
          //fan::print("here");
          last_block_id = lpnr;
        }
        block->close(gloco);
        blocks.Unlink(block_id);
        blocks.Recycle(block_id);
      }
      id->bm_id = 0;
      id->block_id = 0;
      id->instance_id = 0;
      id->instance_id = -1;
      return;
    }

    vi_t* last_instance_data = last_block->uniform_buffer.get_instance(gloco->get_context(), last_instance_id);

    block->uniform_buffer.copy_instance(
      gloco->get_context(),
      &gloco->m_write_queue,
      id->instance_id,
      last_instance_data
    );

    last_block->uniform_buffer.m_size -= sizeof(vi_t);

    block->p[id->instance_id] = last_block->p[last_instance_id];

    block->id[id->instance_id] = last_block->id[last_instance_id];
    block->id[id->instance_id]->block_id = block_id.NRI;
    block->id[id->instance_id]->instance_id = id->instance_id;

    if (last_block->uniform_buffer.size() == 0) {
      auto lpnr = last_block_node->PrevNodeReference;

      last_block->close(gloco);
      blocks.Unlink(last_block_id);
      blocks.Recycle(last_block_id);

      bm_node->data.last_block = lpnr;
    }
    id->bm_id = 0;
    id->block_id = 0;
    id->instance_id = 0;
  }

  block_t* sb_get_block(loco_t::cid_nt_t& id) {
    auto& block_node = blocks[*(bll_block_NodeReference_t*)&id->block_id];
    return &block_node.block;
  }

  //ri_t& sb_get_ri() {
  //  return 
  //}

  template <typename T, typename T2>
  T get(loco_t::cid_nt_t& id, T T2::* member) {
    auto block = sb_get_block(id);
    return block->uniform_buffer.get_instance(gloco->get_context(), id->instance_id)->*member;
  }
  template <typename T, typename T2>
  void set(loco_t::cid_nt_t& id, T T2::* member, const auto& value) {
    auto block = sb_get_block(id);

    block->uniform_buffer.edit_instance(gloco->get_context(), &gloco->m_write_queue, id->instance_id, member, value);
    block->uniform_buffer.common.edit(
      gloco->get_context(),
      &gloco->m_write_queue,
      id->instance_id * sizeof(vi_t) + fan::ofof(member),
      id->instance_id * sizeof(vi_t) + fan::ofof(member) + sizeof(T)
    );
  }

  template <typename T = void>
  loco_t::camera_t* get_camera(loco_t::cid_nt_t& id) requires fan::has_camera_t<properties_t> {
    auto ri = sb_get_ri(id);
    return gloco->camera_list[*ri.key.get_value<
      bm_properties_t::key_t::get_index_with_type<loco_t::camera_list_NodeReference_t>()
    >()].camera_id;
  }

  template <typename T = void>
  fan::graphics::viewport_t* get_viewport(loco_t::cid_nt_t& id) requires fan::has_viewport_t<properties_t> {
    auto ri = sb_get_ri(id);
    return gloco->get_context()->viewport_list[*ri.key.get_value<
      bm_properties_t::key_t::get_index_with_type<fan::graphics::viewport_list_NodeReference_t>()
    >()].viewport_id;
  }

  void set_vertex(const fan::string& str) {
    m_current_shader->set_vertex(gloco->get_context(), str);
  }
  void set_fragment(const fan::string& str) {
    m_current_shader->set_fragment(gloco->get_context(), str);
  }
  void compile() {
    m_current_shader->compile(gloco->get_context());
  }

  static inline std::vector<fan::function_t<void()>> draw_queue_helper;

  template <uint32_t depth = 0>
  void traverse_draw(loco_bdbt_NodeReference_t nr, uint32_t draw_mode) {
    if constexpr (depth == bm_properties_t::key_t::count + 1) {

      #if fan_debug >= 2
        if(nr >= bm_list.NodeList.Current){
          __abort();
        }
      #endif
      auto bmn = bm_list.GetNodeByReference(*(shape_bm_NodeReference_t*)&nr);
      auto bnr = bmn->data.first_block;

      while (1) {
        auto node = blocks.GetNodeByReference(bnr);
        #ifdef fan_unit_test
          for (uint32_t i = 0; i < node->data.block.uniform_buffer.size(); i++) {
            if (idlist.find(*(uint32_t*)&node->data.block.id[i]) == idlist.end()) {
              fan::throw_error(__LINE__);
            }
            idlist[(uint32_t)node->data.block.id[i]].TraverseFound = true;
          }
        #endif
        node->data.block.uniform_buffer.bind_buffer_range(
          gloco->get_context(),
          node->data.block.uniform_buffer.size()
        );

        node->data.block.uniform_buffer.draw(
          gloco->get_context(),
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
      while (kt.t(&gloco->bdbt, &o)) {
        gloco->process_block_properties_element(this, o);
        traverse_draw<depth + 1>(kt.Output, draw_mode);
      }
    }
  }

  void sb_draw(loco_bdbt_NodeReference_t key_root, uint32_t draw_mode = fan::opengl::GL_TRIANGLES) {
    m_current_shader->use(gloco->get_context());
    m_current_shader->set_vec3(gloco->get_context(), loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
    m_current_shader->set_int(gloco->get_context(), "_t00", 0);
    m_current_shader->set_int(gloco->get_context(), "_t01", 1);
    traverse_draw(key_root, draw_mode);
  }

  properties_t sb_get_properties(loco_t::cid_nt_t& id) {
    auto block = sb_get_block(id);
    properties_t p;
    auto& something = bm_list[*(shape_bm_NodeReference_t*)&id->bm_id];
    *(bm_properties_t*)&p = something.instance_properties;
    *(vi_t*)&p = *block->uniform_buffer.get_instance(gloco->get_context(), id->instance_id);
    *(ri_t*)&p = block->p[id->instance_id];
    return p;
  }

  template <uint32_t i>
  void sb_set_key(loco_t::cid_nt_t& id, auto value) {
    auto block = sb_get_block(id);
    properties_t p;
    *(vi_t*)&p = *block->uniform_buffer.get_instance(gloco->get_context(), id->instance_id);
    *(ri_t*)&p = block->p[id->instance_id];
    *p.key.get_value<i>() = value;

    sb_erase(id);

    sb_push_back(id, p);
  }

  void sb_set_depth(loco_t::cid_nt_t& id, f32_t depth) {
    auto block = sb_get_block(id);
    properties_t p;
    *(vi_t*)&p = *block->uniform_buffer.get_instance(gloco->get_context(), id->instance_id);
    *(ri_t*)&p = block->p[id->instance_id];
    p.sb_depth_var.z = depth;
    sb_erase(id);
    sb_push_back(id, p);
  }

  fan::opengl::shader_t m_shader;
  fan::opengl::shader_t m_blending_shader;
  fan::opengl::shader_t* m_current_shader = nullptr;

  vi_t& sb_get_vi(loco_t::cid_nt_t& cid) {
    return *sb_get_block(cid)->uniform_buffer.get_instance(gloco->get_context(), cid->instance_id);
  }
  template <typename T>
  void sb_set_vi(loco_t::cid_nt_t& cid, auto T::* member, auto value) {
    auto& instance = sb_get_vi(cid);
    instance.*member = value;
    sb_get_block(cid)->uniform_buffer.edit_instance(gloco->get_context(), &gloco->m_write_queue, cid->instance_id, member, value);
    //sb_get_block(cid)->uniform_buffer.copy_instance(gloco->get_context(), &instance);
  }

  ri_t& sb_get_ri(loco_t::cid_nt_t& cid) {
    return sb_get_block(cid)->p[cid->instance_id];
  }
  template <typename T>
  void sb_set_ri(loco_t::cid_nt_t& cid, auto T::* member, auto value) {
    sb_get_ri(cid).*member = value;
  }

  #undef sb_shader_vertex_path
  #undef sb_shader_fragment_path
  #undef sb_vertex_count
  #undef sb_shader_fragment_string
  #undef sb_depth_var