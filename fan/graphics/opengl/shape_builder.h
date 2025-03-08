//          for 3d remove depth
// <redraw key> <depth> <shape type> <context key> == block manager
#pragma once

struct block_t {
  void open(loco_t* loco, auto* shape) {
    uniform_buffer.open(gloco->get_context());
    uniform_buffer.init_uniform_block(gloco->get_context(), shape->m_shader.get_shader().id, "instance_t");
    #ifndef sb_no_blending
    uniform_buffer.init_uniform_block(gloco->get_context(), shape->m_blending_shader.get_shader().id, "instance_t");
    #endif
  }
  void close(loco_t* loco) {
    uniform_buffer.close(gloco->get_context(), &gloco->m_write_queue);
  }

  fan::opengl::core::uniform_block_t<vi_t, max_instance_size> uniform_buffer;
  loco_t::cid_nt_t id[max_instance_size];
  ri_t ri[max_instance_size];
};

protected:


  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix bll_block
  #include <fan/fan_bll_preset.h>
  #define BLL_set_Link 1
  #define bcontainer_set_StoreFormat 1
  //#define BLL_set_StoreFormat1_alloc_open malloc
  //#define BLL_set_StoreFormat1_alloc_close free
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType block_t
  #include <BLL/BLL.h>

  bll_block_t blocks;

  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix shape_bm
  #include <fan/fan_bll_preset.h>
  #define BLL_set_IsNodeRecycled 0
  #define BLL_set_Link 0
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeData \
    bll_block_NodeReference_t first_block; \
    bll_block_NodeReference_t last_block; \
    key_t key;
  #include <BLL/BLL.h>

  shape_bm_t bm_list;

  fan::window_t::resize_callback_NodeReference_t resize_nr;

  #pragma pack(push, 1)
  struct key_pack_t {
    loco_t::redraw_key_t redraw_key;
    uint16_t depth;
    loco_t::shape_type_t shape_type;
    context_key_t::key_t shape_key;
  };
  #pragma pack(pop)

public:

  void suck_block_element(loco_t::cid_nt_t& id, auto&&... suckers) {

    auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
    auto bm_node = bm_list.GetNodeByReference(bm_id);

    auto block_id = *(bll_block_NodeReference_t*)&id->block_id;
    auto block_node = blocks.GetNodeByReference(block_id);
    auto block = &block_node->data;

    auto instance_id = id->instance_id;

    auto& last_block_id = bm_node->data.last_block;
    auto* last_block_node = blocks.GetNodeByReference(last_block_id);
    block_t* last_block = &last_block_node->data;
    uint32_t last_instance_id = last_block->uniform_buffer.size() - 1;

    if constexpr (sizeof...(suckers)) {

      block_element_t* block_element = fan::get_variadic_element<0>(suckers...);

      vi_t* current_vi = &((vi_t*)block->uniform_buffer.buffer)[instance_id];
      ri_t* current_ri = &block->ri[instance_id];

      block_element->key = bm_node->data.key;

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

    if (block_id == last_block_id && instance_id == block->uniform_buffer.size() - 1) {
      block->uniform_buffer.m_size -= sizeof(vi_t);
      if (block->uniform_buffer.size() == 0) {
        auto lpnr = block_node->PrevNodeReference;
        if (last_block_id == bm_node->data.first_block) {
          sb_erase_key_from(id);

          bm_list.Recycle(bm_id);
        }
        else {
          last_block_id = lpnr;
        }
        block->close(gloco);
        blocks.Unlink(block_id);
        blocks.Recycle(block_id);
      }
    }
    else {
      vi_t* last_instance_data = last_block->uniform_buffer.get_instance(gloco->get_context(), last_instance_id);

      block->uniform_buffer.copy_instance(
        gloco->get_context(),
        &gloco->m_write_queue,
        instance_id,
        last_instance_data
      );

      last_block->uniform_buffer.m_size -= sizeof(vi_t);

      block->ri[instance_id] = std::move(last_block->ri[last_instance_id]);

      block->id[instance_id] = last_block->id[last_instance_id];
      block->id[instance_id]->block_id = block_id.NRI;
      block->id[instance_id]->instance_id = instance_id;

      if (last_block->uniform_buffer.size() == 0) {
        auto lpnr = last_block_node->PrevNodeReference;

        last_block->close(gloco);
        blocks.Unlink(last_block_id);
        blocks.Recycle(last_block_id);

        bm_node->data.last_block = lpnr;
      }
    }
  }

  void get_block_id_from_push(
    shape_bm_NodeReference_t& bm_id,
    bll_block_NodeReference_t& block_id,
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
    block_id = bm.last_block;
    auto block = &blocks[block_id];

    if (block->uniform_buffer.size() == max_instance_size) {
      block_id = blocks.NewNode();
      blocks.linkNext(bm.last_block, block_id);
      bm.last_block = block_id;
      block = &blocks[block_id];
      block->open(gloco, this);
    }
  }

  void unsuck_block_element(
    loco_t::cid_nt_t& id,
    block_element_t& block_element
  ) {
    shape_bm_NodeReference_t bm_id;
    bll_block_NodeReference_t block_id;

    push_key_t key;
    key.iterate([&](auto i, const auto& data) {
      *data = { .data = *block_element.key.template get_value<i.value>() };
      });

    get_block_id_from_push(bm_id, block_id, key);

    auto block = &blocks[block_id];
    block->uniform_buffer.push_ram_instance(gloco->get_context(), *(vi_t*)block_element.vi);

    const uint32_t instance_id = block->uniform_buffer.size() - 1;

    block->id[instance_id] = id;

    block->uniform_buffer.common.edit(
      gloco->get_context(),
      &gloco->m_write_queue,
      instance_id * sizeof(vi_t),
      instance_id * sizeof(vi_t) + sizeof(vi_t)
    );

    id->bm_id = bm_id.NRI;
    id->block_id = block_id.NRI;
    id->instance_id = instance_id;
    id->shape_type = *(std::underlying_type<decltype(shape_type)>::type*) & shape_type;

    block->ri[instance_id] = std::move(block_element.ri);
  }

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

  // creates shader with given vertex_path and fragment_path
  void sb_open(const fan::string vertex_path, const fan::string& fragment_path) {

    #if sb_has_own_key_root == 1
    key_root = loco_bdbt_NewNode(&gloco->bdbt);
    #endif

    fan::string vertex_code;
    #if defined(sb_shader_vertex_string)
    vertex_code = vertex_path;
    #else
    vertex_code = loco_t::read_shader(vertex_path);
    #endif
    m_shader.open();
    m_shader.set_vertex(
      vertex_code
    );
    fan::string fragment_code;
    #if defined(sb_shader_fragment_string)
    fragment_code = sb_shader_fragment_string;
    #else
    fragment_code = loco_t::read_shader(fragment_path);
    #endif
    m_shader.set_fragment(
      fragment_code
    );
    m_shader.compile();

    #ifndef sb_no_blending
    m_blending_shader.open();
    m_blending_shader.set_vertex(
      vertex_code
    );
    #endif

    #ifndef sb_no_blending
    auto found = fragment_code.find("discard;");
    if (found != fan::string::npos) {
      fragment_code.erase(found, std::string_view("discard;").size());
    }
    m_blending_shader.set_fragment(
      fragment_code
    );
    m_blending_shader.compile();

    m_current_shader = &m_blending_shader;
    #else
    m_current_shader = &m_shader;
    #endif

    m_shader.use();
    m_shader.set_vec3(loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
    #ifndef sb_no_blending
    m_blending_shader.use();
    m_blending_shader.set_vec3(loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
    #endif

    m_shader.use();
    m_shader.set_vec2("window_size", gloco->window.get_size());
    #ifndef sb_no_blending
    m_blending_shader.use();
    m_blending_shader.set_vec2("window_size", gloco->window.get_size());
    #endif

    resize_nr = gloco->window.add_resize_callback([this](const auto& d) {
      m_shader.use();
      m_shader.set_vec2("window_size", gloco->window.get_size());
      #ifndef sb_no_blending
      m_blending_shader.use();
      m_blending_shader.set_vec2("window_size", gloco->window.get_size());
      #endif
      });
  }
  void sb_close() {

    gloco->window.remove_resize_callback(resize_nr);

    //assert(0);
    //loco_bdbt_close(&gloco->bdbt);

    m_shader.close();
    #ifndef sb_no_blending
    m_blending_shader.close();
    #endif

    //for (uint32_t i = 0; i < blocks.size(); i++) {
    //  blocks[i].uniform_buffer.close(gloco->get_context());
    //}
  }

  struct block_t;

  shape_bm_NodeReference_t push_new_bm(auto& key) {
    auto lnr = bm_list.NewNode();
    auto ln = &bm_list[lnr];
    ln->first_block = blocks.NewNodeLast();
    blocks[ln->first_block].open(gloco, this);
    ln->last_block = ln->first_block;
    ln->key.iterate([&](auto i, const auto& data) {
      *data = key.template get_value<i.value>()->data;
      });
    return lnr;
  }

  // STRUCT MANUAL PADDING IS REQUIRED (32 BIT), even vi_t element needs to be vec4 (use padding)
  block_t* sb_push_back(loco_t::cid_nt_t& id, properties_t p) {

    #if fan_debug >= 2
    [&id] <typename T>(T & p, auto * This) {
      if constexpr (fan::has_camera_t<T>) {
        if (p.camera == nullptr) {
          fan::throw_error("invalid camera");
        }
      }
    }(p, this);

    [&id] <typename T>(T & p, auto * This) {
      if constexpr (fan::has_viewport_t<T>) {
        if (p.viewport == nullptr) {
          fan::throw_error("invalid viewport");
        }
      }
    }(p, this);

    [&id] <typename T>(T & p, auto * This) {
      if constexpr (fan::has_image_t<T>) {
        if (p.image == nullptr) {
          fan::throw_error("invalid image");
        }
      }
    }(p, this);

    #endif


    if constexpr (fan_has_variable(properties_t, camera)) {
      get_key_value(loco_t::camera_list_NodeReference_t) = p.camera;
    }
    if constexpr (fan_has_variable(properties_t, camera)) {
      get_key_value(fan::graphics::viewport_list_NodeReference_t) = p.viewport;
    }
    // implement for sprite

    push_key_t key{
    #if sb_ignore_3_key == 0
      loco_t::make_push_key_t<loco_t::redraw_key_t>{.data = {.blending = p.blending}},
      loco_t::make_push_key_t<uint16_t, true>{.data = (uint16_t)p.sb_depth_var.z},
      {.data = shape_type },
    #endif
      {.data = p.key}
    };

    shape_bm_NodeReference_t bm_id;
    bll_block_NodeReference_t block_id;
    get_block_id_from_push(bm_id, block_id, key);

    auto block = &blocks[block_id];
    block->uniform_buffer.push_ram_instance(gloco->get_context(), *dynamic_cast<const vi_t*>(&p));

    const uint32_t instance_id = block->uniform_buffer.size() - 1;

    block->id[instance_id] = id;

    block->uniform_buffer.common.edit(
      gloco->get_context(),
      &gloco->m_write_queue,
      instance_id * sizeof(vi_t),
      instance_id * sizeof(vi_t) + sizeof(vi_t)
    );


    id->bm_id = bm_id.NRI;
    id->block_id = block_id.NRI;
    id->instance_id = instance_id;
    id->shape_type = (std::underlying_type<decltype(shape_type)>::type)shape_type;

    block->ri[instance_id] = std::move(*(ri_t*)&p);
    return block;
  }

  block_t* sb_get_block(loco_t::cid_nt_t& id) {
    auto& block_node = blocks[*(bll_block_NodeReference_t*)&id->block_id];
    return &block_node;
  }

  template <typename T, typename T2, typename T3>
  void set(loco_t::cid_nt_t& id, T T2::* member, const T3& value) {
    if constexpr (std::is_same_v<T, loco_t::position3_t>) {
      if constexpr (std::is_same_v<T3, fan::vec3>)
        if (value.z != get(id, member).z) {
          sb_set_depth(id, value.z);
        }
    }

    auto block = sb_get_block(id);

    block->uniform_buffer.edit_instance(gloco->get_context(), &gloco->m_write_queue, id->instance_id, member, value);
    block->uniform_buffer.common.edit(
      gloco->get_context(),
      &gloco->m_write_queue,
      id->instance_id * sizeof(vi_t) + fan::ofof(member),
      id->instance_id * sizeof(vi_t) + fan::ofof(member) + sizeof(T)
    );
  }

  void set_vertex(const fan::string& str) {
    m_current_shader->set_vertex(str);
  }
  void set_fragment(const fan::string& str) {
    m_current_shader->set_fragment(str);
  }
  void compile() {
    m_current_shader->compile();
  }

  static inline std::vector<fan::function_t<void()>> draw_queue_helper;

  template <uint32_t depth = 0>
  void traverse_draw(loco_bdbt_NodeReference_t nr, uint32_t draw_mode) {
    if constexpr (depth == context_key_t::key_t::count + 1) {

      #if fan_debug >= 2
      if (nr >= bm_list.NodeList.Current) {
        __abort();
      }
      #endif
      auto& bm = bm_list[*(shape_bm_NodeReference_t*)&nr];
      auto block_id = bm.first_block;

      m_current_shader->set_float("m_time", (double)gloco->m_time.elapsed() / 1e+9);

      while (1) {
        auto block_node = blocks.GetNodeByReference(block_id);
        #ifdef fan_unit_test
        for (uint32_t i = 0; i < block_node->data.uniform_buffer.size(); i++) {
          if (idlist.find(*(uint32_t*)&block_node->data.id[i]) == idlist.end()) {
            fan::throw_error(__LINE__);
          }
          idlist[(uint32_t)block_node->data.id[i].NRI].TraverseFound = true;
        }
        #endif
        block_node->data.uniform_buffer.bind_buffer_range(
          gloco->get_context(),
          block_node->data.uniform_buffer.size()
        );

        block_node->data.uniform_buffer.draw(
          gloco->get_context(),
          0 * sb_vertex_count,
          block_node->data.uniform_buffer.size() * sb_vertex_count,
          draw_mode
        );
        if (block_id == bm.last_block) {
          break;
        }
        block_id = block_node->NextNodeReference;
      }
    }
    else {
      typename loco_bdbt_Key_t<sizeof(typename context_key_t::key_t::get_type<depth>::type) * 8>::Traverse_t kt;
      kt.i(nr);
      typename context_key_t::key_t::get_type<depth>::type o;
      while (kt.t(&gloco->bdbt, &o)) {
        gloco->process_block_properties_element(this, o);
        traverse_draw<depth + 1>(kt.Output, draw_mode);
      }
    }
  }

  template <uint32_t depth = 0>
  void traverse_draw1(loco_bdbt_NodeReference_t nr, uint32_t draw_mode) {
    if constexpr (depth == context_key_t::key_t::count + 1) {

      #if fan_debug >= 2
      if (nr >= bm_list.NodeList.Current) {
        __abort();
      }
      #endif
      auto& bm = bm_list[*(shape_bm_NodeReference_t*)&nr];
      auto block_id = bm.first_block;

      m_shader.set_float("m_time", (double)gloco->m_time.elapsed() / 1e+9);

      while (1) {
        auto block_node = blocks.GetNodeByReference(block_id);
        #ifdef fan_unit_test
        for (uint32_t i = 0; i < block_node->data.uniform_buffer.size(); i++) {
          if (idlist.find(*(uint32_t*)&block_node->data.id[i]) == idlist.end()) {
            fan::throw_error(__LINE__);
          }
          idlist[(uint32_t)block_node->data.id[i].NRI].TraverseFound = true;
        }
        #endif

        fan_if_has_function(this, custom_draw, (block_node, draw_mode));

        if (block_id == bm.last_block) {
          break;
        }
        block_id = block_node->NextNodeReference;
      }
    }
    else {
      typename loco_bdbt_Key_t<sizeof(typename context_key_t::key_t::get_type<depth>::type) * 8>::Traverse_t kt;
      kt.i(nr);
      typename context_key_t::key_t::get_type<depth>::type o;
      while (kt.t(&gloco->bdbt, &o)) {
        gloco->process_block_properties_element(this, o);
        traverse_draw1<depth + 1>(kt.Output, draw_mode);
      }
    }
  }

  void sb_draw(loco_bdbt_NodeReference_t key_nr, uint32_t draw_mode = GL_TRIANGLES) {
    m_current_shader->use();
    // todo remove
    m_current_shader->set_vec3(loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
    m_current_shader->set_int("_t00", 0);
    m_current_shader->set_int("_t01", 1);
    traverse_draw(key_nr, draw_mode);
  }

  void sb_draw1(loco_bdbt_NodeReference_t key_nr, uint32_t draw_mode) {
    m_current_shader->use();
    // todo remove
    m_current_shader->set_vec3(loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
    m_current_shader->set_int("_t00", 0);
    m_current_shader->set_int("_t01", 1);
    traverse_draw1(key_nr, draw_mode);
  }

  loco_t::shader_t m_shader;
  #ifndef sb_no_blending
  loco_t::shader_t m_blending_shader;
  #endif
  loco_t::shader_t* m_current_shader = nullptr;

  vi_t& sb_get_vi(loco_t::cid_nt_t& cid) {
    return *sb_get_block(cid)->uniform_buffer.get_instance(gloco->get_context(), cid->instance_id);
  }
  template <typename T>
  void sb_set_vi(loco_t::cid_nt_t& cid, auto T::* member, auto value) {
    auto& instance = sb_get_vi(cid);
    instance.*member = value;
    sb_get_block(cid)->uniform_buffer.edit_instance(gloco->get_context(), &gloco->m_write_queue, cid->instance_id, member, value);
    //sb_get_block(cid)->uniform_buffer.copy_instance(&instance);
  }

  ri_t& sb_get_ri(loco_t::cid_nt_t& cid) {
    return sb_get_block(cid)->ri[cid->instance_id];
  }
  template <typename T>
  void sb_set_ri(loco_t::cid_nt_t& cid, auto T::* member, auto value) {
    sb_get_ri(cid).*member = value;
  }

  #undef sb_shader_vertex_path
  #undef sb_shader_fragment_path

  #undef sb_shader_vertex_string
  #undef sb_shader_fragment_string

  #undef sb_no_blending
  #undef sb_mark