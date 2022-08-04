struct rectangle_t {

  struct instance_t {
    fan::vec3 position = 0;
  private:
    f32_t pad[1];
  public:
    fan::vec2 size = 0;
    fan::vec2 rotation_point = 0;
    fan::color color = fan::colors::white;
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    f32_t angle = 0;
  };

  static constexpr uint32_t max_instance_size = 128;

  struct block_properties_t {
    fan::opengl::matrices_list_NodeReference_t matrices;
    fan::opengl::viewport_list_NodeReference_t viewport;
  };
      
  struct properties_t : instance_t {
    union {
      struct {
        // sb block properties contents come here
        fan::opengl::matrices_list_NodeReference_t matrices;
        fan::opengl::viewport_list_NodeReference_t viewport;
      };
      block_properties_t block_properties;
    };
  };

  void push_back(loco_t* loco, fan::opengl::cid_t* cid, properties_t& p) {
    instance_t it = p;

    loco_bdbt_rectangle_matrices_KeySize_t KeyIndex;
    loco_bdbt_NodeReference_t nr = root;
    loco_bdbt_rectangle_matrices_KeyQuery(&loco->bdbt, &p.matrices, &KeyIndex, &nr);
    if (KeyIndex != 8) {
      auto bdbt_nr = loco_bdbt_NewNode(&loco->bdbt);
      loco_bdbt_rectangle_matrices_KeyInFrom(&loco->bdbt, &p.matrices, KeyIndex, nr, bdbt_nr);
      nr = bdbt_nr;
    }

    loco_bdbt_rectangle_viewport_KeyQuery(&loco->bdbt, &p.viewport, &KeyIndex, &nr);
    if (KeyIndex != 8) {
      auto lnr = loco_rectangle_viewport_list_NewNode(&viewport_list);
      auto ln = loco_rectangle_viewport_list_GetNodeByReference(&viewport_list, lnr);
      ln->data.blocks.open();
      loco_bdbt_rectangle_viewport_KeyInFrom(&loco->bdbt, &p.viewport, KeyIndex, nr, lnr.NRI);
      nr = lnr.NRI;
    }
    auto ln = loco_rectangle_viewport_list_GetNodeByReference(&viewport_list, *(loco_rectangle_viewport_list_NodeReference_t*)&nr);
    
    sb_push_back(loco, ln->data.blocks, cid, p);
  }

  void draw(loco_t* loco) {
    m_shader.use(loco->get_context());

    loco_bdbt_rectangle_matrices_KeyTraverse_t kt0;
    loco_bdbt_rectangle_matrices_KeyTraverse_init(&kt0, root);
    while (loco_bdbt_rectangle_matrices_KeyTraverse(&loco->bdbt, &kt0)) {
      fan::opengl::matrices_list_NodeReference_t nmr = *(fan::opengl::matrices_list_NodeReference_t*)kt0.Key;
      auto node = fan::opengl::matrices_list_GetNodeByReference(&loco->get_context()->matrices_list, nmr);
      m_shader.set_matrices(loco->get_context(), node->data.matrices_id);
      loco_bdbt_rectangle_viewport_KeyTraverse_t kt1;
      loco_bdbt_rectangle_viewport_KeyTraverse_init(&kt1, kt0.Output);
      while (loco_bdbt_rectangle_viewport_KeyTraverse(&loco->bdbt, &kt1)) {
        fan::opengl::viewport_list_NodeReference_t nmr = *(fan::opengl::viewport_list_NodeReference_t*)kt1.Key;
        auto node = fan::opengl::viewport_list_GetNodeByReference(&loco->get_context()->viewport_list, nmr);
        node->data.viewport_id->set_viewport(
          loco->get_context(),
          node->data.viewport_id->get_viewport_position(),
          node->data.viewport_id->get_viewport_size()
        );
        auto kt1_node = loco_rectangle_viewport_list_GetNodeByReference(&viewport_list, *(loco_rectangle_viewport_list_NodeReference_t*)&kt1.Output);
        for (uint32_t i = 0; i < kt1_node->data.blocks.size(); i++) {
          kt1_node->data.blocks[i].uniform_buffer.bind_buffer_range(loco->get_context(), kt1_node->data.blocks[i].uniform_buffer.size());

          kt1_node->data.blocks[i].uniform_buffer.draw(
            loco->get_context(),
            0 * 6,
            kt1_node->data.blocks[i].uniform_buffer.size() * 6
          );
        }
      }
    }
  }

  struct block_t;

  #define BDBT_set_prefix loco_bdbt_rectangle_matrices
  #define BDBT_set_type_node uint16_t
  #define BDBT_set_KeySize 8
  #define BDBT_set_BitPerNode 2
  #define BDBT_set_declare_basic_types 0
  #define BDBT_set_declare_rest 0
  #define BDBT_set_declare_Key 1
  #define BDBT_set_base_prefix loco_bdbt
  #define BDBT_set_BaseLibrary 1
  #include _FAN_PATH(BDBT/BDBT.h)

  #define BDBT_set_prefix loco_bdbt_rectangle_viewport
  #define BDBT_set_type_node uint16_t
  #define BDBT_set_KeySize 8
  #define BDBT_set_BitPerNode 2
  #define BDBT_set_declare_basic_types 0
  #define BDBT_set_declare_rest 0
  #define BDBT_set_declare_Key 1
  #define BDBT_set_base_prefix loco_bdbt
  #define BDBT_set_BaseLibrary 1
  #include _FAN_PATH(BDBT/BDBT.h)

  #define BLL_set_prefix loco_rectangle_viewport_list
  #define BLL_set_BaseLibrary 1
  #define BLL_set_Link 0
  #define BLL_set_StoreFormat 1
  #define BLL_set_StoreFormat1_alloc_open malloc
  #define BLL_set_StoreFormat1_alloc_close free
  #define BLL_set_type_node uint16_t
  #define BLL_set_node_data \
  fan::hector_t<rectangle_t::block_t> blocks;
  #include _FAN_PATH(BLL/BLL.h)

  loco_rectangle_viewport_list_t viewport_list;

  struct idr_t {
    uint64_t block_id : 16;
    uint64_t hector_id : 40;
    uint64_t instance_id : 8;
  };

  struct id_t{
    id_t(rectangle_t* rectangle, fan::opengl::cid_t* cid) {

      idr_t idr = *(idr_t *)&cid->id;
      loco_rectangle_viewport_list_Node_t *n = loco_rectangle_viewport_list_GetNodeByReference(&rectangle->viewport_list, *(loco_rectangle_viewport_list_NodeReference_t*)&idr);
      block = &n->data.blocks[idr.hector_id];
      instance_id = idr.instance_id;
    }
    rectangle_t::block_t *block;
    uint32_t instance_id;
  };

  #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.vs)
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

  void open(loco_t* loco) {
    sb_open(loco);
    loco_rectangle_viewport_list_open(&viewport_list);
  }
  void close(loco_t* loco) {
    sb_close(loco);
    loco_rectangle_viewport_list_close(&viewport_list);
  }
};