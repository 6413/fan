#ifndef stage_loader_path
#define stage_loader_path
#endif

#include _FAN_PATH(graphics/gui/fgm/common.h)

struct stage_loader_t {
protected:
  #define BLL_set_Link 1
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #define BLL_set_BaseLibrary 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix stage_list
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeData \
  loco_t::update_callback_nr_t update_nr; \
  fan::window_t::resize_callback_NodeReference_t resize_nr; \
  stage::variant_t stage;


  #include _FAN_PATH(BLL/BLL.h)
public:
  using nr_t = stage_list_NodeReference_t;

  struct stage_open_properties_t {
    
    loco_t::camera_t* camera;
    fan::graphics::viewport_t* viewport;
    loco_t::theme_t* theme;

    stage_loader_t::nr_t parent_id;
    uint32_t itToDepthMultiplier = 0x100;
  };

  template <typename T>
  struct stage_common_t_t {

    using value_type_t = stage_common_t_t<T>;

    stage_common_t_t(auto* loader, auto* loco, const stage_open_properties_t& properties) {
    }

    void close(auto* loco) {
      T* stage = (T*)this;
      stage->close(*loco);
    }


    nr_t stage_id;
    uint32_t it;

  protected:
   /* #define BLL_set_CPP_ConstructDestruct
    #define BLL_set_CPP_Node_ConstructDestruct
    #define BLL_set_BaseLibrary 1
    #define BLL_set_prefix cid_list
    #define BLL_set_type_node uint32_t
    #define BLL_set_NodeDataType loco_t::id_t
    #define BLL_set_Link 1
    #define BLL_set_AreWeInsideStruct 1
    #include _FAN_PATH(BLL/BLL.h)*/
  public:

    std::vector<loco_t::id_t> cid_list;

    stage_loader_t::nr_t parent_id;
  };

  #include _PATH_QUOTE(stage_loader_path/stages_compile/stage.h)

protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix stage_list
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeData \
  loco_t::update_callback_nr_t update_nr; \
  fan::window_t::resize_callback_NodeReference_t resize_nr; \
  stage::variant_t stage;
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  stage_list_t stage_list;

  using key_t = std::pair<void*, fan::string>;

  struct pair_hasher_t {
    std::size_t operator()(const key_t& pair) const {
      return std::hash<decltype(pair.first)>()(pair.first) ^ std::hash<std::string>()(pair.second);
    }
  };

  struct pair_equal_t {
    bool operator()(const key_t& lhs, const key_t& rhs) const {
      return lhs.first == rhs.first && lhs.second == rhs.second;
    }
  };

  using cid_map_t = std::unordered_map<key_t, loco_t::cid_nr_t, pair_hasher_t, pair_equal_t>;
  cid_map_t cid_map;

  loco_t::id_t get_id(void* stage_ptr, const fan::string id) {
    auto found = cid_map.find(std::make_pair(stage_ptr, id));
    if (found == cid_map.end()) {
      return nullptr;
    }

    loco_t::id_t idt;
    idt.cid = found->second;
    return idt;
  }

	void open(loco_t::texturepack_t* tp) {
    texturepack = tp;
	}
  void close() {

  }

  void load_fgm(auto* stage, const stage_open_properties_t& op, const char* stage_name) {

    fan::string full_path = fan::string("stages_runtime/") + stage_name + ".fgm";
    fan::string f;
    fan::io::file::read(full_path, &f);
    uint64_t off = 0;

    uint32_t file_version = fan::read_data<uint32_t>(f, off);

    switch (file_version) {
      case stage_maker_format_version: {
        #include _FAN_PATH(graphics/gui/stage_maker/loader_versions/011.h)
        break;
      }
      default: {
        fan::throw_error("invalid version fgm version number", file_version);
        break;
      }
    }
  }

	template <typename stage_t>
	stage_loader_t::nr_t push_and_open_stage(const stage_open_properties_t& op) {
    auto stage = (stage_t*)malloc(sizeof(stage_t));

    stage->stage_id = stage_list.NewNodeLast();
    if (stage->stage_id.Prev(&stage_list) != stage_list.src) {
      std::visit([&](auto o) { stage->it = o->it + 1; }, stage_list[stage->stage_id.Prev(&stage_list)].stage);
      //stage->it = ((stage_common_t *)stage_list[stage->stage_id.Prev(&stage_list)].stage)->it + 1;
    }
    else {
      stage->it = 0;
    }
    stage->parent_id = op.parent_id;

    std::construct_at(stage, this, (loco_access), op);

    load_fgm(stage, op, stage->stage_name);
    //age_list_NodeData_t
		stage_list[stage->stage_id].stage = (stage_t*)stage;
    stage_list[stage->stage_id].update_nr = (loco_access)->m_update_callback.NewNodeLast();
    (loco_access)->m_update_callback[stage_list[stage->stage_id].update_nr] = [&, stage](loco_t* loco) {
      stage->update(*(loco_access));
    };
    stage_list[stage->stage_id].resize_nr = (loco_access)->get_window()->add_resize_callback([&, stage](const auto&) {
      stage->window_resize_callback(*(loco_access));
    });
    stage->open(*(loco_access));
		return stage->stage_id;
	}
	void erase_stage(nr_t id) {
    // ugly
    void* ptr_to_free;
    std::visit([&](auto stage) {
      stage->close(*(loco_access));
      (loco_access)->m_update_callback.unlrec(stage_list[id].update_nr);
      (loco_access)->get_window()->remove_resize_callback(stage_list[id].resize_nr);
      std::destroy_at(stage);
      ptr_to_free = stage;
    }, stage_list[id].stage);
    stage_list.unlrec(id);
    std::free(ptr_to_free);
	}

  loco_t::texturepack_t* texturepack = 0;
};

#undef stage_loader_path