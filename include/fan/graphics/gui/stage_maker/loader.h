#ifndef stage_loader_path
#define stage_loader_path
#endif

#include _FAN_PATH(graphics/gui/fgm/common.h)

struct stage_loader_t;

static inline struct gstage_t {

  stage_loader_t* loader;

  operator stage_loader_t* () {
    return loader;
  }
  gstage_t& operator=(stage_loader_t* l) {
    loader = l;
    return *this;
  }
  stage_loader_t* operator->() {
    return loader;
  }
}gstage;

template <typename Variant>
struct ptrless_variant_helper {
    template <std::size_t... Is>
    static auto helper(std::index_sequence<Is...>) {
        return std::variant<std::remove_pointer_t<std::variant_alternative_t<Is, Variant>>...>{};
    }
};

struct stage_loader_t {

  stage_loader_t() {
    gstage = this;
  }

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
  #define BLL_set_StoreFormat 1
  #define BLL_set_NodeData \
  loco_t::update_callback_nr_t update_nr; \
  fan::window_t::resize_callback_NodeReference_t resize_nr; \
  void *stage;
  #include _FAN_PATH(BLL/BLL.h)
public:

  protected:
  // for safety for getting reference to shape_t in get_id()
  #define BLL_set_StoreFormat 1
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType loco_t::shape_t
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  using cid_nr_t = cid_list_NodeReference_t;

  struct nr_t : stage_list_NodeReference_t {
    using base_nr_t = stage_list_NodeReference_t;
    using base_nr_t::base_nr_t;

    nr_t() = default;
    nr_t(base_nr_t nr) : base_nr_t(nr) {}

    void erase() {
      #if fan_debug >= 2
      if (iic()) {
        fan::throw_error("double erase or uninitialized erase");
      }
      #endif
      gstage->erase_stage(*this);
      sic();
    }
  };

  struct stage_open_properties_t {
    
    loco_t::camera_t* camera;
    fan::graphics::viewport_t* viewport;
    loco_t::theme_t* theme;

    stage_loader_t::nr_t parent_id;
    uint32_t itToDepthMultiplier = 0x100;

    void *sod; // stage open data
  };

  struct stage_common_t {
    using open_t = void(*)(void *, void *);
    open_t open;
    using close_t = void(*)(void *);
    close_t close;
    using window_resize_t = void(*)(void *);
    window_resize_t window_resize;
    using update_t = void(*)(void *);
    update_t update;

    stage_list_NodeReference_t stage_id;
    uint32_t it;

    cid_list_t cid_list;

    stage_loader_t::stage_list_NodeReference_t parent_id;
  };

  template <typename T>
  static stage_list_NodeReference_t open_stage(const stage_open_properties_t& op) {
    T* stage = new T{ .structor{op} };
    T::_stage_open(stage, op.sod);
    return stage->stage_common.stage_id;
  }

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
  void *stage;
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

  // cid_list to bll and use bll nr as key to get 
  // nr = newnodelast()
  // list[nr] = p
  // map[name] = nr
  // in get_id
  // return reference to list[map[name]]
  using cid_map_t = std::unordered_map<key_t, cid_nr_t, pair_hasher_t, pair_equal_t>;
  cid_map_t cid_map;

  loco_t::shape_t& get_id(auto* stage_ptr, const fan::string id) {
    auto found = cid_map.find(std::make_pair(stage_ptr, id));
    if (found == cid_map.end()) {
      fan::throw_error("invalid fetch for id - usage shape_{id}", id);
    }

    return stage_ptr->stage_common.cid_list[found->second];
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

	//template <typename stage_t>
	//stage_loader_t::nr_t open_stage(const stage_open_properties_t& op) {


	//	return stage->stage_id;
	//}
	void erase_stage(nr_t id) {
    auto* sc = (stage_common_t*)stage_list[id].stage;
    fan::print(stage_list[id].update_nr.NRI, stage_list[id].resize_nr.NRI);
    gloco->m_update_callback.unlrec(stage_list[id].update_nr);
    gloco->get_window()->remove_resize_callback(stage_list[id].resize_nr);
    sc->close(stage_list[id].stage);
    //stage->close();
   // std::destroy_at(stage);
	}

  loco_t::texturepack_t* texturepack = 0;

  lstd_defstruct(custom_base_t)
    #include "preset.h"
    
    static constexpr auto stage_name = "";

    void open(void *sod) {
  
    }

    void close() {
		
    }

    void window_resize(){
		
    }

    void update(){
	
    }

  };

};

#define fan_make_custom_stage(name) \
  lstd_defstruct(name) \
  static constexpr const char* stage_name = ""; \
  stage_loader_t::stage_common_t stage_common = { \
    .open = _stage_open, \
    .close = _stage_close, \
    .window_resize = _stage_window_resize, \
    .update = _stage_update \
  }; \
   \
  static void _stage_open(void* ptr, void *sod) { \
    ((lstd_current_type*)ptr)->open(sod); \
  } \
   \
  static void _stage_close(void* ptr) { \
    ((lstd_current_type*)ptr)->close(); \
    delete (lstd_current_type*)ptr; \
  }   \
   \
  static void _stage_window_resize(void* ptr){ \
	  ((lstd_current_type*)ptr)->window_resize(); \
  } \
   \
  static void _stage_update(void* ptr){ \
    ((lstd_current_type*)ptr)->update(); \
  } \
   \
  struct structor_t{ \
    structor_t(const stage_loader_t::stage_open_properties_t& op) { \
      auto outside = OFFSETLESS(this, lstd_current_type, structor); \
      auto nr = gstage->stage_list.NewNodeLast(); \
      outside->stage_common.stage_id = nr; \
      outside->stage_common.parent_id = op.parent_id; \
      gstage->stage_list[nr].stage = outside; \
      if (outside->stage_common.stage_id.Prev(&gstage->stage_list) != gstage->stage_list.src) { \
        outside->stage_common.it = ((stage_loader_t::stage_common_t *)gstage->stage_list[outside->stage_common.stage_id.Prev(&gstage->stage_list)].stage)->it + 1; \
      } \
      else { \
        outside->stage_common.it = 0; \
      } \
       \
      gstage->stage_list[outside->stage_common.stage_id].update_nr = gloco->m_update_callback.NewNodeLast(); \
      gloco->m_update_callback[gstage->stage_list[outside->stage_common.stage_id].update_nr] = [&, outside](loco_t*  loco) { \
        outside->update(); \
      }; \
      gstage->stage_list[outside->stage_common.stage_id].resize_nr = gloco->get_window()->add_resize_callback([&,  outside](const auto&) { \
        outside->window_resize(); \
      }); \
    } \
  }structor;


#undef stage_loader_path