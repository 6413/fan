#ifndef stage_loader_path
#define stage_loader_path
#endif

#define stage_maker_loader
#include <fan/graphics/gui/fgm/common.h>

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


struct stage_loader_t {

  stage_loader_t() {
    gstage = this;
  }

protected:
  #define BLL_set_Link 1
  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #include <fan/fan_bll_preset.h>
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix stage_list
  #define BLL_set_type_node uint16_t
  #define bcontainer_set_StoreFormat 1
  #define BLL_set_NodeData \
  loco_t::update_callback_nr_t update_nr; \
  fan::window_t::resize_callback_NodeReference_t resize_id; \
  void *stage;
  #include <BLL/BLL.h>
public:

protected:
  // for safety for getting reference to shape_t in get_id()
  #define bcontainer_set_StoreFormat 1
  //#define BLL_set_CPP_CopyAtPointerChange 1
  #include <fan/fan_bll_preset.h>
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType loco_t::shape_t
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include <BLL/BLL.h>
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

    loco_t::camera_t* camera = &gloco->orthographic_render_view.camera;
    loco_t::viewport_t* viewport = &gloco->orthographic_render_view.viewport;

    stage_loader_t::nr_t parent_id;
    uint32_t itToDepthMultiplier = 0x100;

    void* sod; // stage open data
  };

  struct stage_common_t {
    using open_t = void(*)(void*, void*);
    open_t open;
    using close_t = void(*)(void*);
    close_t close;
    using window_resize_t = void(*)(void*);
    window_resize_t window_resize;
    using update_t = void(*)(void*);
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
  template <typename T>
  static stage_list_NodeReference_t open_stage() {
    return open_stage<T>(stage_open_properties_t());
  }

  #include _PATH_QUOTE(stage_loader_path/stages_compile/stage.h)

protected:
  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #include <fan/fan_bll_preset.h>
  #define BLL_set_prefix stage_list
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeData \
  loco_t::update_callback_nr_t update_nr; \
  fan::window_t::resize_callback_NodeReference_t resize_id; \
  void *stage;
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include <BLL/BLL.h>
public:

  stage_list_t stage_list;

  using key_t = std::pair<void*, std::string>;

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

  loco_t::shape_t& get_id(auto* stage_ptr, const std::string id) {
    auto found = cid_map.find(std::make_pair(stage_ptr, id));
    if (found == cid_map.end()) {
      fan::throw_error("invalid fetch for id - usage shape_{id}", id);
    }

    return stage_ptr->stage_common.cid_list[found->second];
  }

  void load_fgm(auto* stage, const stage_open_properties_t& op, const char* stage_name) {
    std::string filename = std::string("stages_runtime/") + stage_name + ".fgm";
    #define only_struct_data
    #include <fan/graphics/gui/stage_maker/loader_versions/1.h>
  }

  void erase_stage(nr_t id) {
    auto* sc = (stage_common_t*)stage_list[id].stage;
    gloco->window.remove_resize_callback(stage_list[id].resize_id);
    gloco->m_update_callback.unlrec(stage_list[id].update_nr);
    sc->close(stage_list[id].stage);
    stage_list.unlrec(id);
    //stage->close();
   // std::destroy_at(stage);
  }
};
#undef stage_loader_path