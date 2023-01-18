protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix stage_list
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType void *
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)
public:

using nr_t = stage_list_NodeReference_t;

struct stage_open_properties_t {
	loco_t::matrices_list_NodeReference_t matrices;
	fan::graphics::viewport_list_NodeReference_t viewport;
	fan::graphics::theme_list_NodeReference_t theme;
};

template <typename T = __empty_struct>
struct stage_common_t_t {

  stage_common_t_t(auto* loco, const stage_open_properties_t& properties) {
    T* stage = (T*)this;
    loco->stage_loader.load_fgm((T*)this, properties, stage->stage_name);
    stage->open(loco);
  }
  void close(auto* loco) {
    T* stage = (T*)this;
    stage->close(loco);
  }

  nr_t stage_id;

protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::graphics::cid_t
  #define BLL_set_Link 1
  #define BLL_set_StoreFormat 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_StoreFormat1_ElementPerBlock 0x100
  #include _FAN_PATH(BLL/BLL.h)
public:

  cid_list_t cid_list;
};

using stage_common_t = stage_common_t_t<>;

struct stage {
  inline static stage_list_t stage_list;
  struct stage0_t : stage_common_t_t<stage0_t> {
    #include "stages/stage0.h"
  };
  
};

