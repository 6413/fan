#define BLL_set_CPP_ConstructDestruct
#include _FAN_PATH(fan_bll_preset.h)
#define BLL_set_prefix camera_list
#define BLL_set_type_node uint8_t
#define BLL_set_NodeData loco_t::camera_t* camera_id;
#define BLL_set_Link 0
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeReference_Overload_Declare \
  camera_list_NodeReference_t(loco_t::camera_t* camera);