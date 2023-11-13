#define BLL_set_CPP_ConstructDestruct
#include _FAN_PATH(fan_bll_present.h)
#define BLL_set_prefix image_list
#define BLL_set_type_node uint8_t
#define BLL_set_NodeData fan::opengl::GLuint texture_id; loco_t::image_t* image;
#define BLL_set_Link 0
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeReference_Overload_Declare \
  image_list_NodeReference_t(loco_t::image_t* image);