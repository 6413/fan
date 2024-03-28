#define BLL_set_CPP_ConstructDestruct
#include _FAN_PATH(fan_bll_preset.h)
#define BLL_set_prefix shader_list
#define BLL_set_type_node uint8_t
#define BLL_set_NodeData  \
fan::opengl::GLuint id; \
int projection_view[2]; \
uint32_t vertex, fragment; \
loco_t::shader_t* shader; \
fan::string svertex, sfragment; \
fan::function_t<void(loco_t::shader_t* shader)> on_activate = [](loco_t::shader_t* shader){}; /*used to allow uniform variable binding*/
#define BLL_set_Link 0
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeReference_Overload_Declare \
  shader_list_NodeReference_t(loco_t::shader_t* image);