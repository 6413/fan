#include <fan/fan_bll_preset.h>
#define BLL_set_prefix viewport_list
#define BLL_set_type_node uint8_t
#define BLL_set_NodeData fan::vulkan::viewport_t* viewport_id;
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 0
#define BLL_set_NodeReference_Overload_Declare \
  viewport_list_NodeReference_t(fan::vulkan::viewport_t*);