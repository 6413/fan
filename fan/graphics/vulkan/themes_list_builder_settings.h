#include _FAN_PATH(fan_bll_preset.h)
#define BLL_set_namespace fan::vulkan
#define BLL_set_prefix theme_list
#define BLL_set_type_node uint8_t
#define BLL_set_NodeData void* theme_id;
#define BLL_set_Link 0
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 0
#define BLL_set_NodeReference_Overload_Declare \
  theme_list_NodeReference_t(void*);