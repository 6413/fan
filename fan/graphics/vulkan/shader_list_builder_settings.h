#include <fan/fan_bll_preset.h>
#define BLL_set_StoreFormat 1
#define BLL_set_prefix shader_list
#define BLL_set_type_node uint8_t
#define BLL_set_NodeData \
VkPipelineShaderStageCreateInfo shaderStages[2];
//TODO
//fan::vulkan::core::uniform_block_t<fan::vulkan::context_t::viewprojection_t, fan::vulkan::max_camera> projection_view_block;\
fan::function_t<void(shader_t* shader)> on_activate = [](shader_t* shader){}; /*used to allow uniform variable binding*/
#define BLL_set_Link 0
#define BLL_set_IsNodeRecycled 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeReference_Overload_Declare \
  shader_list_NodeReference_t(shader_t* image);