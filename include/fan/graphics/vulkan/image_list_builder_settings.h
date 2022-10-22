#define BLL_set_BaseLibrary 1
#define BLL_set_namespace fan::vulkan
#define BLL_set_prefix image_list
#define BLL_set_type_node uint8_t
#define BLL_set_node_data \
  VkImage image; \
  VkImageView image_view; \
  VkDeviceMemory image_memory; \
  VkSampler sampler;
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 0
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_CPP_ConstructDestruct
#define BLL_set_NodeReference_Overload_Declare \
  image_list_NodeReference_t() = default; \
  image_list_NodeReference_t(fan::vulkan::image_t* image);