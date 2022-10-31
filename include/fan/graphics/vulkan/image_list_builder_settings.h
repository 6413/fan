#define BLL_set_BaseLibrary 1
#define BLL_set_prefix image_list
#define BLL_set_type_node uint8_t

struct image_list_texture_index_t {
  #if defined(loco_sprite)
    uint8_t sprite = -1;
  #endif
  #if defined(loco_letter)
    uint8_t letter = -1;
  #endif
};

#define BLL_set_node_data \
  image_list_texture_index_t texture_index; \
  VkImage image; \
  VkImageView image_view; \
  VkDeviceMemory image_memory; \
  VkSampler sampler;
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_CPP_ConstructDestruct
#define BLL_set_NodeReference_Overload_Declare \
  image_list_NodeReference_t() = default; \
  image_list_NodeReference_t(loco_t::image_t* image);