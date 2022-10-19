#define BLL_set_BaseLibrary 1
//#define BLL_set_namespace
#define BLL_set_CPP_ConstructDestruct
#define BLL_set_prefix descriptor_list
#define BLL_set_type_node uint16_t
#define BLL_set_node_data \
	VkDescriptorSet descriptor_set[fan::vulkan::MAX_FRAMES_IN_FLIGHT];
	//std::vector<fan::vulkan::write_descriptor_set_t> descriptor_properties; 
#define BLL_set_Link 1
#define BLL_set_AreWeInsideStruct 1