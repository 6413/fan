#define BLL_set_BaseLibrary 1
#define BLL_set_namespace fan::opengl
#define BLL_set_prefix viewport_list
#define BLL_set_type_node uint8_t
#define BLL_set_node_data fan::opengl::viewport_t* viewport_id;
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_StructFormat 1
#define BLL_set_NodeReference_Overload_Declare \
  viewport_list_NodeReference_t() = default; \
  viewport_list_NodeReference_t(fan::opengl::viewport_t*);