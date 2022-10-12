#define BLL_set_BaseLibrary 1
#define BLL_set_namespace fan::opengl
#define BLL_set_prefix matrices_list
#define BLL_set_type_node uint8_t
#define BLL_set_node_data fan::opengl::matrices_t* matrices_id;
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 0
#define BLL_set_NodeReference_Overload_Declare \
  matrices_list_NodeReference_t() = default; \
  matrices_list_NodeReference_t(fan::opengl::matrices_t* matrices);