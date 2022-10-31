#define BLL_set_CPP_ConstructDestruct
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_BaseLibrary 1
#define BLL_set_prefix matrices_list

#if BLL_set_declare_rest != 0
struct matrices_index_t {
  #if defined(loco_rectangle)
    uint8_t rectangle = -1;
  #endif
  #if defined(loco_sprite)
    uint8_t sprite = -1;
  #endif
  #if defined(loco_letter)
    uint8_t letter = -1;
  #endif
  #if defined(loco_button)
    uint8_t button = -1;
  #endif
};
#endif

#define BLL_set_type_node uint8_t
#define BLL_set_node_data \
  loco_t::matrices_t* matrices_id; \
  matrices_index_t matrices_index;
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeReference_Overload_Declare \
  matrices_list_NodeReference_t() = default; \
  matrices_list_NodeReference_t(loco_t::matrices_t* matrices);