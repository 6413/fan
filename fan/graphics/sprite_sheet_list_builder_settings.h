#define BLL_set_BaseLibrary 1
//#define BLL_set_namespace
#define BLL_set_prefix sheet_list
#define BLL_set_type_node uint16_t
#define BLL_set_NodeData \
loco_t::shape_t shape; \
sheet_t sheet; \
bool start = 0; \
fan::ev_timer_t::timer_t timer;
#define BLL_set_Link 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_CPP_ConstructDestruct