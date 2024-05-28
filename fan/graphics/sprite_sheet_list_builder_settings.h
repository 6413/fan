#include <fan/fan_bll_preset.h>
//#define BLL_set_namespace
#define BLL_set_prefix sheet_list
#define BLL_set_type_node uint16_t
#define BLL_set_NodeData \
loco_t::shape_t shape; \
sheet_t sheet; \
bool start = 0; \
fan::ev_timer_t::id_t timer_id;
#define BLL_set_Link 1
#define BLL_set_AreWeInsideStruct 1