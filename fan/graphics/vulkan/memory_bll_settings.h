#include <fan/fan_bll_preset.h>
#define BLL_set_SafeNext 1
#define BLL_set_prefix write_queue
#define BLL_set_type_node uint32_t
#define BLL_set_NodeData \
	memory_edit_cb_t cb;
#define BLL_set_Link 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_CPP_CopyAtPointerChange 1