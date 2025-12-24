#define BLL_set_prefix CONCAT3(shape, _, list)
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node shape_nr_t
#define BLL_set_NodeDataType fan::graphics::shapes::CONCAT3(shape, _, t)::properties_t
#define BLL_set_CPP_CopyAtPointerChange 1
#define BLL_set_AreWeInsideStruct 1
#include <BLL/BLL.h>
CONCAT4(shape, _, list, _t) CONCAT3(shape, _, list);
#undef shape