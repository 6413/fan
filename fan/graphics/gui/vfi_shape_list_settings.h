#define bcontainer_set_StoreFormat 1
#define BLL_set_AreWeInsideStruct 1

#include <fan/fan_bll_preset.h>
#define BLL_set_prefix shape_list
#define BLL_set_type_node shape_id_integer_t
#define BLL_set_NodeData \
    shape_type_t shape_type; \
    common_shape_data_t shape_data; \
    iflags_t flags;
#define BLL_set_Link 1
#define BLL_set_NodeReference_Overload_Declare \
    bool is_invalid() const { \
      return NRI == (shape_id_integer_t)fan::uninitialized; \
    } \
    void invalidate() { \
      NRI = fan::uninitialized; \
    }