#include <fan/fan_bll_preset.h>
#define BLL_set_prefix camera_list

#if BLL_set_declare_rest != 0
struct camera_index_t {
  #if defined(loco_line)
    uint8_t line = -1;
  #endif
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
  #if defined(loco_text_box)
    uint8_t text_box = -1;
  #endif
  #if defined(loco_yuv420p)
    uint8_t yuv420p = -1;
  #endif
};
#endif

#define BLL_set_type_node uint8_t
#define BLL_set_NodeData \
  loco_t::camera_t* camera_id; \
  camera_index_t camera_index;
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeReference_Overload_Declare \
  camera_list_NodeReference_t(loco_t::camera_t* camera);