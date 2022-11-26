struct stage_common_t {

	#define BLL_set_StoreFormat 1
	#define BLL_set_BaseLibrary 1
	#define BLL_set_AreWeInsideStruct 1
	#define BLL_set_prefix instance
	#define BLL_set_type_node uint16_t
	#define BLL_set_node_data \
			fan::graphics::cid_t cid;
	#define BLL_set_Link 1
	#include _FAN_PATH(BLL/BLL.h)

	instance_t instances;

	struct open_properties_t {
		fan::graphics::matrices_list_NodeReference_t matrices;
		fan::graphics::viewport_list_NodeReference_t viewport;
		fan::graphics::theme_list_NodeReference_t theme;
	};

	fan::function_t<void()> open;
	fan::function_t<void()> close;
	fan::function_t<void()> window_resize_callback;
	fan::function_t<void()> update;
};

struct stage {
	struct stage0_t {
    #include "stages/stage0.h"
  };
  struct stage1_t {
    #include "stages/stage1.h"
  };
  struct stage2_t {
    #include "stages/stage2.h"
  };
  inline static std::vector<stage_common_t*> stages;
};
