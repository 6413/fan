struct stage_common_t {

	struct open_properties_t {
		loco_t::matrices_list_NodeReference_t matrices;
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
  inline static std::vector<stage_common_t*> stages;
};
