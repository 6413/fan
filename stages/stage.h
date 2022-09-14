struct stage_common_t{
	fan::function_t<void()> open;
	fan::function_t<void()> close;
	fan::function_t<void()> window_resize_callback;
	fan::function_t<void()> update;
};
		
struct stage {
	static struct stage0_t {
    #include "stage/stage0.h"
  }stage0;
  static std::vector<uint64_t> stages{
	   new stage0_t(stage0.stage_common),
  };
};
