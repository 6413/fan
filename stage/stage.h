struct stage_common_t{
	fan::function_t<void()> open;
	fan::function_t<void()> close;
	fan::function_t<void()> window_resize_callback;
	fan::function_t<void()> update;
};
		
struct stage {
  struct stage0_t {
    #include "stage/stage0.h"
  };
  struct stage1_t {
    #include "stage/stage1.h"
  };
  struct stage2_t {
    #include "stage/stage2.h"
  };
};
