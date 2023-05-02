struct stage {

  lstd_defstruct(stage0_t)
    #include "preset.h"
    
    static constexpr auto stage_name = "stage0";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage0.h)
  };
};