struct stage {

  lstd_defstruct(stage0_t)
    #include "preset.h"
    
    static constexpr auto stage_name = "stage0";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage0.h)
  };

  lstd_defstruct(stage1_t)
    #include "preset.h"
    
    static constexpr auto stage_name = "stage1";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage1.h)
  };

  lstd_defstruct(stage2_t)
    #include "preset.h"
    
    static constexpr auto stage_name = "stage2";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage2.h)
  };

  lstd_defstruct(stage_hud_t)
    #include "preset.h"
    
    static constexpr auto stage_name = "stage_hud";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage_hud.h)
  };
};