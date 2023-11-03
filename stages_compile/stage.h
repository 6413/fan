struct stage {
  lstd_defstruct(stage6_t)
    #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

    static constexpr auto stage_name = "stage6";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage6.h)
  };
  lstd_defstruct(stage0_t)
    #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

    static constexpr auto stage_name = "stage0";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage0.h)
  };
};