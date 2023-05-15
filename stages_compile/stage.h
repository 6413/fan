struct stage {
  struct lstd_defstruct(stage1_t) {
    #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

    static constexpr auto stage_name = "stage1";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage1.h)
  };
  struct lstd_defstruct(stage_test_t) {
    #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

    static constexpr auto stage_name = "stage_test";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage_test.h)
  };
  struct lstd_defstruct(stage0_t) {
    #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

    static constexpr auto stage_name = "stage0";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage0.h)
  };
};