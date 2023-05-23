struct stage {
  lstd_defstruct(stage_fuel_station_t)
    #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

    static constexpr auto stage_name = "stage_fuel_station";
    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage_fuel_station.h)
  };
};