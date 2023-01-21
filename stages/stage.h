struct stage {

  struct stage0_t : stage_common_t_t<stage0_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage0";

    typedef int(stage0_t::* cb_table_t)(const loco_t::mouse_button_data_t& v);

    cb_table_t button_click_cb_table[1] = { };    

    #include _PATH_QUOTE(stage_loader_path/stages/stage0.h)
  };

  struct stage1_t : stage_common_t_t<stage1_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage1";

    typedef int(stage1_t::* cb_table_t)(const loco_t::mouse_button_data_t& v);

    cb_table_t button_click_cb_table[2] = {&stage1_t::button0_click_cb,&stage1_t::button1_click_cb,};

    #include _PATH_QUOTE(stage_loader_path/stages/stage1.h)
  };
};