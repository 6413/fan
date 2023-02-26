struct stage {

  struct stage0_t : stage_common_t_t<stage0_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage0";

    typedef int(stage0_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage0_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage0_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage0_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = { };
    button_mouse_move_cb_table_t button_mouse_move_cb_table[1] = { };
    button_keyboard_cb_table_t button_keyboard_cb_table[1] = { };
    button_text_cb_table_t button_text_cb_table[1] = { };
    //button_dst
  
    typedef int(stage0_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage0_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage0_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage0_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage0.h)
  };

  struct stage1_t : stage_common_t_t<stage1_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage1";

    typedef int(stage1_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage1_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage1_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage1_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[2] = {&stage1_t::button0_mouse_button_cb,&stage1_t::button1_mouse_button_cb,};
    button_mouse_move_cb_table_t button_mouse_move_cb_table[2] = {&stage1_t::button0_mouse_move_cb,&stage1_t::button1_mouse_move_cb,};
    button_keyboard_cb_table_t button_keyboard_cb_table[2] = {&stage1_t::button0_keyboard_cb,&stage1_t::button1_keyboard_cb,};
    button_text_cb_table_t button_text_cb_table[2] = {&stage1_t::button0_text_cb,&stage1_t::button1_text_cb,};
    //button_dst
  
    typedef int(stage1_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage1_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage1_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage1_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage1.h)
  };

  struct stage2_t : stage_common_t_t<stage2_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage2";

    typedef int(stage2_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage2_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage2_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage2_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = {&stage2_t::button0_mouse_button_cb,};
    button_mouse_move_cb_table_t button_mouse_move_cb_table[1] = {&stage2_t::button0_mouse_move_cb,};
    button_keyboard_cb_table_t button_keyboard_cb_table[1] = {&stage2_t::button0_keyboard_cb,};
    button_text_cb_table_t button_text_cb_table[1] = {&stage2_t::button0_text_cb,};
    //button_dst
  
    typedef int(stage2_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage2_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage2_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage2_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage2.h)
  };

  struct stage_parallax_t : stage_common_t_t<stage_parallax_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage_parallax";

    typedef int(stage_parallax_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage_parallax_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage_parallax_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage_parallax_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = { };
    button_mouse_move_cb_table_t button_mouse_move_cb_table[1] = { };
    button_keyboard_cb_table_t button_keyboard_cb_table[1] = { };
    button_text_cb_table_t button_text_cb_table[1] = { };
    //button_dst
  
    typedef int(stage_parallax_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage_parallax_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage_parallax_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage_parallax_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage_parallax.h)
  };

  struct stage_gui_t : stage_common_t_t<stage_gui_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage_gui";

    typedef int(stage_gui_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage_gui_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage_gui_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage_gui_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = { };
    button_mouse_move_cb_table_t button_mouse_move_cb_table[1] = { };
    button_keyboard_cb_table_t button_keyboard_cb_table[1] = { };
    button_text_cb_table_t button_text_cb_table[1] = { };
    //button_dst
  
    typedef int(stage_gui_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage_gui_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage_gui_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage_gui_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage_gui.h)
  };

  struct stage3_t : stage_common_t_t<stage3_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage3";

    typedef int(stage3_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage3_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage3_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage3_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = { };
    button_mouse_move_cb_table_t button_mouse_move_cb_table[1] = { };
    button_keyboard_cb_table_t button_keyboard_cb_table[1] = { };
    button_text_cb_table_t button_text_cb_table[1] = { };
    //button_dst
  
    typedef int(stage3_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage3_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage3_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage3_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage3.h)
  };

  struct stage4_t : stage_common_t_t<stage4_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage4";

    typedef int(stage4_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage4_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage4_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage4_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = { };
    button_mouse_move_cb_table_t button_mouse_move_cb_table[1] = { };
    button_keyboard_cb_table_t button_keyboard_cb_table[1] = { };
    button_text_cb_table_t button_text_cb_table[1] = { };
    //button_dst
  
    typedef int(stage4_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage4_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage4_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage4_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage4.h)
  };

  struct stage5_t : stage_common_t_t<stage5_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage5";

    typedef int(stage5_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage5_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage5_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage5_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = { };
    button_mouse_move_cb_table_t button_mouse_move_cb_table[1] = { };
    button_keyboard_cb_table_t button_keyboard_cb_table[1] = { };
    button_text_cb_table_t button_text_cb_table[1] = { };
    //button_dst
  
    typedef int(stage5_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage5_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage5_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage5_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    //hitbox_src
    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };
    //hitbox_dst

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage5.h)
  };
};