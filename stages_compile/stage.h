struct stage {

  struct stage0_t : stage_common_t_t<stage0_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage0";

    typedef int(stage0_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& v);

    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = {};

    typedef int(stage0_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage0_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage0_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage0_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage0.h)
  };

  struct stage1_t : stage_common_t_t<stage1_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage1";

    typedef int(stage1_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& v);

    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = { };

    typedef int(stage1_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage1_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage1_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage1_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[1] = { };
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[1] = { };
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[1] = { };
    hitbox_text_cb_table_t hitbox_text_cb_table[1] = { };

    #include _PATH_QUOTE(stage_loader_path/stages_compile/stage1.h)
  };
};