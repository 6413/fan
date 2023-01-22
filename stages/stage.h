struct stage {

  struct stage0_t : stage_common_t_t<stage0_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage0";

    typedef int(stage0_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& v);

    button_mouse_button_cb_table_t button_mouse_button_cb_table[1] = {&stage0_t::button0_click_cb,};

    typedef int(stage0_t::* hitbox_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage0_t::* hitbox_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage0_t::* hitbox_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage0_t::* hitbox_text_cb_table_t)(const loco_t::text_data_t& d);

    hitbox_mouse_button_cb_table_t hitbox_mouse_button_cb_table[2] = {&stage0_t::hitbox0_mouse_button_cb,&stage0_t::hitbox1_mouse_button_cb,};
    hitbox_mouse_move_cb_table_t hitbox_mouse_move_cb_table[2] = {&stage0_t::hitbox0_mouse_move_cb,&stage0_t::hitbox1_mouse_move_cb,};
    hitbox_keyboard_cb_table_t hitbox_keyboard_cb_table[2] = {&stage0_t::hitbox0_keyboard_cb,&stage0_t::hitbox1_keyboard_cb,};
    hitbox_text_cb_table_t hitbox_text_cb_table[2] = {&stage0_t::hitbox0_text_cb,&stage0_t::hitbox1_text_cb,};

    #include _PATH_QUOTE(stage_loader_path/stages/stage0.h)
  };
};