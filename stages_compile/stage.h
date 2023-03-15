struct stage {

  struct stage0_t : stage_common_t_t<stage0_t> {

    using stage_common_t_t::stage_common_t_t;

    static constexpr auto stage_name = "stage0";

    typedef int(stage0_t::* button_mouse_button_cb_table_t)(const loco_t::mouse_button_data_t& d);
    typedef int(stage0_t::* button_mouse_move_cb_table_t)(const loco_t::mouse_move_data_t& d);
    typedef int(stage0_t::* button_keyboard_cb_table_t)(const loco_t::keyboard_data_t& d);
    typedef int(stage0_t::* button_text_cb_table_t)(const loco_t::text_data_t& d);

    //button_src
    button_mouse_button_cb_table_t button_mouse_button_cb_table[2] = {&stage0_t::buttoninspector_mouse_button_cb,&stage0_t::buttoninspector_mouse_button_cb,};
    button_mouse_move_cb_table_t button_mouse_move_cb_table[2] = {&stage0_t::buttoninspector_mouse_move_cb,&stage0_t::buttoninspector_mouse_move_cb,};
    button_keyboard_cb_table_t button_keyboard_cb_table[2] = {&stage0_t::buttoninspector_keyboard_cb,&stage0_t::buttoninspector_keyboard_cb,};
    button_text_cb_table_t button_text_cb_table[2] = {&stage0_t::buttoninspector_text_cb,&stage0_t::buttoninspector_text_cb,};
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

  using variant_t = std::variant<stage0_t*>;
};