struct dropdown_t {
  struct open_properties_t {
    loco_t::camera_t* camera = 0;
    loco_t::viewport_t* viewport = 0;
    loco_t::theme_t* theme = 0;

    fan::vec3 position;
    f32_t gui_size = 0;
    bool title = false;

    open_properties_t() = default;
  };

  struct element_properties_t {
    fan::string text;
    loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> int { return 0; };
  };

  struct element_t : loco_t::shape_t {
    using shape_t::shape_t;
    __create_assign_operators(loco_t::shape_t);
  };

protected:
  #define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix element_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType element_t
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)

  #define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix menu_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType __nameless_type_t<open_properties_t, element_list_t>
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  using element_list_nr_t = element_list_NodeReference_t;
  using menu_nr_t = menu_list_NodeReference_t;

  element_list_t element_list;
  menu_list_t menu_list;

  struct menu_id_t : menu_nr_t{
    menu_id_t(const open_properties_t& op) : 
      menu_nr_t(gloco->dropdown.menu_list.NewNodeLast()) {
      gloco->dropdown.menu_list[*this] = op;
    }
    void add(const element_properties_t& ep) {
      auto& instance = gloco->dropdown.menu_list[*this];
      // idk if i need this anywhere
      auto nr = instance.NewNodeLast();
      loco_t::button_t::properties_t p;
      p.camera = instance.camera;
      p.viewport = instance.viewport;
      p.theme = instance.theme;
      p.size = fan::vec2(instance.gui_size * 5, instance.gui_size);
      p.position = instance.position + fan::vec2(0, p.size.y * 2 * instance.Usage());
      p.text = ep.text;
      p.font_size = instance.gui_size;
      p.mouse_button_cb = ep.mouse_button_cb;
      p.mouse_move_cb = ep.mouse_move_cb;
      p.keyboard_cb = ep.keyboard_cb;
      instance[nr] = p;
    }
  };

}dropdown;