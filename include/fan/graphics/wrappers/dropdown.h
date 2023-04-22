struct dropdown_t {
  struct open_properties_t {
    loco_t::camera_t* camera = 0;
    loco_t::viewport_t* viewport = 0;
    loco_t::theme_t* theme = 0;

    fan::vec3 position;
    f32_t gui_size = 0;
    bool title = false;
  };

  struct element_t : loco_t::shape_t {

  };

protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix element_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType element_t
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix menu_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType fan::return_type_of_t<decltype([]{ struct : open_properties_t, element_list_t{}v; return v;})>
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  using element_list_nr_t = element_list_NodeReference_t;
  using menu_nr_t = menu_list_NodeReference_t;

  element_list_t element_list;
  menu_list_t menu_list;

  struct menu_id_t : menu_nr_t{
    menu_id_t(const open_properties_t& op) : menu_nr_t(gloco->dropdown.menu_list.NewNodeLast()) {
        gloco->dropdown.menu_list[*this] = op;
    }
  };
}dropdown;