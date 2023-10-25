struct dropdown_t {
  struct open_properties_t {
    loco_t::camera_t* camera = 0;
    loco_t::viewport_t* viewport = 0;
    loco_t::theme_t* theme = 0;

    fan::vec3 position = 0;
    fan::vec2 gui_size = 0;

    bool titleable = true;
    fan::string title;

    fan::vec2 direction = fan::vec2(0, 1);

    // todo remove and use only text box for everything
    bool text_box = false;

    open_properties_t() = default;
  };

  struct element_properties_t {
    fan::string text;
    
    loco_t::mouse_button_cb_t internal_mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
    loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; };
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> int { return 0; };
  };

  struct element_t : loco_t::shape_t {
    using shape_t::shape_t;
    //__create_assign_operators(loco_t::shape_t, element_properties_t);

    element_properties_t ep;

    fan::string get_text() const {
      return ep.text;
    }

    fan::string get_text_shape() {
      return loco_t::shape_t::get_text();
    }

    void disable_draw() {
      erase();
    }
    void enable_draw(auto& instance, uint32_t index) {
      if (instance.text_box) {
        loco_t::text_box_t::properties_t p;
        p.camera = instance.camera;
        p.viewport = instance.viewport;
        p.theme = instance.theme;
  //      fan::print(instance.gui_size);
        p.size = instance.gui_size;
        p.position = instance.position + fan::vec3(p.size * 2 * instance.direction, 0) * index;
        p.text = ep.text;
        p.font_size = instance.gui_size.y;
        p.mouse_button_cb = ep.internal_mouse_button_cb;
        p.mouse_move_cb = ep.mouse_move_cb;
        p.keyboard_cb = ep.keyboard_cb;
        *(loco_t::shape_t*)this = p;
      }
      else {
        loco_t::button_t::properties_t p;
        p.camera = instance.camera;
        p.viewport = instance.viewport;
        p.theme = instance.theme;
  //      fan::print(instance.gui_size);
        p.size = instance.gui_size;
        p.position = instance.position + fan::vec3(p.size * 2 * instance.direction, 0) * index;
        p.text = ep.text;
        p.font_size = instance.gui_size.y;
        p.mouse_button_cb = ep.internal_mouse_button_cb;
        p.mouse_move_cb = ep.mouse_move_cb;
        p.keyboard_cb = ep.keyboard_cb;
        *(loco_t::shape_t*)this = p;
        fan::print("pushed", p.text, (int)((loco_t::shape_t*)this)->NRI);
      }
    }
  };

protected:
  //#define BLL_set_StoreFormat 1
  //#define BLL_set_debug_InvalidAction 1
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix element_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType element_t
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)

  using element_list_nr_t = element_list_NodeReference_t;

  struct menu_data_t {
    element_list_nr_t selected_id;
    struct {
      uint8_t titleable : 1 = true, expanded : 1;
    }flags;
    menu_data_t() {
      selected_id.sic();
    }
  };

  //  #define BLL_set_debug_InvalidAction 1
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix menu_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType __nameless_type_t<menu_data_t, open_properties_t, element_list_t>
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  using menu_nr_t = menu_list_NodeReference_t;
 
  menu_list_t menu_list;

  static element_list_nr_t find_element_from_button(auto& instance, const loco_t::cid_nt_t& id) {
    auto nr = instance.GetNodeFirst();
    while (nr != instance.dst) {
      if (id == instance[nr]) {
        return nr;
      }
      nr = nr.Next(&instance);
    }
    __abort();
    return nr;
  }

  struct menu_id_t : menu_nr_t{

    void clear() {
      if (iic()) {
        return;
      }
      gloco->dropdown.menu_list.unlrec(*this);
      sic();
    }

    menu_id_t() {
      sic();
    }
    ~menu_id_t() {
      clear();
    }

    auto add(element_properties_t ep) {
      menu_nr_t instance_nr = *this;
      auto& instance = gloco->dropdown.menu_list[instance_nr];
      // idk if i need this anywhere
      auto element_nr = instance.NewNodeLast();
      // dont use pointer inside lambda
      ep.internal_mouse_button_cb = [this, instance_nr, element_nr, count = instance.Usage()](const auto& d) -> int {
        if (d.button != fan::mouse_left) {
          return 0;
        }
        if (d.button_state != fan::mouse_state::release) {
          return 0;
        }

        auto& instance = gloco->dropdown.menu_list[instance_nr];
        instance.selected_id = element_nr;
        if (instance.flags.titleable == true) {
          uint32_t index = 1;
          if (instance.flags.expanded == false) {
            auto inr = instance.GetNodeFirst();
            if (inr == instance.dst) {
              goto gt_end_expanded0;
            }
            inr = inr.Next(&instance);
            while (inr != instance.dst) {
              auto& ii = instance[inr];
              ii.enable_draw(instance, index);
              inr = inr.Next(&instance);
              index++;
            }
            gt_end_expanded0:;
          }
          else {
            auto inr = instance.GetNodeFirst();
            if (inr == instance.dst) {
              goto gt_end_expanded1;
            }
            if (instance[inr] != d.id) {
              auto text = ((loco_t::shape_t*)&(*(loco_t::cid_nr_t*)&d.id))->get_text();
              instance[inr].set_text(text);
              instance[inr].ep.text = text;
              auto dst_element = loco_t::dropdown_t::find_element_from_button(instance, d.id);
              instance.selected_id = dst_element;
            }
            inr = inr.Next(&instance);
            while (inr != instance.dst) {
              auto& ii = instance[inr];
              ii.disable_draw();
              inr = inr.Next(&instance);
            }
          gt_end_expanded1:;
            instance.flags.expanded ^= 1;
            return 1;
          }
          instance.flags.expanded ^= 1;
        }

        auto user_r = instance[element_nr].ep.mouse_button_cb(d);
        if (user_r != 0) {
          return user_r;
        }
        return instance[element_nr].iic() == true;
      };
      instance[element_nr].ep = ep;
      
      if (instance.flags.expanded == true || instance.Usage() == 1) {
        instance[element_nr].enable_draw(instance, instance.Usage() - 1);
      }

      return element_nr;
    }
    void open(const open_properties_t& op) {
      *(menu_nr_t*)this = gloco->dropdown.menu_list.NewNodeLast();
      auto& instance = gloco->dropdown.menu_list[*this];

      instance = op;
      menu_data_t menu_data;
      menu_data.flags.titleable = op.titleable;
      menu_data.flags.expanded = menu_data.flags.titleable == false;  

      instance = menu_data;

      if (menu_data.flags.titleable == true) {
        element_properties_t p;
        p.text = op.title;
        add(p);
      }
    }
    element_properties_t get_element_properties() {
      auto& selected_id = gloco->dropdown.menu_list[*this].selected_id;
      if (selected_id.iic()) {
        fan::throw_error("invalid selection");
      }
      return gloco->dropdown.menu_list[*this][selected_id].ep;
    }
    // node pointer can change
    dropdown_t::element_t& get_active_element() {
      return gloco->dropdown.menu_list[*this][gloco->dropdown.menu_list[*this].GetNodeFirst()];
    }
    void set_selected(element_list_nr_t element) {
      auto& instance = gloco->dropdown.menu_list[*this];
      auto title_id = instance.GetNodeFirst();
      instance.selected_id = element;
      instance[title_id].set_text(instance[element].ep.text);
    }
  };

}dropdown;