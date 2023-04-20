#ifndef fgm_shape_loco_name
  #define fgm_shape_loco_name fgm_shape_name
#endif

#ifndef fgm_shape_manual_properties
  using properties_t = loco_t:: CONCAT(fgm_shape_loco_name, _t) ::properties_t;
#endif

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix instances
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeData fgm_shape_instance_data
  #define BLL_set_Link 1
  #define BLL_set_StoreFormat 1
  #include _FAN_PATH(BLL/BLL.h)

using nr_t = instances_NodeReference_t;
instances_t instances;

auto shape_builder_push_back() {
  return instances.NewNodeLast();
}

fgm_t* get_fgm() {
  return OFFSETLESS(this, fgm_t, fgm_shape_name);
}

void release() {
  get_fgm()->move_offset = 0;
  get_fgm()->action_flag &= ~action::move;
}

#ifndef fgm_no_gui_properties
  #define fgm_make_clear_f(user_f) \
    void clear() { \
      close_properties(); \
      auto it = instances.GetNodeFirst(); \
   \
      while (it != instances.dst) { \
        user_f \
        it = it.Next(&instances); \
      } \
      instances.Clear(); \
    }
#else
  #define fgm_make_clear_f(user_f) \
    void clear() { \
      auto it = instances.GetNodeFirst(); \
   \
      while (it != instances.dst) { \
        user_f \
        it = it.Next(&instances); \
      } \
      instances.Clear(); \
    }
#endif

#ifndef fgm_shape_non_moveable_or_resizeable
void create_shape_move_resize(auto shape_nr, const loco_t::mouse_move_data_t& ii_d) {
  if (ii_d.flag->ignore_move_focus_check == false) {
    return;
  }
  if (!(get_fgm()->action_flag & action::move)) {
    return;
  }

  if (holding_special_key) {
    fan::vec3 ps = get_position(&instances[shape_nr]);
    fan::vec2 rs = get_size(&instances[shape_nr]);

    static constexpr f32_t minimum_rectangle_size = 0.03;
    static constexpr fan::vec2i multiplier[] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };

    rs += (ii_d.position - get_fgm()->resize_offset) * multiplier[get_fgm()->resize_side] / 2;

    if (rs.x == minimum_rectangle_size && rs.y == minimum_rectangle_size) {
      get_fgm()->resize_offset = ii_d.position;
    }

    bool ret = 0;
    if (rs.y < minimum_rectangle_size) {
      rs.y = minimum_rectangle_size;
      if (!(rs.x < minimum_rectangle_size)) {
        ps.x += (ii_d.position.x - get_fgm()->resize_offset.x) / 2;
        get_fgm()->resize_offset.x = ii_d.position.x;
      }
      ret = 1;
    }
    if (rs.x < minimum_rectangle_size) {
      rs.x = minimum_rectangle_size;
      if (!(rs.y < minimum_rectangle_size)) {
        ps.y += (ii_d.position.y - get_fgm()->resize_offset.y) / 2;
        get_fgm()->resize_offset.y = ii_d.position.y;
      }
      ret = 1;
    }

    if (rs != minimum_rectangle_size) {
      ps += (ii_d.position - get_fgm()->resize_offset) / 2;
    }
    if (rs.x == minimum_rectangle_size && rs.y == minimum_rectangle_size) {
      ps = get_position(&instances[shape_nr]);
    }

    set_size(&instances[shape_nr], rs);
    set_position(&instances[shape_nr], ps);

    if (ret) {
      return;
    }

    get_fgm()->resize_offset = ii_d.position;
    get_fgm()->move_offset = ps - fan::vec3(ii_d.position, 0);
    get_fgm()->fgm_shape_name.open_properties(shape_nr);
    return;
  }

  fan::vec3 ps = get_position(&instances[shape_nr]);
  fan::vec3 p;
  p.x = ii_d.position.x + get_fgm()->move_offset.x;
  p.y = ii_d.position.y + get_fgm()->move_offset.y;
  p.z = ps.z;
  set_position(&instances[shape_nr], p);

  get_fgm()->fgm_shape_name.open_properties(shape_nr);
}
#endif

#ifndef fgm_shape_non_moveable_or_resizeable
auto keyboard_cb(auto shape_nr) {
  return [this, shape_nr](const loco_t::keyboard_data_t& kd) -> int {

    switch (kd.key) {
      case fan::key_delete: {
        if (kd.keyboard_state != fan::keyboard_state::press) {
          return 0;
        }
        switch (kd.keyboard_state) {
          case fan::keyboard_state::press: {
            get_fgm()->erase_shape(this, shape_nr);
            get_fgm()->invalidate_focus();
            break;
          }
        }
        break;
      }
      case fan::key_c: {
        holding_special_key = kd.keyboard_state == fan::keyboard_state::release ? 0 : 1;
        break;
      }
      case fan::key_left:
      case fan::key_right:
      case fan::key_up:
      case fan::key_down:
      {
        if (kd.keyboard_state == fan::keyboard_state::release) {
          return 0;
        }
        if (kd.key == fan::key_left) get_fgm()->move_shape(this, &instances[shape_nr], fan::vec2(-1, 0));
        if (kd.key == fan::key_right) get_fgm()->move_shape(this, &instances[shape_nr], fan::vec2(1, 0));
        if (kd.key == fan::key_up) get_fgm()->move_shape(this, &instances[shape_nr], fan::vec2(0, -1));
        if (kd.key == fan::key_down) get_fgm()->move_shape(this, &instances[shape_nr], fan::vec2(0, 1));
        open_properties(shape_nr);
        break;
      }
    }
    return 0;
  };
}
#endif