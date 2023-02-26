#ifndef fgm_shape_loco_name
  #define fgm_shape_loco_name fgm_shape_name
#endif

#ifndef fgm_shape_manual_properties
  using properties_t = loco_t:: CONCAT(fgm_shape_loco_name, _t) ::properties_t;
#endif

struct instance_t {
  fgm_shape_instance_data
};

void shape_builder_push_back() {
  #if !defined(fgm_dont_init_shape)
    instances.resize(instances.size() + 1);
    uint32_t i = instances.size() - 1;
    instances[i] = new instance_t;
    instances[i]->shape = loco_t::shape_type_t::fgm_shape_loco_name;
  #else
    instances.resize(instances.size() + 1);
    uint32_t i = instances.size() - 1;
    instances[i] = new instance_t;
  #endif
}

std::vector<instance_t*> instances;

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
      for (auto& it : instances) { \
        user_f \
        delete it; \
      }\
      instances.clear(); \
    }
#else
  #define fgm_make_clear_f(user_f) \
    void clear() { \
      for (auto& it : instances) { \
        user_f \
        delete it; \
      }\
      instances.clear(); \
    }
#endif

#ifndef fgm_shape_non_moveable_or_resizeable
void create_shape_move_resize(instance_t* instance, const loco_t::mouse_move_data_t& ii_d) {
  if (ii_d.flag->ignore_move_focus_check == false) {
    return;
  }
  if (!(get_fgm()->action_flag & action::move)) {
    return;
  }

  if (holding_special_key) {
    fan::vec3 ps = get_position(instance);
    fan::vec2 rs = get_size(instance);

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
      ps = get_position(instance);
    }

    set_size(instance, rs);
    set_position(instance, ps);

    if (ret) {
      return;
    }

    get_fgm()->resize_offset = ii_d.position;
    get_fgm()->move_offset = ps - fan::vec3(ii_d.position, 0);
    get_fgm()->fgm_shape_name.open_properties(instance);
    return;
  }

  fan::vec3 ps = get_position(instance);
  fan::vec3 p;
  p.x = ii_d.position.x + get_fgm()->move_offset.x;
  p.y = ii_d.position.y + get_fgm()->move_offset.y;
  p.z = ps.z;
  set_position(instance, p);

  get_fgm()->fgm_shape_name.open_properties(instance);
}
#endif

#ifndef fgm_shape_non_moveable_or_resizeable
auto keyboard_cb(instance_t* instance) {
  return [this, instance](const loco_t::keyboard_data_t& kd) -> int {

    switch (kd.key) {
      case fan::key_delete: {
        if (kd.keyboard_state != fan::keyboard_state::press) {
          return 0;
        }
        switch (kd.keyboard_state) {
          case fan::keyboard_state::press: {
            get_fgm()->erase_shape(this, instance);
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
        if (kd.key == fan::key_left) get_fgm()->move_shape(this, instance, fan::vec2(-1, 0));
        if (kd.key == fan::key_right) get_fgm()->move_shape(this, instance, fan::vec2(1, 0));
        if (kd.key == fan::key_up) get_fgm()->move_shape(this, instance, fan::vec2(0, -1));
        if (kd.key == fan::key_down) get_fgm()->move_shape(this, instance, fan::vec2(0, 1));
        open_properties(instance);
        break;
      }
    }
    return 0;
  };
}
#endif