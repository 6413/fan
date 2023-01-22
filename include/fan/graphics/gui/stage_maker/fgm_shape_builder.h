#ifndef fgm_shape_loco_name
  #define fgm_shape_loco_name fgm_shape_name
#endif

#ifndef fgm_shape_manual_properties
  using properties_t = loco_t:: CONCAT(fgm_shape_loco_name, _t) ::properties_t;
#endif

loco_t* get_loco() {
  return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, fgm_shape_name), stage_maker_t, fgm))->get_loco();
}
pile_t* get_pile() {
  return OFFSETLESS(get_loco(), pile_t, loco_var_name);
}

struct instance_t {
  fgm_shape_instance_data
};

#if !defined(fgm_dont_init_shape)
  #define shape_builder_push_back \
    instances.resize(instances.size() + 1); \
    uint32_t i = instances.size() - 1; \
    instances[i] = new instance_t; \
    instances[i]->shape = shapes::fgm_shape_loco_name;
#else
  #define shape_builder_push_back \
      instances.resize(instances.size() + 1); \
      uint32_t i = instances.size() - 1; \
      instances[i] = new instance_t;
#endif

std::vector<instance_t*> instances;

void release() {
  pile_t* pile = get_pile();
  pile->stage_maker.fgm.move_offset = 0;
  pile->stage_maker.fgm.action_flag &= ~action::move;
}

#ifndef fgm_no_gui_properties
  #define fgm_make_clear_f(user_f) \
    void clear() { \
      close_properties(); \
      auto* pile = get_pile(); \
      for (auto& it : instances) { \
        user_f \
        delete it; \
      }\
      instances.clear(); \
    }
#else
  #define fgm_make_clear_f(user_f) \
    void clear() { \
      auto* pile = get_pile(); \
      for (auto& it : instances) { \
        user_f \
        delete it; \
      }\
      instances.clear(); \
    }
#endif

#define create_shape_move_resize \
	if (ii_d.flag->ignore_move_focus_check == false) {\
    return 0;\
  }\
  pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);\
  if (!(pile->stage_maker.fgm.action_flag & action::move)) {\
    return 0;\
  }\
\
  if (holding_special_key) {\
    fan::vec3 ps = get_position(instance);\
    fan::vec2 rs = get_size(instance);\
\
    static constexpr f32_t minimum_rectangle_size = 0.03;\
    static constexpr fan::vec2i multiplier[] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };\
\
    rs += (ii_d.position - pile->stage_maker.fgm.resize_offset) * multiplier[pile->stage_maker.fgm.resize_side] / 2;\
\
    if (rs.x == minimum_rectangle_size && rs.y == minimum_rectangle_size) {\
      pile->stage_maker.fgm.resize_offset = ii_d.position;\
    }\
\
    bool ret = 0;\
    if (rs.y < minimum_rectangle_size) {\
      rs.y = minimum_rectangle_size;\
      if (!(rs.x < minimum_rectangle_size)) {\
        ps.x += (ii_d.position.x - pile->stage_maker.fgm.resize_offset.x) / 2;\
        pile->stage_maker.fgm.resize_offset.x = ii_d.position.x;\
      }\
      ret = 1;\
    }\
    if (rs.x < minimum_rectangle_size) {\
      rs.x = minimum_rectangle_size;\
      if (!(rs.y < minimum_rectangle_size)) {\
        ps.y += (ii_d.position.y - pile->stage_maker.fgm.resize_offset.y) / 2;\
        pile->stage_maker.fgm.resize_offset.y = ii_d.position.y;\
      }\
      ret = 1;\
    }\
\
    if (rs != minimum_rectangle_size) {\
      ps += (ii_d.position - pile->stage_maker.fgm.resize_offset) / 2;\
    }\
    if (rs.x == minimum_rectangle_size && rs.y == minimum_rectangle_size) {\
      ps = get_position(instance);\
    }\
\
    set_size(instance, rs);\
    set_position(instance, ps);\
\
    if (ret) {\
      return 0;\
    }\
\
    pile->stage_maker.fgm.resize_offset = ii_d.position;\
    pile->stage_maker.fgm.move_offset = ps - fan::vec3(ii_d.position, 0);\
    pile->stage_maker.fgm.fgm_shape_name.open_properties(instance);\
    return 0;\
  }\
\
  fan::vec3 ps = get_position(instance);\
  fan::vec3 p;\
  p.x = ii_d.position.x + pile->stage_maker.fgm.move_offset.x;\
  p.y = ii_d.position.y + pile->stage_maker.fgm.move_offset.y;\
  p.z = ps.z;\
  set_position(instance, p);\
\
  pile->stage_maker.fgm.fgm_shape_name.open_properties(instance);