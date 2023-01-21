#infdef fgm_shape_loco_name
  #define fgm_shape_loco_name fgm_shape_name
#endif

using properties_t = loco_t:: CONCAT(fgm_shape_loco_name, _t) ::properties_t;

loco_t* get_loco() {
  return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, fgm_shape_name), stage_maker_t, fgm))->get_loco();
}
pile_t* get_pile() {
  return OFFSETLESS(get_loco(), pile_t, loco_var_name);
}

#define shape_builder_push_back \
  instance.resize(instance.size() + 1); \
  uint32_t i = instance.size() - 1; \
  instance[i] = new instance_t; \
  instance[i]->shape = shapes::fgm_shape_loco_name;

struct instance_t {
  fgm_shape_instance_data
};

std::vector<instance_t*> instance;