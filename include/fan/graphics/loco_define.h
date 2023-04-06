#pragma once

//fan_create_get_set_define_extra(fan::vec3, position,  
//  if (get_position().z != data.z) {
//    global_loco.get_loco()->shape_set_depth(*this, data.z);
//  }
//, ;);
//fan_create_set_define_custom(fan::vec2, position, 
//  global_loco.get_loco()shape_set_position(*this, fan::vec3(data, get_position().z));
//);
//fan_create_get_set_define(fan::vec2, size);
//fan_create_get_set_define(fan::color, color);
//fan_create_get_set_define(f32_t, angle);
//fan_create_get_set_define(fan::string, text);
//fan_create_get_set_define(fan::vec2, rotation_point);
//fan_create_get_set_define(f32_t, font_size);
//
//fan_create_set_define(f32_t, depth);
//                   
//fan_create_set_define(loco_t::camera_list_NodeReference_t, camera);
//fan_create_set_define(fan::graphics::viewport_list_NodeReference_t, viewport);

//

fan_build_get_set_define(fan::vec3, position);
fan_build_get_set_define(fan::vec2, size);
fan_build_get_set_define(fan::color, color);
fan_build_get_set_define(f32_t, angle);
fan_build_get_set_define(fan::vec2, rotation_point);

make_global_function_define(erase,
  if constexpr (has_erase_v<shape_t, loco_t::cid_t*>) {
    (*shape)->erase(cid);
  },
  loco_t::cid_t* cid
);

fan_build_get_set_generic_define(f32_t, font_size);
fan_build_get_set_generic_define(loco_t::camera_list_NodeReference_t, camera);
fan_build_get_set_generic_define(fan::graphics::viewport_list_NodeReference_t, viewport);

fan_build_get_set_generic_define(fan::string, text);

fan_has_function_concept(sb_set_depth);

make_global_function_define(set_depth,
  if constexpr (has_set_depth_v<shape_t, loco_t::cid_t*, f32_t>) { 
    (*shape)->set_depth(cid, data); 
  } 
  else if constexpr (has_sb_set_depth_v<shape_t, loco_t::cid_t*, f32_t>) { 
    (*shape)->sb_set_depth(cid, data); 
  }, 
  loco_t::cid_t* cid, 
  const auto& data 
);


#define make_shape_id_define(name) \
  loco_t::name ## _id_t::name ## _id_t(const properties_t& p) { \
    global_loco.get_loco()->name.push_back(*this, *(loco_t::name ## _t::properties_t*)&p); \
  } \
   \
  loco_t::name ## _id_t& loco_t::name ## _id_t::operator[](const properties_t& p) { \
    global_loco.get_loco()->name.push_back(*this, *(loco_t::name ## _t::properties_t*)&p); \
    return *this; \
  } \
   \
  loco_t::name ## _id_t::~name##_id_t() { \
    global_loco.get_loco()->name.erase(*this); \
  }


#if defined(loco_vfi)
make_shape_id_define(vfi);
#endif


#undef loco_access

#undef loco_rectangle_vi_t