#pragma once

#define make_shape_id_define(name) \
  loco_t::name ## _id_t::name ## _id_t(const properties_t& p) { \
    (loco_access)->name.push_back(&cid, *(loco_t::name ## _t::properties_t*)&p); \
  } \
   \
  loco_t::name ## _id_t::~name##_id_t() { \
    (loco_access)->name.erase(&cid); \
  }

#if defined(loco_rectangle)
  make_shape_id_define(rectangle);
#endif

#if defined(loco_sprite)
  make_shape_id_define(sprite);
#endif

#if defined(loco_button)
  make_shape_id_define(button);
#endif

#if defined(loco_letter)
  make_shape_id_define(letter);
#endif

#if defined(loco_text)
  make_shape_id_define(text);
#endif

#if defined(loco_vfi)
  make_shape_id_define(vfi);
#endif

#undef loco_access

#undef loco_rectangle_vi_t

#undef loco_rectangle
#undef loco_sprite
#undef loco_letter
#undef loco_text
#undef loco_text_box
#undef loco_button
#undef loco_wboit