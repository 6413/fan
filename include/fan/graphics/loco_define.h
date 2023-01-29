#pragma once

#define make_shape_id_define(name) \
  loco_t::name ## _id_t::name ## _id_t(const properties_t& p) { \
    (loco_access)->name.push_back(*this, *(loco_t::name ## _t::properties_t*)&p); \
  } \
   \
  loco_t::name ## _id_t& loco_t::name ## _id_t::operator[](const properties_t& p) { \
    (loco_access)->name.push_back(*this, *(loco_t::name ## _t::properties_t*)&p); \
    return *this; \
  } \
   \
  loco_t::name ## _id_t::~name##_id_t() { \
    (loco_access)->name.erase(*this); \
  }

#if defined(loco_rectangle)
  make_shape_id_define(rectangle);
#endif

#if defined(loco_sprite)
  loco_t::sprite_id_t::sprite_id_t(const properties_t& p) {
    auto& p2 = *(loco_t::sprite_t::properties_t*)&p;
    if (p.ti) {
      p2.load_tp(p.ti);
    }
    (loco_access)->sprite.push_back(*this, p2);
  }
  loco_t::sprite_id_t& loco_t::sprite_id_t::operator[](const properties_t& p) {
    auto& p2 = *(loco_t::sprite_t::properties_t*)&p;
    if (p.ti) {
      p2.load_tp(p.ti);
    }
    (loco_access)->sprite.push_back(*this, p2);
    return *this;
  }
  loco_t::sprite_id_t::~sprite_id_t() {
    (loco_access)->sprite.erase(*this);
  }

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

#if defined(loco_text_box)
  make_shape_id_define(text_box);
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