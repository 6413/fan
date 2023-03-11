#pragma once

loco_t::cid_nr_t::cid_nr_t() {
  *(base_t*)this = (loco_access)->cid_list.NewNodeLast();
}
loco_t::cid_nr_t::~cid_nr_t() {
  invalidate();
}
bool loco_t::cid_nr_t::is_invalid() {
  return (loco_access)->cid_list.inric(*this);
}

void loco_t::cid_nr_t::invalidate() {
  if (is_invalid()) {
    return;
  }
  (loco_access)->cid_list.unlrec(*this);
  *(base_t*)this = (loco_access)->cid_list.gnric();
}

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

#if defined(loco_light)
  make_shape_id_define(light);
#endif

loco_t::id_t::id_t(const auto& properties) {
  (loco_access)->push_shape(*this, properties);
}

inline loco_t::id_t::id_t(const id_t& id) {
  (loco_access)->shape_get_properties(*(id_t*)&id, [&](const auto& properties) {
    (loco_access)->sb_push_back(*this, properties);
  });
}
inline loco_t::id_t::id_t(id_t&& id) {
  cid = id.cid;
  id.cid.invalidate();
}

loco_t::id_t::~id_t() {
  if (cid.is_invalid()) {
    return;
  }
  erase();
}

loco_t::id_t& loco_t::id_t::operator=(const id_t& id) {
  if (this != &id) {
    (loco_access)->shape_get_properties(*(id_t*)&id, [&](const auto& properties) {
      (loco_access)->sb_push_back(*this, properties);
    });
  }
  return *this;
}

loco_t::id_t& loco_t::id_t::operator=(id_t&& id) {
  if (this != &id) {
    if (!cid.is_invalid()) {
      erase();
    }
    cid = id.cid;

    id.cid.invalidate();
  }
  return *this;
}


void loco_t::id_t::erase() {
  (loco_access)->shape_erase(*this);
}

loco_t::id_t::operator fan::opengl::cid_t *(){
  return &(loco_access)->cid_list[cid].cid;
}

fan_create_id_definition(fan::vec3, get_position) {
  return (loco_access)->shape_get_position(*this);
}
fan_create_id_definition(void, set_position, const fan::vec3& position) {
  (loco_access)->shape_set_position(*this, position);
}
fan_create_id_definition(void, set_size, const fan::vec2& size) {
  (loco_access)->shape_set_size(*this, size);
}
fan_create_id_definition(fan::vec2, get_size) {
  return (loco_access)->shape_get_size(*this);
}


#undef loco_access

#undef loco_rectangle_vi_t