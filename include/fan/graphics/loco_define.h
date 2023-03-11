#pragma once

loco_t::cid_nr_t::~cid_nr_t() {
  invalidate();
}

void loco_t::cid_nr_t::init() {
  *(base_t*)this = (loco_access)->cid_list.NewNodeLast();
}

bool loco_t::cid_nr_t::is_invalid() {
  return cid_list_inric(*this);
}

void loco_t::cid_nr_t::invalidate() {
  if (is_invalid()) {
    return;
  }
  (loco_access)->cid_list.unlrec(*this);
  *(base_t*)this = (loco_access)->cid_list.gnric();
}

loco_t::id_t::id_t(const auto& properties) {
  cid.init();
  (loco_access)->push_shape(*this, properties);
}

inline loco_t::id_t::id_t(const id_t& id) {
  (loco_access)->shape_get_properties(*(id_t*)&id, [&](const auto& properties) {
    cid.init();
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
      cid.init();
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
  if (get_position().z != position.z) {
    (loco_access)->shape_set_depth(*this, position.z);
  }
  (loco_access)->shape_set_position(*this, position);
}
fan_create_id_definition(void, set_size, const fan::vec2& size) {
  (loco_access)->shape_set_size(*this, size);
}
fan_create_id_definition(fan::vec2, get_size) {
  return (loco_access)->shape_get_size(*this);
}

fan_create_id_definition(void, set_color, const fan::color& size) {
  (loco_access)->shape_set_color(*this, size);
}
fan_create_id_definition(fan::color, get_color) {
  return (loco_access)->shape_get_color(*this);
}

#undef loco_access

#undef loco_rectangle_vi_t