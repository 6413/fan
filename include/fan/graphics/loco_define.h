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


#if defined(loco_vfi)
make_shape_id_define(vfi);
#endif

loco_t::cid_nr_t::~cid_nr_t() {
  invalidate();
}

loco_t::cid_nr_t::cid_nr_t(const cid_nr_t& nr) {
  init();
}

loco_t::cid_nr_t::cid_nr_t(cid_nr_t&& nr) {
  NRI = nr.NRI;
  nr.invalidate();
}

loco_t::cid_nr_t& loco_t::cid_nr_t::operator=(const cid_nr_t& id) {
  if (this != &id) {
    init();
  }
  return *this;
}

loco_t::cid_nr_t& loco_t::cid_nr_t::operator=(cid_nr_t&& id) {
  if (this != &id) {
    if (!is_invalid()) {
      invalidate();
    }
    NRI = id.NRI;

    id.invalidate();
  }
  return *this;
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

inline loco_t::id_t::id_t(const id_t& id) : cid(id.cid) {
  (loco_access)->shape_get_properties(*(id_t*)&id, [&](const auto& properties) {
    cid.init();
    (loco_access)->push_shape(*this, properties);
  });
}
inline loco_t::id_t::id_t(id_t&& id) : cid(std::move(id.cid)) {
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
      (loco_access)->push_shape(*this, properties);
    });
  }
  return *this;
}

loco_t::id_t& loco_t::id_t::operator=(id_t&& id) {
  if (this != &id) {
    if (!cid.is_invalid()) {
      erase();
    }
    cid = std::move(id.cid);

    id.cid.invalidate();
  }
  return *this;
}


void loco_t::id_t::erase() {
  if (cid.is_invalid()) {
    return;
  }
  (loco_access)->shape_erase(*this);
  cid.invalidate();
}

loco_t::id_t::operator fan::opengl::cid_t *(){
  return &(loco_access)->cid_list[cid].cid;
}

loco_t* loco_t::id_t::get_loco() {
  return loco_access;
}

#undef loco_access

#undef loco_rectangle_vi_t