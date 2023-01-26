#pragma once

#if defined(loco_rectangle)

loco_t::rectangle_id_t::rectangle_id_t(properties_t p) {
  (loco_access)->rectangle.push_back(&cid, *(loco_t::rectangle_t::properties_t*)&p);
}

loco_t::rectangle_id_t::~rectangle_id_t() {
  (loco_access)->rectangle.erase(&cid);
}

#endif

#if defined(loco_sprite)
loco_t::sprite_id_t::sprite_id_t(properties_t p) {
  sprite_t::properties_t p2;
  memcpy(&p2, &p, sizeof(sprite_t::properties_t));
  //*(sprite_t::properties_t*)&p2 = *(loco_t::sprite_t::properties_t*)&p;
  p2.image = p.image;
  (loco_access)->sprite.push_back(&cid, *(loco_t::sprite_t::properties_t*)&p);
}

loco_t::sprite_id_t::~sprite_id_t() {
  (loco_access)->sprite.erase(&cid);
}
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