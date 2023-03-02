#pragma once

loco_t::instance_t::instance_t(const auto& properties) {
  initialized = true;
  (loco_access)->push_shape(&cid, properties);
}

void loco_t::instance_t::set_position(const fan::vec3& position) {
  #if fan_debug >= fan_debug_medium
  if (!initialized) {
    fan::throw_error("using uninitialized shape", typeid(decltype(*this)).name());
  }
  #endif
  (loco_access)->set_position(&cid, position);
}

#undef make_address_wrapper