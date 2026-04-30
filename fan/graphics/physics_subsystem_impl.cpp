module;

#if defined(FAN_PHYSICS_2D)
#endif

module fan.graphics.physics_subsystem;

import std;

#if defined(FAN_PHYSICS_2D)

namespace fan::graphics {
  void physics_subsystem_t::init() {
  }

  void physics_subsystem_t::destroy() {
  }

  void physics_subsystem_t::update(f32_t dt) {
    if (!is_updating) { return; }
    context.step(dt);
  }

  void physics_subsystem_t::draw() {
    if (!(context.debug.enabled && context.debug_draw_cb)) {
      return;
    }
    context.debug_draw_cb();
  }

  void physics_subsystem_t::set_enabled(bool flag) {
    is_updating = flag;
  }
}
#endif