module;

#if defined(FAN_PHYSICS_2D)
  #include <fan/utility.h>
#endif

module fan.graphics.physics_subsystem;

namespace fan::graphics {
  void physics_subsystem_t::init() {
  }

  void physics_subsystem_t::destroy() {
  }

  void physics_subsystem_t::update(f32_t dt) {
#if defined(FAN_PHYSICS_2D)
    if (!is_updating) { return; }
    context.debug_draw_cb();
    context.step(dt);
#endif
  }

  void physics_subsystem_t::set_enabled(bool flag) {
#if defined(FAN_PHYSICS_2D)
    is_updating = flag;
#endif
  }
}