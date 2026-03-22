module;

#if defined(FAN_PHYSICS_2D)
  #include <fan/utility.h>
#endif

export module fan.graphics.physics_subsystem;

#if defined(FAN_PHYSICS_2D)
  export import fan.physics.b2_integration;
  export import fan.physics.common_context;
#endif

export namespace fan::graphics {
  struct physics_subsystem_t {
    void init();
    void destroy();
    void update(f32_t dt);
    void set_enabled(bool flag);

#if defined(FAN_PHYSICS_2D)
    fan::physics::context_t context{{}};
    bool is_updating = false;
#endif
  };
}