module;

#if defined(FAN_PHYSICS_2D)
#endif

export module fan.graphics.physics_subsystem;

import std;

#if defined(FAN_PHYSICS_2D)
  import fan.types;
  export import fan.physics.b2_integration;
  import fan.physics.common_context;

  export namespace fan::graphics {
    struct physics_subsystem_t {
      void init();
      void destroy();
      void update(f32_t dt);
      void draw();
      void set_enabled(bool flag);

      fan::physics::context_t context{{}};
      bool is_updating = false;
    };
  }
#endif