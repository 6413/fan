export module fan;

export import fan.print;
export import fan.types.vector;
export import fan.types.color;
export import fan.types.matrix;

export import fan.random;
export import fan.io.file;
export import fan.io.directory;

export import fan.window;
export import fan.window.input;
export import fan.graphics.common_context;
export import fan.graphics.loco;
export import fan.graphics;

export import fan.event;

#define fan_audio
#if defined(fan_audio)
  export import fan.audio;
#endif
#if defined(fan_gui)
export import fan.graphics.gui;
#endif
#if defined(fan_physics)
export import fan.physics.b2_integration;
export import fan.graphics.physics_shapes;
#endif