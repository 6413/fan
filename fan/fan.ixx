export module fan;

import std;

export import fan.types;
export import fan.types.vector;
export import fan.types.color;
export import fan.types.matrix;
export import fan.types.fstring;
#if defined(FAN_JSON)
  export import fan.types.json;
#endif
export import fan.math;
export import fan.random;
export import fan.time;
export import fan.print;
export import fan.print.error;
export import fan.utility;
export import fan.memory;
//export import fan.types.masterpiece;
export import fan.io.file;
export import fan.io.directory;

export import fan.window.input;
export import fan.window.input_common;
export import fan.window;

export import fan.graphics.image_load;

export import fan.graphics.common_context;

export import fan.graphics.loco;

#if defined(FAN_2D)
  export import fan.graphics.shapes.types;
  export import fan.graphics.shapes;
  export import fan.graphics;
#endif
#if defined(FAN_GUI)
  export import fan.console;
  export import fan.file_dialog;
  export import fan.graphics.gui.types;
  export import fan.graphics.gui.base;
  export import fan.graphics.gui;
  export import fan.graphics.gui.text_logger;
  export import fan.graphics.gui.settings_menu;
#endif
export import fan.texture_pack.tp0;

#if defined(FAN_NETWORK)
  export import fan.network;
#endif

export import fan.event.types;
export import fan.event;

#if defined(FAN_AUDIO)
  export import fan.audio;
#endif
#if defined(FAN_PHYSICS_2D)
  export import fan.physics.collision.rectangle;
  export import fan.physics.collision.circle;
  export import fan.physics.types;
  export import fan.physics.b2_integration;
  export import fan.graphics.physics_shapes;
#endif

export import fan.spatial;
export import fan.process;

export import fan.noise;
export import fan.pathfind;

//export import fan.types.slot_map;

export import fan.types.compile_time_string;

export import fan.formatter;

export import fan.ecs;

#if defined(FAN_GUI)
export import fan.graphics.gui.tilemap_editor.renderer;
export import fan.graphics.gameplay;
export import fan.graphics.gui.inventory_hotbar;
export import fan.graphics.gameplay.items;
export import fan.graphics.gui.gameplay.equipment;
export import fan.graphics.gui.input;
export import fan.graphics.gui.inventory;
#endif

export import fan.crypto;