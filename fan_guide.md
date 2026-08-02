module system:
  // impl file does NOT re-import what the interface already imports
  // impl file is part of the same module — inherits all interface imports
  // only import in impl what is NOT in the interface:
  //   fan.graphics.physics_shapes   — physics shape sync, impl only
  //   fan.graphics.gui.settings_menu — settings cast, impl only
  // For module files (.ixx / .cpp), standard includes MUST go inside the global module fragment:
  module;
  #include <vector>
  export module my_module; // or module my_module;
  import fan;

preprocessor style:
  // #if blocks indent contents 2 spaces at file/namespace scope:
  #if defined(FAN_AUDIO)
    #include <fan/utility.h>
    export import fan.audio;
  #endif
  // inside function bodies, #if contents follow function indentation (no extra indent)

error handling in subsystems:
  fan::throw_error_impl(const char*)  // always available via utility.h preamble include
  fan::throw_error(std::string)       // requires import fan.print — avoid in subsystems
  // prefer throw_error_impl in subsystem impls to avoid extra imports

subsystem module pattern:
  // new subsystem = fan/graphics/X_subsystem.ixx + X_subsystem_impl.cpp
  // interface exports subsystem struct + re-exports user-facing deps
  // impl has no re-imports (inherits from interface)
  // subsystem owns its data, init()/destroy() handle lifetime
  // #if defined(FAN_FEATURE) guards live inside subsystem, not at call site
  // completed: audio_subsystem, physics_subsystem, input_subsystem

  // OFFSETLESS with nested members:
  auto* loco = OFFSETLESS(self, loco_t, gui.console);
  // recovers loco_t* from fan::console_t* inside gui_state_t

---

fan library (import fan; — github.com/6413/fan):
  // C++23/26 2D OpenGL framework, use import fan; at top, #includes above it
  // using namespace fan::graphics; is common

BLL (Base Linked List) & Storage:
  // Engine's core high-performance container (similar to sparse set/handle map)
  // node_ref_t (NRI) is a stable handle {index, generation}
  // bll_t holds objects, safe for deletions during iteration
  auto nr = bll.NewNodeLast();
  bll[nr] = my_object;
  bll.unlrec(nr);           // safe removal
  
  // iteration:
  for (auto nr = bll.GetNodeFirst(); nr != bll.dst; nr = nr.Next(&bll)) {
    auto& obj = bll[nr];
  }
  // C++20 range iteration:
  for (auto& [nr, obj] : bll | fan::enumerate) { ... }

stable IDs:
  // do not store pointers or vector indices across frames
  // use bll_t::nr_t (NRI) or int/uint64_t IDs mapped via registry
  // if unit is deleted, NRIs pointing to it become invalid (safe check)
  if (bll.is_valid(target_nri)) { ... }

engine / main loop:
  engine_t engine;                          // or engine_t e([&]{ ... }); for inline loop
  engine.loop([&] { ... });                 // main loop, called every frame
  engine.get_delta_time()                   // f64_t& — returns window.m_delta_time by ref
  engine.time                               // f64_t seconds since start (not delta)
  engine.update_physics(true);              // enable box2d physics
  engine.viewport_get_size()               // fan::vec2, size of default viewport
  engine.viewport_get_size(viewport)       // size of specific viewport

engine properties / lifecycle:
  loco_t::properties_t props;
  props.window_size       = {1280, 720};
  props.window_position   = {0, 0};
  props.samples           = 4;              // MSAA
  props.renderer          = fan::window_t::renderer_t::opengl;
  props.render_shapes_top = false;          // shapes render below GUI when false
  engine_t engine{props};

  engine.set_target_fps(60)                 // 0 = unlimited (idle mode)
  engine.set_vsync(bool)
  engine.set_window_name(str)
  engine.set_window_icon(image_t)
  engine.toggle_console()                   // requires FAN_GUI
  engine.should_close()                     // bool
  engine.show_fps = true                    // shows FPS overlay
  engine.allow_docking                      // bool, ImGui docking when true or Ctrl held
  engine.get_clear_color()                  // fan::color& background clear color
  engine.get_render_shapes_top()            // bool& shapes render above GUI
  // single_queue, m_pre_draw, m_post_draw, draw_end_cb are push-only vectors
  // no removal API — only push lambdas that capture objects with guaranteed lifetime
  // for removable callbacks use add_update_callback() which returns a handle

loco_t internals:
  // internal access (inside loco_impl.cpp):
  renderer_state.clear_color / renderer_state.lighting / renderer_state.force_line_draw
  renderer_state.render_shapes_top / renderer_state.reload_renderer_to
  gui.console / gui.text_logger / gui.gui_draw_cb
  gui.show_fps / gui.allow_docking / gui.enable_overlay / gui.settings_menu / gui.render_settings_menu
  timing.vsync / timing.shape_draw_timer / timing.frame_timer
  timing.target_frame_time / timing.accumulated_time
  input.input_action

  // user-facing API (gloco()->...):
  gloco()->get_clear_color()                // fan::color&
  gloco()->get_lighting()                   // fan::graphics::lighting_t&
  gloco()->get_render_shapes_top()          // bool&
  gloco()->get_vsync()                      // bool&
  gloco()->get_delta_time()                 // f64_t& (window.m_delta_time)
  gloco()->get_console()                    // fan::console_t&
  gloco()->get_input_action()               // fan::window::input_action_t&
  gloco()->get_show_fps()                   // bool&
  gloco()->get_allow_docking()              // bool&
  gloco()->get_enable_overlay()             // bool&
  gloco()->get_settings_menu()              // fan::graphics::gui::settings_menu_t*&
  gloco()->get_render_settings_menu()       // bool&
  gloco()->get_physics_context()            // fan::physics::context_t&

  // subsystem inits:
  audio.init() / audio.destroy()
  physics.set_enabled(flag)
  input.init(window)

  // ctx() pointer updates in bind_global_context():
  ctx.input_action = &input.input_action;
  ctx.lighting     = &renderer_state.lighting;
  ctx.console      = &gui.console;
  ctx.text_logger  = &gui.text_logger;

  // renderer dispatch macros:
  renderer_set(func, ...)                   // void return, dispatches to gl/vk context
  renderer_get(type, gl, vk, func, ...)     // non-void return
  renderer_call(func)                       // calls gl.func() or vk.func()
  render_context_call_raw(gl_expr, vk_expr) // raw expression dispatch

pile (stage system):
  pile.get_delta_time()                     // use this for dt, not a manual clock
  pile.viewport_get_size()                  // matches world coordinate space
  open()   // called on stage load and on restart — always reinitialize all state here
  close()  // called on stage unload — shape_t destructors handle GPU cleanup automatically
  
  // stage commands:
  pile.stage_open<level1_t>(data_ptr)
  pile.stage_close<level1_t>()
  pile.stage_restart<level1_t>()
  pile.stage_change<level1_t, level2_t>(data_ptr) // crossfades to next stage

  // full pile/level structure:
  struct my_pile_t : engine_t, fan::frame_task_t<my_pile_t> {
    struct level1_t : fan::stage_t<level1_t> {
      void open(void*) { /* init level */ }
      void close() { /* cleanup — shape_t destructors handle GPU */ }
      void update() { /* per-frame logic */ }
    };
    my_pile_t() { stage_open<level1_t>(); }
    void update() { if (is_key_clicked(fan::key_r)) stage_restart<level1_t>(); }
  } pile;
  int main() { pile.loop(); }

world coords:
  // origin (0,0) is top-left, x right, y down
  // world coords are 0 to viewport_size (not -vp/2 to +vp/2)
  // bottom of screen is vs.y, no -s/2 offset needed
  // camera_set_position offset: cam_pos - vs / 2.f to center on a point
  // use viewport_get_size() not window::get_size() for world-space calculations
  // use screen_to_world() when converting mouse position to world coords

render_view_t (usually not needed, default exists):
  fan::graphics::render_view_t rv;
  rv.create();                              // allocates camera + viewport
  rv.set(ortho_x, ortho_y, vp_pos, vp_size, window_size)
  rv.remove()                              // free resources
  // most shapes accept .render_view = &view or omit for default
  fan::graphics::get_orthographic_render_view()  // default 2D view
  fan::graphics::get_perspective_render_view()   // default 3D view
  fan::graphics::camera_set_position(view.camera, pos)

  fan::graphics::add_render_view()
  fan::graphics::add_render_view(ortho_x, ortho_y, vp_pos, vp_size)
  fan::graphics::viewport_set(pos, size)
  fan::graphics::viewport_set(nr, pos, size)
  fan::graphics::viewport_get_size(nr)     // fan::vec2
  fan::graphics::inside(render_view, screen_pos)         // bool
  fan::graphics::is_mouse_inside(render_view)            // bool
  fan::graphics::screen_to_world(screen_pos, render_view)  // fan::vec2
  fan::graphics::world_to_screen(pos, render_view)         // fan::vec2

shapes (all use designated init syntax):
  fan::graphics::rectangle_t r{{
    .position = fan::vec3(x, y, z),   // z = depth layer
    .size     = fan::vec2(w, h),      // half-size
    .color    = fan::colors::red,
    .render_view = &view,             // optional, omit for default
    .blending = true,                 // for transparency
  }};
  // same pattern: circle_t, sprite_t, line_t, capsule_t, light_t, gradient_t, shadow_t

  // short-form constructors (no designated init needed):
  sprite_t(position, size, image)
  sprite_t(position, size, fan::color)
  sprite_t(position, size, {color, color, ...})           // gradient from color list
  sprite_t(position, size, vector<uint8_t> data, vec2ui tex_size)
  rectangle_t(position, size, color)
  circle_t(position, radius, color)
  line_t(src, dst, color, thickness)
  unlit_sprite_t(...)                // same overloads as sprite_t, no lighting applied

  shape.set_position(fan::vec3(...))
  shape.get_position()               // fan::vec3
  shape.set_size(fan::vec2(...))     
  shape.get_size()                   // fan::vec2
  shape.set_color(fan::color(...))
  shape.set_angle(fan::vec3(0, 0, radians))
  fan::graphics::shape_t             // type-erased handle, holds any shape

  // VERIFIED — polygon_t is UNTEXTURED: fan::graphics::vertex_t (used by
  // polygon_t::properties_t.vertices) is {fan::vec3 position; fan::color color;} only —
  // no uv/tc field, no .image member on polygon_t::properties_t. For a mesh made of
  // many cells that needs a texture (tile atlas etc.), polygon_t cannot do it; use one
  // sprite_t per cell (has .tc_position/.tc_size for atlas frame selection) or extend
  // the engine's vertex_t. Don't assume polygon_t can be textured by analogy with
  // sprite_t/rectangle_t.

  // VERIFIED — canonical properties_t field declaration order (designated
  // initializers must be listed in this order or it won't compile / silently
  // reorders if you get it wrong — always check the header, don't guess):
  //   sprite_t: position, parallax_factor, size, rotation_point, color, angle,
  //             flags, tc_position, tc_size, seed, texture_pack_unique_id,
  //             sprite_sheet_data, image
  //   rectangle_t: position, size, color, outline_color, angle
  //   (most shapes end with #include <fan/graphics/base_props.inl>, which appends:
  //   visible, camera, viewport, draw_mode, vertex_count, blending — in that order.
  //   VERIFIED: camera/viewport there default through
  //   get_orthographic_render_view()/ctx().orthographic_render_view, which is also
  //   null pre-engine — so this is a third distinct field hitting the same
  //   global-init-order hazard above; virtually any shape's default properties_t
  //   is unsafe to materialize before engine_t exists, not just position.)

shape_t lifetime:
  // default-constructed shape_t is invalid — operator bool returns false, safe to check
  // moved-from shape_t is invalid (NRI set to sentinel) — do not call methods
  // copying shape_t creates a full new GPU-side shape — not free, avoid in hot loops
  // calling get_position/get_size/any method on invalid shape_t crashes in iic()
  // sprite_t inherits shape_t — same rules apply
  if (!shape) { /* invalid */ }  // safe validity check
  // erase_if on vector<unit_t> destroys sprite_t destructors which may corrupt renderer batch
  // use stable IDs instead of vector indices for cross-frame references
  // dead units: mark dead=true, set_color({0,0,0,0}) at death site, skip in update, clear on restart

shaper / rendering:
  // shapes with identical render state (same texture, camera, viewport) batch into one draw call
  // mixing cameras/viewports/textures across many shapes = many draw calls — minimize state changes
  set_static()   // opts into culling system — use for non-moving shapes
  set_dynamic()  // forces per-frame GPU upload — use for moving shapes

shape_from_json / shapes_from_json:
  // results ARE cached internally after first call — safe to call every open()
  // cache is keyed by path string — same path = same cached GPU shapes copied
  // json path is relative to executable, not source file

immediate draw helpers (no variable needed, auto-managed lifetime):
  fan::graphics::rectangle(position, size, color)
  fan::graphics::sprite(position, size, image)
  fan::graphics::circle(position, radius, color)
  fan::graphics::line(src, dst, color, thickness)
  // these return shape_t& into an internal immediate_render_list — do not store the reference

sprite_t / sprite sheets:
  // sprite_t is a shape_t with sprite sheet support — all shape_t methods apply
  play_sprite_sheet("name")         // loops continuously
  play_sprite_sheet_once("name")    // plays once then stops
  is_sprite_sheet_finished()        // returns true once per completion — poll every frame
  set_random_sprite_sheet_frame()   // stagger spawned units so they don't animate in sync
  // sprite sheet names must match names defined in the JSON exactly
  // character2d_t JSON can define generic animation states without hardcoded animation names:
  // "animation_states": [{
  //   "name": "airborne", "animation": "jump", "condition": "airborne",
  //   "frames": [0, 1, 2, 3, 4, 5, 6], "playback": "hold"
  // }, {
  //   "name": "landing", "animation": "jump", "condition": "landed",
  //   "frames": [7], "fps": 12, "playback": "once"
  // }]
  // frames refer to source animation frame indices. Conditions: always, airborne,
  // landed, grounded, moving, idle, rising, falling. Playback: continuous, loop,
  // hold, hold_last, once. Call character2d_t::update_animations() each frame.

particles:
  fan::graphics::particle_system_t ps{max_particles};
  ps.spawn([&]() -> shape_props_t {    // shape_fn — called each spawn
    return { .position = ..., .size = ..., .color = ... };
  }, [](sim_data_t& s, f32_t dt) {     // sim_fn — called each frame per particle
    s.position.y += s.velocity.y * dt;  // sim_data_t has position, velocity, life, color
    s.life -= dt;
    if (s.life <= 0) s.dead = true;
  });
  ps.update(dt)                         // call each frame
  // pre-allocated slots: particle_system_t<100> uses stack array, no heap
  fan::graphics::trail_particle_t trail;
  trail.update(position, dt)

move_towards:
  // signature: move_towards(target, speed, image_orientation)
  // image_orientation {-1.f, 0.f} locks Y axis — use for pure horizontal movement
  // image_orientation {-1.f, 1.f} allows Y drift
  // flips sprite automatically based on direction sign vs image_orientation sign
  // image_orientation.y == 0 zeroes tc_size.y — never pass y=0 if moving in Y
  // fix in move_towards: guard the Y sign line so it only fires when image_orientation.y != 0
  sprite.move_towards(target_pos, speed, fan::vec2(-1.f, 0.f)); // horizontal only

shader_shape_t:
  // sprite with custom GLSL shader
  fan::graphics::shader_shape_t s{{
    .position = ..., .size = ..., .shader = my_shader, .image = img
  }};
  fan::graphics::shader_update_fragment(shape_type, fragment_glsl_string)  // hot-reload

camera helpers:
  fan::graphics::camera_set_target(camera, target, move_speed)  // smooth follow
  fan::graphics::camera_look_at(target, move_speed)             // default camera
  // move_speed == 0 teleports instantly, >0 lerps at speed * delta_time
  engine.camera_move_to(shape)          // snap camera to shape position
  engine.camera_move_to_smooth(shape)   // lerp camera toward shape
  engine.camera_set_target(target, speed)
  // camera_set_target / camera_look_at must be called ONCE — not inside engine.loop.
  // each call pushes a new frame update; repeated calls leak since there's no RAII cleanup.
  engine.screen_to_ndc(screen_pos)      // fan::vec2
  engine.ndc_to_screen(ndc_pos)         // fan::vec2
  engine.convert_mouse_to_ndc()         // fan::vec2
  engine.convert_mouse_to_ray(proj, view)  // fan::ray3_t, uses perspective camera

culling:
  // culling is enabled by default, shapes auto-register via set_static()
  // set_static() is the default — every shape is static+culled unless set_dynamic() is called
  engine.set_culling_enabled(bool)
  engine.set_cull_padding(fan::vec2)    // expand culling frustum
  engine.rebuild_static_culling()       // call after bulk static shape changes
  engine.get_culling_stats(visible, culled)  // uint32_t& each
  engine.visualize_culling()            // draws frustum outline for debug

debug helpers:
  fan::graphics::aabb(min, max, thickness)
  fan::graphics::aabb(shape, depth, color, thickness)

update / draw callbacks:
  auto handle = engine.add_update_callback([](void*){ /* runs every frame */ });
  engine.remove_update_callback(handle);
  engine.add_update_callback_front(cb)     // runs before user cb
  engine.single_queue.push_back([&]{ ... });   // single-shot, runs once next frame
  engine.m_pre_draw.push_back([&]{ ... });
  engine.m_post_draw.push_back([&]{ ... });
  engine.draw_end_cb.push_back([&]{ ... });

  auto nr = fan::graphics::get_gui_draw_cbs().NewNodeLast();
  fan::graphics::get_gui_draw_cbs()[nr] = [&]() { /* persistent per-frame GUI */ };
  // remove: fan::graphics::get_gui_draw_cbs().unlrec(nr);

  auto nr = fan::graphics::ctx().update_callback->NewNodeLast();
  (*fan::graphics::ctx().update_callback)[nr] = [](void*) { /* persistent per-frame logic */ };
  // remove: fan::graphics::ctx().update_callback->unlrec(nr);

physics (box2d wrapper, enable with engine.update_physics(true)):
  fan::physics::entity_t body = engine.get_physics_context().create_box(pos, half_size, angle, body_type, shape_props);
  engine.get_physics_context().create_rectangle(...)
  engine.get_physics_context().create_circle(pos, radius, angle, body_type, shape_props)
  engine.get_physics_context().create_sensor_rectangle(pos, half_size)
  fan::physics::create_sensor_rectangle(pos, half_size)  // free function version

  body_type: fan::physics::body_type_e::static_body / dynamic_body / kinematic_body

  shape_properties_t:
    {.fixed_rotation = true, .is_sensor = true, .friction = 0.f,
     .filter = {.categoryBits = 0x1, .maskBits = 0x2}}

  entity methods:
    body.get_position()                // fan::vec2, pixel coords
    body.set_physics_position(pos)
    body.get_linear_velocity()         // fan::vec2
    body.set_linear_velocity(v)
    body.apply_linear_impulse_center(v)
    body.apply_force_center(v)
    body.set_gravity_scale(f32_t)
    body.set_mass(f32_t)
    body.set_restitution(f32_t)
    body.set_friction(f32_t)
    body.get_aabb()                    // fan::physics::aabb_t {min, max}
    body.destroy() / body.is_valid()

  sensors:
    fan::physics::is_on_sensor(body, sensor)  // bool, poll each physics step
    // sensor must be static, test body dynamic with matching filter

  physics step callback (RAII):
    fan::physics::step_callback_nr_t cb = fan::physics::add_physics_step_callback([&]() { ... });

  combined visual+physics shapes (auto-sync position):
    fan::graphics::physics::rectangle_t
    fan::graphics::physics::circle_t
    fan::graphics::physics::capsule_t
    fan::graphics::physics::sprite_t         // fuses a textured sprite + its Box2D body
    fan::graphics::physics::character2d_t  // full character controller

  // physics::sprite_t (fan/graphics/physics_shapes.ixx):
  // inherits base_shape_t : shape_t, fan::physics::entity_t — one object owns BOTH the
  // GPU-side sprite and the physics body, so per-tile terrain (each solid cell needs a
  // static collider) can skip a separate fan::physics::entity_t collider vector entirely.
  // .body_type defaults to static_body — set explicitly for dynamic tiles/props.
  // TRADEOFF vs hand-rolled greedy-merged-rect colliders: one physics::sprite_t per cell
  // means one Box2D static body per solid cell, not one merged rect per contiguous run —
  // far more bodies for dense terrain. Fine for moderate cell counts; for very dense/large
  // worlds the merged-rect-collider + separate polygon/sprite-mesh approach still wins on
  // body count, at the cost of losing fused sprite+body ownership (extra bookkeeping).

  // physics sprite (body + image):
  fan::graphics::physics::sprite_t ps
  ps.open(parent_body, {
    .position = fan::vec3(x, y, z), .size = fan::vec2(w, h),
    .image = {"path.png", image_presets::pixel_art()}
  }, [](physics::sprite_t& s) { /* on interact */ })
  // trigger (sensor that fires callback on contact):
  trigger_t trigger
  trigger.open(parent_body, { .position = ..., .size = ..., .image = ... },
    [](physics::sprite_t& s) { /* pickup/door/etc logic */ })
  // character controller shorthand:
  physics::character2d_t body = physics::character_capsule({
    .center0 = {0.f, -24.f}, .center1 = {0.f, 24.f}, .radius = 12
  })
  body.enable_default_movement(300.f, 32.f)   // speed, jump_force

  collision filtering:
    static constexpr uint64_t category_a = 0x1;
    static constexpr uint64_t category_b = 0x2;
    // body a: {.categoryBits = category_a, .maskBits = category_b}
    // body b: {.categoryBits = category_b, .maskBits = category_a}

  // ECS + physics: sync entity positions from bodies each frame
  registry.each([&](uint32_t id, c_pos& p, c_phys& phys) {
    p.v = phys.body.get_position()
  })

  // moving platforms:
  fan::graphics::physics::elevator_t elevator
  elevator.create_elevator_box()                  // creates the lift body
  elevator.set_route({pos1, pos2, ...})           // waypoint path
  elevator.set_speed(f32_t)                       // movement speed
  elevator.update()                               // call each physics step

fan::graphics::spatial (import fan; includes it):
  // three structs, templated on id_t (use int for unit IDs)
  spatial::static_grid_t<id_t>     // non-moving objects (terrain, bases)
  spatial::dynamic_grid_t<id_t>    // moving objects (units, projectiles)
  spatial::registry_t<id_t>        // tracks membership + cached AABBs

  // init (cell_size ~2× largest object radius; grid_size * cell_size covers world)
  spatial::static_grid_init(static_grid,  world_min, cell_size, grid_size);
  spatial::dynamic_grid_init(dynamic_grid, world_min, cell_size, grid_size);
  spatial::reset(static_grid, dynamic_grid, registry);   // full clear on restart

  // register or update an object (static or dynamic)
  spatial::upsert_object(registry, static_grid, dynamic_grid,
    id, aabb, spatial::movement_static or spatial::movement_dynamic);

  // remove object and clean internal tables
  spatial::remove_and_clean(registry, static_grid, dynamic_grid, id);

  // queries
  spatial::query_radius(dynamic_grid, center, radius, [](id_t id){ ... });
  spatial::query_aabb(dynamic_grid, aabb, [](id_t id){ ... });
  spatial::query_nearest(dynamic_grid, center, radius,
    [](id_t id){ return predicate; });

  // area query (static + dynamic)
  spatial::query_area(static_grid, dynamic_grid, view_min, view_max,
    [](id_t id){ ... });
 		
pathfinding (import fan.pathfind; — included via import fan;):
  fan::pathfind::generator gen;
  gen.set_world_size(world_size)                   // fan::vec2i in cells
  gen.set_diagonal_movement(bool)                  // allow diagonal movement
  gen.add_collision(fan::vec2i{x, y})              // mark cell as wall
  gen.remove_collision(fan::vec2i{x, y})
  gen.clear_collisions()
  fan::pathfind::coordinate_list path = gen.find_path(src, dst)
  // returns empty vector if no path found
  // coordinate_list = std::vector<fan::vec2i>
  // tilemap integration:
  tilemap.add_wall(cell, gen)                      // mark tilemap tile as wall
  tilemap.remove_wall(cell, gen)
  auto path = tilemap.find_path(gen, src, dst, heuristic, allow_diagonal)
  // heuristic: fan::pathfind::heuristic::manhattan / euclidean / octagonal

unit/game patterns:
  // locked target IDs: store target_enemy_id as stable int ID, not vector index
  // find_unit_idx(id) linear scan — fine for <100 units, use unordered_map for more
  // separation_force: push units apart with radius-based repulsion, apply as 0.3x nudge to target
  // pending_units vector: defer spawn to end of frame to avoid mid-loop invalidation
  // fighting state: use is_sprite_sheet_finished() to gate damage application

gui (fan::graphics::gui:: or via using):
  #define gui fan::graphics::gui  // common shorthand or using namespace fan::graphics; using outside library

  // windows:
  gui::fullscreen_window("##id")            // RAII fullscreen window, still accepts input
  gui::centered_window("##id", size)        // RAII window centered on screen
  gui::hud("id")                            // fullscreen overlay, ignores input — not for interactive panels
  gui::hud_interactive("id")                // fullscreen overlay that accepts input
  // hud() sets window_flags_no_inputs — buttons inside hud() won't fire
  // use hud_interactive() or a plain window() with no_title_bar flags for interactive overlays
  gui::begin("Title") / gui::end()
  gui::begin_child("id", size) / gui::end_child()
  gui::set_next_window_pos(pos) / gui::set_next_window_size(size)
  gui::set_next_window_bg_alpha(0.f)
  gui::push_style_var(gui::style_var_window_border_size, 0.f) / gui::pop_style_var()

  // layout:
  gui::same_line() / gui::new_line()
  gui::separator() / gui::spacing(px) / gui::dummy(size)
  gui::indent(f32_t) / gui::unindent(f32_t)
  gui::set_cursor_pos(pos) / gui::get_cursor_pos()
  gui::set_cursor_screen_pos(pos) / gui::get_cursor_screen_pos()
  gui::get_content_region_avail()           // fan::vec2
  gui::get_window_size() / gui::get_window_pos()  // fan::vec2
  gui::fill_width() / gui::fill_width_except("btn_label")
  gui::set_next_item_width(f32_t)
  gui::get_style().WindowPadding / .ItemSpacing / .CellPadding / .FramePadding  // camelCase members
  gui::col_window_bg / gui::style_var_cell_padding                               // enums snake_case

  // text:
  gui::text("str") / gui::text(color, "str")
  gui::text_sized("str", size)
  gui::text_centered(str, color)
  gui::text_centered_outlined(str, color)
  gui::text_centered_outlined_big(str, font_size, color, outline_color)
  gui::text_box_at("str", pos)
  gui::text_unformatted(str) / gui::text_wrapped(str, color)
  gui::calc_text_size(str)                  // fan::vec2, useful for manual centering

  // font:
  gui::font_scope_t fs(48.f)               // push font size for scope, pops on destruction
    // available sizes: 4,5,6,7,8,9,10,11,12,14,16,18,20,22,24,28,32,36,48,60,72
  gui::push_font(font) / gui::pop_font()
  gui::get_font(size, bold)
  gui::get_font_size()                      // returns actual size (fonts loaded at size*2)
  gui::get_text_line_height()               // returns doubled value — use get_font_size() for centering

  // buttons:
  gui::button("label") / gui::button("label", size)   // returns bool
  gui::button_fill("label")                            // full width button
  gui::button_grid({"a","b",...}, cols, size, cb, font_size)
  gui::button_row({"a","b",...}, size, font_size, cb)
  gui::calc_button_width("label")           // f32_t
  gui::invisible_button("id", size)
  gui::begin_disabled() / gui::end_disabled()

  // game ui helpers:
  gui::healthbar(value, max, size, fill, bg)
  gui::healthbar_labeled(label, value, max, size, label_color, fill, bg)
    // use get_font_size() not get_text_line_height() for vertical centering
    // fonts loaded at size*2, get_text_line_height returns doubled value
  gui::gold_text(amount, color)
  gui::disabled_button_row(labels*, enabled*, count, size, on_click)
  gui::disabled_button_row(span<const str_view_t>, span<const bool>, size, on_click)

  // inputs:
  gui::input_text("##id", &std::string)
  gui::input_text_multiline("##id", &str, size, flags)
  gui::input_int("label", &int)
  gui::input_float("label", &f32_t)
  gui::drag("label", &value)               // works for any numeric type
  gui::drag("label", &value, speed, min, max)
  gui::checkbox("label", &bool)
  gui::color_edit4("##id", &fan::color)
  gui::combo("##id", &int_index, count, [](int i) -> const char* { return "label"; })
  gui::selectable("label", selected, flags, size)

  // images:
  gui::image(texture, size)
  gui::image_button("id", img, size)

  // tables:
  gui::begin_table("id", cols, flags) / gui::end_table()
  gui::table_next_row() / gui::table_next_column()
  gui::table_setup_column("label", flags)
  gui::collapsing_header("label", nullptr, flags)

  // popups:
  gui::begin_popup_modal(id) / gui::end_popup() / gui::open_popup(id) / gui::close_current_popup()
  gui::begin_popup("id") / gui::end_popup()

  // query:
  gui::is_item_hovered() / gui::is_item_active() / gui::is_item_clicked()
  gui::is_window_hovered() / gui::is_window_focused()
  gui::want_io()                            // true if gui is consuming input
  gui::set_keyboard_focus_here()
  gui::default_logger()                    // log_dispatcher_t for console output

gui gameplay (inventory / hotbar / drag-drop):
  // requires #define FAN_GUI
  using namespace fan::graphics::gui
  inventory_t inv{cols, rows};          // grid-based inventory
  inv.set_slot(pos, item);              // place item at col,row
  inv.get_slot(pos)                     // returns slot data
  inv.on_click = [&](fan::vec2i pos, mouse_button btn) { ... }
  inv.on_drop = [&](fan::vec2i from, fan::vec2i to) { ... }
  inv.render()                          // call each frame in GUI context
  hotbar_t bar{num_slots};              // horizontal item bar
  bar.set_slot(index, item)
  bar.get_selected()                    // currently selected index
  bar.render()
  drag_drop_t dd
  dd.begin(source_id, data_ptr)         // start drag
  dd.target(target_id, [&](void* data) { ... })  // drop target callback
  dd.end()
  equipment_t equip{slot_names...}
  equip.set_slot("head", item)
  equip.render()

  // str_view_t notes:
  // str_view_t accepts const std::string& safely
  // temporary std::string not allowed — store dynamically built labels as named variables
  // str_view_t inherits std::string_view — has_subscript_and_size concept must use is_base_of_v:
  //   && (!std::is_base_of_v<std::string_view, std::remove_cvref_t<T>>)
  // otherwise str_view_t prints character by character with spaces in format_args

input:
  fan::window::is_key_clicked(fan::key_*)
  fan::window::is_key_down(fan::key_*)
  fan::window::is_key_released(fan::key_*)
  fan::window::is_mouse_clicked()           // left by default
  fan::window::is_mouse_clicked(fan::mouse_right)
  fan::window::is_mouse_down()
  fan::window::is_mouse_released()
  fan::window::get_mouse_drag()             // fan::vec2 drag delta
  fan::window::get_char_pressed()           // uint32_t, 0 if none
  fan::window::get_size()                   // fan::vec2 window size
  fan::window::get_input_vector()           // fan::vec2 WASD/arrow normalized
  fan::window::get_input_vector("Forward", "Back", "Left", "Right")
  fan::window::is_gamepad_button_down(key)
  fan::window::get_current_gamepad_axis(key)  // fan::vec2
  fan::window::get_mouse_position()         // raw screen space

  // named actions:
  fan::window::add_input_action({fan::key_a, fan::key_left}, "move_left")
  fan::window::is_input_clicked("Move Left")
  fan::window::is_input_down("Move Left")
  fan::window::is_input_released("Move Left")
  fan::window::exists("move_left")          // bool

  // predefined action names (use instead of raw strings):
  fan::actions::move_forward / move_back / move_left / move_right / move_up
  fan::actions::light_attack / block_attack
  fan::actions::toggle_settings / toggle_console
  fan::actions::toggle_debug_physics / recompile_shaders

  // mouse / world position:
  fan::graphics::get_mouse_position()                           // screen space
  fan::graphics::get_mouse_position(render_view)                // world space for that view

global init order (graphics objects — crash source):
  // any object that touches the GL/Vulkan context or ctx().window (image_t, shape_t,
  // sprite_t, rectangle_t, circle_t, polygon_t, render_view_t, etc.) must NOT be
  // constructed with a REAL (non-default) initializer as a global/namespace-scope
  // variable. dynamic initializers for globals run before main(), before engine_t
  // exists — ctx().window / ctx().image_list / ctx().camera_list etc. are all still
  // nullptr at that point (ctx() itself is a Meyer's singleton, safe to call anytime;
  // it's the members inside it that aren't populated yet). Constructing anything that
  // dereferences those members crashes with an access violation reading near 0x0
  // inside the image/shape constructor.
  //
  // VERIFIED: image_t's own no-arg default constructor (image_t()) is SAFE at global
  // scope — it only copies ctx().default_texture (a default-constructed, zeroed
  // image_nr_t pre-engine, not a null deref). It's the constructors that take a path,
  // color, or explicit properties that crash, because those call into
  // ctx()->image_load_path_props(...) etc., which dereference ctx().window /
  // ctx().image_list. Same logic applies to shape properties_t default member
  // initializers — e.g. sprite_t::properties_t::position defaults to
  // ctx().window->get_size()/2, which dereferences a null ctx().window if a sprite_t
  // with real properties is constructed at global scope, even with otherwise-trivial
  // values.
  //
  // WRONG:
  fan::graphics::image_t tile_atlas{"tiles.webp", image_presets::pixel_art()};  // crash at startup
  // RIGHT — declare empty (default ctor only), assign after engine_t is constructed:
  fan::graphics::image_t tile_atlas;
  int main() {
    fan::graphics::engine_t engine;
    tile_atlas = fan::graphics::image_t{"tiles.webp", image_presets::pixel_art()};
  }
  // same rule for any shape_t-derived global (sprite_t, rectangle_t, polygon_t, ...):
  // default-construct at namespace scope, assign/construct for real inside main() or
  // inside a callback that runs after the loop starts. Containers of shapes
  // (std::vector<sprite_t>, std::unordered_map<K, sprite_t>) are fine to declare
  // empty at global scope — only the individual element construction must wait.

  // TEARDOWN ORDER — the same rule applies in reverse at shutdown, and is easy to miss:
  // engine_t (loco_t) is almost always a LOCAL variable inside main() ("engine_t
  // engine;"), so it is destroyed when main() returns — BEFORE any namespace-scope
  // globals are destroyed (globals are torn down after main() returns, in reverse
  // declaration order). VERIFIED: loco_t::~loco_t() calls destroy(), which tears down
  // the image list, camera list, and render context. A global image_t/sprite_t/
  // container-of-shapes whose destructor runs after that (i.e. any global still
  // holding GPU-side shapes when main() exits) will have its destructor touch an
  // already-destroyed ctx(). In practice this mostly matters for globals that outlive
  // engine's scope; keep graphics-owning globals's *lifetime* bounded by main() where
  // possible — e.g. wrap world state in a struct constructed inside main() after
  // engine_t, instead of namespace-scope containers, if you're not certain they'll be
  // emptied before engine goes out of scope. Emptying a container of shapes it's
  // in main() before engine_t goes out of scope (or declaring it after engine_t inside
  // main so it's destroyed first, per reverse construction order) sidesteps this.

  // THREAD SAFETY — NOT VERIFIED. Guide does not currently know whether shape/image
  // construction (sprite_t{...}, image_t{...}, etc.) is safe to call off the main/GL
  // thread (e.g. from fan::event::thread_create or an async task). Treat as
  // main-thread-only until confirmed; do not construct shapes/images from worker
  // threads without checking the renderer dispatch (renderer_set/renderer_get) for
  // thread affinity first.

images:
  // RAII wrapper (preferred over raw image_nr_t):
  fan::graphics::image_t img{"path.png"};
  fan::graphics::image_t img{"path.png", image_presets::pixel_art()};
  fan::graphics::image_t img{fan::colors::red};
  fan::graphics::image_t img{info};
  fan::graphics::image_t img{colors*, vec2ui size};
  fan::graphics::image_t img{span<const fan::color>, vec2ui size};  // pixel art preset auto
  fan::graphics::image_t img{vec2 size, channels, props};           // blank texture
  img.valid()                    // bool — check before use
  img.get_size()                 // fan::vec2
  img.get_path()                 // std::string
  img.reload(path)               // hot-reload from disk
  img.update(data, channels)     // push new pixel data to GPU
  img.get_pixel_data(format)     // std::vector<uint8_t> — reads back from GPU
  img.unload()
  img.get_handle()               // uint64_t, raw texture handle for ImGui::Image etc

  // free functions:
  fan::graphics::image_load(path)
  fan::graphics::image_load(path, props)
  fan::graphics::image_load(colors*, vec2ui size)
  fan::graphics::image_load(span<const fan::color>, size)
  fan::graphics::image_load(image::info_t, props)
  fan::graphics::image_load_pixel_art(path)
  fan::graphics::image_load_smooth(path)
  fan::graphics::image_create(fan::color)
  fan::graphics::image_unload(nr)
  fan::graphics::is_image_valid(nr)        // bool
  engine.image_load("path.webp")
  engine.image_reload(img, info, props)
  engine.create_noise_image(size, seed)
  engine.create_noise_image(size, precomputed_data)

  // presets:
  fan::graphics::image_presets::pixel_art()         // nearest filter, clamp_to_border
  fan::graphics::image_presets::pixel_art_repeat()  // nearest filter, repeat
  fan::graphics::image_presets::smooth()            // linear filter, clamp_to_border
  fan::graphics::image_presets::mipmapped()         // linear_mipmap_linear

texture packs:
  // pre-compile sprite sheets into .ftp binary for fast loading
  texture_pack.open_compiled("pack.ftp")  // call once at startup
  // images referenced in tilemaps/JSON refer to texture pack by name

lighting:
  fan::graphics::get_lighting().ambient              // fan::vec3, current ambient light
  fan::graphics::get_lighting().set_target(vec3, duration_s)  // smooth transition
  fan::graphics::get_lighting().is_near_target()     // bool

  // point light attached to a physics body:
  light_t light{body, radius, fan::colors::white}
  // auto-follows body position each frame

post-processing / screen effects:
  gloco()->enable_bloom = true
  gloco()->set_post_process("bloom_strength", f32_t)
  gloco()->set_post_process("bloom_threshold", f32_t)
  gloco()->set_post_process("bloom_knee", f32_t)
  gloco()->set_post_process("bloom_filter_radius", f32_t)
  gloco()->set_post_process("bloom_tint", fan::vec3)
  gloco()->set_post_process("gamma", f32_t)
  gloco()->set_post_process("contrast", f32_t)
  gloco()->set_post_process("exposure", f32_t)
  // defaults: strength=0.0445, threshold=1.0, knee=0.1, radius=0.1
  // gamma=1.0, contrast=1.0, exposure=1.0
  // raw access:
  *gloco()->get_bloom_threshold_ptr() = 0.8f
  *gloco()->get_bloom_filter_radius_ptr() = 0.05f

window (fan::window_t):
  window.set_size(fan::vec2i)
  window.set_position(fan::vec2)             // move window, clamps to title bar on Windows
  window.set_display_mode(int)               // windowed/borderless/fullscreen, use mode:: enum
  window.get_size()                          // fan::vec2i
  window.get_position()                      // fan::vec2
  window.get_current_monitor()               // GLFWmonitor* of monitor window is mostly on
  window.get_current_monitor_resolution()    // fan::vec2
  window.get_primary_monitor_resolution()    // fan::vec2
  window.display_mode                        // current mode
  window.set_display_mode(mode::windowed)    // NOTE: windowed calls set_windowed() which recenters
                                             // skip calling if already windowed to avoid recenter
  fan::window_t::mode::windowed              // check what value this is — may be 0 or 1 depending
                                             // on how display_mode is stored (render uses i+1)
  fan::window_t::resolutions[]               // predefined resolution list
  fan::window_t::resolution_labels[]         // matching string labels

settings / config restore order:
  // correct order to restore window state on startup:
  // 1. set_size first (before set_display_mode for windowed)
  // 2. set_display_mode (skip if already windowed to avoid recenter)
  // 3. set_position last (overrides whatever set_display_mode did)
  // register resize/move callbacks AFTER restoring — avoids saving bogus positions during init
  // move callback fires with restore position (not snapped position) on Windows snap
  // set_position now clamps y to title bar minimum internally — no need to handle in settings

file io (import fan.io.file;):
  // Namespace shortcut or absolute paths: fan::io::file::
  // Functions accept path_t template concept (std::string_view, std::string, std::filesystem::path)
  
  // Checking existence, sizes, and metadata:
  fan::io::file::exists("path.dat");                // bool
  fan::io::file::file_size("path.dat");             // std::uint64_t, returns 0 if missing
  fan::io::file::extension("folder/file.png");      // returns ".png"
  fan::io::file::strip_extension("file.png");       // returns "file"
  fan::io::file::rename("old.txt", "new.txt");      // bool
  fan::io::file::get_exe_path();                    // std::string path to executable folder
  
  // Relative path helpers:
  fan::io::file::relative_path(path, base);         // std::filesystem::path relative to base
  fan::io::file::find_relative_path("file.json");   // std::filesystem::path, runs multi-directory fallback lookups
  
  // Read workflows (return true on failure, false on success):
  std::string content;
  bool fail = fan::io::file::read("path.txt", &content);
  
  // Read overloads:
  std::string data = fan::io::file::read("path.txt", &success_bool);
  bool fail_bytes  = fan::io::file::read_bytes("path.bin", buffer_ptr, byte_size);
  std::vector<std::string> lines = fan::io::file::read_line("path.txt"); // throws if missing
  
  // Template type vector reads:
  std::vector<std::uint32_t> data = fan::io::file::read<std::uint32_t>("path.bin");
  
  // Write workflows (return true on success, false on failure):
  // fs_mode modes: std::ios_base::binary, std::ios_base::app, std::ios_base::trunc
  bool ok = fan::io::file::write("path.txt", data_string, mode);
  bool written = fan::io::file::try_write("path.txt", data_string, mode); // skips if file exists
  
  // Classic fstream wrapper state:
  fan::io::file::fstream fs("path.dat");
  fs.open("path.dat");
  fs.read(&content_str);
  fs.write(&content_str);
  
  // Serializer stream functions (uses old-school file_t* under the hood):
  fan::io::file::file_t* f;
  fan::io::file::open(&f, "path.dat", props);
  fan::io::file::write_to_file(f, my_string_or_vector_or_pod);
  fan::io::file::close(f);

async / coroutines:
  fan::event::task_t              // fire-and-forget, starts immediately (suspend_never)
  fan::event::task_suspend_t      // starts suspended, must be co_await'd
  fan::event::task_resume_t       // alias for task_t
  fan::event::task_value_t<T>     // coroutine returning T, starts suspended
  fan::event::task_value_resume_t<T>  // coroutine returning T, starts immediately

  fan::event::task_t my_coro() { co_await fan::graphics::co_next_frame(); }
  auto task = my_coro();          // starts immediately, self-managing lifetime

  task.valid()                    // bool — coroutine still running
  task.request_stop()             // sets cancelled flag, coroutine exits at next co_await
  task.join()                     // block until done (sync, avoid on main thread)
  task.stop_and_join() / task.destroy()
  // assigning a new task_t to an existing one cancels the old one automatically

  // cancellation inside coroutine:
  // co_await checks cancelled flag on every suspension point automatically
  co_await fan::graphics::co_next_frame();
  co_await fan::co_sleep(1000);   // throws task_cancelled_exception if cancelled

  co_await fan::event::timer_t(ms);             // suspend for duration
  fan::event::after(ms, []{ ... });             // call lambda once after delay
  fan::event::every(ms, []{ return false; });   // repeat until lambda returns true
  fan::event::when_all(task1, task2, ...);      // wait for all tasks to complete
  fan::event::sleep(ms);                        // BLOCKING sleep — avoid on main thread
  fan::event::now();                            // uint64_t milliseconds since loop start
  fan::event::thread_create([&] { ... })

  // signal between coroutines:
  fan::event::signal_awaitable_t<int> sig;
  sig.signal(42);
  int val = co_await sig;
  // void version: sig.signal(); co_await sig;
  // Coroutines capturing `this` or references will dangle if the owner is destroyed while suspended.
  // Always store task_t inside the owning struct so it auto-cancels on destruction.

async file I/O:
  auto content = co_await fan::io::file::async_read(path);          // std::string
  co_await fan::io::file::async_write(path, data);
  co_await fan::io::file::async_read_cb(path, [](std::string chunk){ ... });
  auto size = co_await fan::io::file::async_size(path);             // intptr_t

  fan::io::file::async_read_t reader;
  co_await reader.open(path);  auto chunk = co_await reader.read();  co_await reader.close();

  fan::io::file::async_write_t writer;
  co_await writer.open(path);  co_await writer.write(data);  co_await writer.close();

  // low-level:
  int fd = co_await fan::io::file::async_open(path, fan::fs_in);
  co_await fan::io::file::async_close(fd);
  // file open flags: fan::fs_in / fs_out / fs_app / fan::fs_o_rdwr etc.

  fan::event::fs_watcher_t watcher{path};
  watcher.start([](const std::string& filename, int events) {
    // events: fan::fs_change or fan::fs_rename
  });
  watcher.stop();

  fan::io::async_directory_iterator_t it;
  it.sort_alphabetically = true;
  fan::io::async_directory_iterate(&it, path);
  // it.callback = [](const std::filesystem::directory_entry& e) -> fan::event::task_t { ... };
  it.stop();

process spawning:
  auto result = co_await fan::process::run_async(args_vector, log_dispatcher);
  result.ok()  // bool

networking (import fan.network;):
  // TCP server:
  fan::network::tcp_t server
  server.listen(port, [](fan::network::connection_t& conn) {
    conn.on_data = [&](std::span<const std::uint8_t> data) { ... }
  })
  // TCP client:
  fan::network::client_t client
  client.connect("127.0.0.1", port)
  client.send(data)
  // HTTP client:
  fan::network::client_t http
  co_await http.get(url, [](fan::network::response_t& res) { ... })
  http.get_sync(url)                              // blocking, use in coroutine
  // UDP:
  fan::network::udp_send_t sender
  sender.send(ip, port, data)
  fan::network::udp_recv_t receiver
  receiver.bind(port, [](std::span<const std::uint8_t> data) { ... })

shader:
  fan::graphics::read_shader("path/to/shader.glsl")  // reads file relative to source location
  fan::graphics::shader_update_fragment(shape_type, glsl_string)  // hot-reload fragment
  engine.shader_recompile_all()             // recompiles all shaders from disk
  // Ctrl+Shift+R triggers this automatically via input action

timers:
  fan::time::timer t{(uint64_t)8e9, true};  // bool operator, finished when true
  fan::time::seconds_timer(f32_t)
  fan::time::millis_timer(f32_t)
  t.restart() / t.millis() / t.seconds() / t.duration_seconds()
  fan::time::interval_t iv{seconds};        // dt-based, use for per-frame intervals
  if (iv.tick(dt)) { /* fires every interval */ }
  // every<>() — same-line aliasing risk within same file, put calls on separate lines

random:
  fan::random::value(min, max)              // f32_t, int, etc.
  fan::random::value_f32(min, max)
  fan::random::value_i64(min, max)
  fan::random::color() / fan::random::bright_color()
  fan::random::vec2(min, max)

math / vectors:
  fan::vec2, fan::vec3, fan::vec4           // .x .y .z .w
  fan::vec2(scalar)                         // broadcasts
  v.normalize() / v.length() / v.dot(v2) / v.snap_to_grid(n)
  v.min() / v.max() / v.clamp(lo, hi) / v.floor() / v.offset_z(f)
  v.offset_x(f) / v.offset_y(f)             // returns copy with x/y offset
  fan::math::pi / fan::math::two_pi / fan::math::sgn(x)
  fan::color::hsv(h, s, v) / fan::color::hsl(h, s, l)
  fan::color::from_rgb(0xRRGGBB) / fan::color::from_rgba(0xRRGGBBAA)
  fan::colors::red / ::green / ::blue / ::white / ::black / ::transparent
  color.set_alpha(f32_t)                    // returns copy

json / save games:
  fan::json j; j.get_if("key", value); j.set("key", value);
  fan::json::load_struct(path, struct_with_json_read)
  fan::json::save_struct(path, struct_with_json_write)
  // implement json_read(fan::json&) and json_write(fan::json&) on struct
  struct game_state_t {
    int level = 1
    fan::vec2 player_pos
    void json_read(fan::json& j) { j.get_if("level", level); j.get_if("pos", player_pos); }
    void json_write(fan::json& j) { j.set("level", level); j.set("pos", player_pos); }
  }
  game_state_t state
  fan::json::load_struct("save.json", state)      // loads into state via json_read
  fan::json::save_struct("save.json", state)      // writes state via json_write
  // manual JSON:
  fan::json j = fan::json::load("data.json")
  j["nested"]["key"]                               // operator chaining
  j.get("arr")[0]                                  // array access

file / path:
  fan::io::file::read(path, &str)           // bool
  fan::io::file::exists(path)
  fan::path::join(dir, file)

printing:
  fan::print(args...)                       // space separated, newline
  fan::printcl(args...)                     // prints to console
  fan::printcl_err(args...)                 // error highlight
  fan::print_throttled(args...)             // rate-limited
  fan::print_error("msg")

misc:
  fan::enumerate(container)                 // yields [index, value]
  fan::get_hash("string")                   // constexpr hash
  fan::get_key_name(key)
  fan::format_number(f64_t)                 // trims trailing zeros
  fan::throw_error("msg")
  fan::to_string(value, precision)
  fan::join_keys(keys_vec, separator)
  fan::auto_color_transition_t
    .start(from, to, duration_s, cb)
    .start_once(...) / .on_end = []{...}
  fan::noise_t                              // fractal noise
    .seed .frequency .gain .lacunarity .octaves
    .apply() / .generate_data(size)         // returns std::vector<uint8_t>
  fan::graphics::folder_open_dialog_t
    .open(initial_path, on_confirm_cb)
    .is_finished()                          // call each frame
interactive_camera_t:
  fan::graphics::interactive_camera_t icam
  icam.create(camera, viewport, zoom, angle)
  icam.pan_with_middle_mouse = true               // middle-click drag to pan
  icam.reset_view()                               // snap to default view
  icam.set_position(pos)                          // move camera to world pos
  icam.ignore_input = true                        // disable user input control
  // usage: call icam.update() each frame before drawing

tilemap_t:
  fan::graphics::tilemap_t map{tile_size, color, area, offset, render_view};
  map.to_grid(world_pos)                    // fan::vec2i
  map.get_tile(cell)                        // shape_t&
  map.set_tile_color(cell, color)
  map.set_tile_image(cell, image)
  map[fan::vec2i{x,y}]                      // shape_t&
  map.in_bounds(cell)                       // bool
  map.highlight(shape, color)              // works for circle_t and line_t shapes

tilemap_instance_t (level editor format):
  // loads .fte files from the tilemap editor, supports physics collisions + markers
  tilemap_instance_t map(renderer, "level.fte", {
    .position = fan::vec2(0, 0),
    .size = fan::vec2i(16, 9) * 5.f,  // tiles wide * tall, multiplied by tile display size
    .build_collisions = true           // auto-creates Box2D colliders from collision layer
  });
  map.setup_view(player_body, interactive_camera, zoom)
  map.update(player_position)           // call each frame, updates viewport
  // editor-placed markers (keys, doors, spawns, triggers):
  map.iterate_marks({
    {"key", [&](auto& m) {
      key.open(player.body, {
        .position = m.position, .size = 32.f,
        .image = {"images/key.webp", image_presets::pixel_art()}
      }, [](physics::sprite_t& s) { s.erase(); });
    }},
    {"door", [&](auto& m) {
      door.open(player.body, { .position = ..., .size = ... },
        [&](physics::sprite_t& s) { /* on interact */ });
    }},
  });

transitions / easing:
  fan::auto_color_transition_t t;
  t.start(from, to, duration_s, cb)
  t.start_once(...) / t.on_end = []{...}
  fan::pulse_red(duration)                  // white <-> red loop
  fan::fade_out(duration)                   // white -> transparent
  fan::move_linear(from, to, duration)
  fan::move_pingpong(from, to, duration)
  fan::ease_e::linear / sine / pulse / ease_in / ease_out
  fan::apply_ease(ease_e, t)                // f32_t in, f32_t out

trail_t:
  fan::graphics::trail_t trail;
  trail.color     = fan::colors::white;
  trail.thickness = 4.f;
  trail.set_point(position, drift_intensity)  // call each frame with current pos
  // auto-fades, auto-cleans old segments

console commands (built-in, FAN_GUI — F3 in-game):
  set_vsync [0/1]              set_target_fps [n]
  set_gamma [f]                set_contrast [f]
  set_exposure [f]             set_bloom_strength [f]
  set_clear_color [{r,g,b,a}]  set_lighting_ambient [{r,g,b}]
  show_fps [0/1]               debug_memory [0/1]
  rectangle {x,y,z} {w,h} {r,g,b,a}   // adds static shape
  remove_shape [id]
  echo [args]  help [command]  list  quit

performance monitoring (FAN_GUI):
  engine.show_fps = true        // shows FPS counter top-right
  // full perf window shows frame/shape/GUI draw times with plot
  // debug_memory shows heap allocation plot with stack trace (requires C++23)


audio (requires #define FAN_AUDIO):
  fan::audio::sound_t sound{"path.ogg"};
  sound.play() / sound.stop() / sound.set_volume(f32_t) / sound.set_pitch(f32_t)
  sound.is_playing()
  
  
ecs (fan::ecs_t, import fan.ecs):
  // standard components: c_pos, c_vel, c_hp {current, max}, c_life {timer}, c_cost
  using registry_t = fan::ecs_t<c_pos, c_vel, c_hp, tag_enemy>;
  registry_t registry;

  // entity lifecycle:
  uint32_t id = registry.create();
  uint32_t id = registry.create_with(c_pos{{0, 0}}, c_vel{{1, 0}}, tag_enemy{});
  registry.destroy(id);
  registry.clear();
  registry.on_destroy_hooks.push_back([](uint32_t id) { ... });

  // component access:
  registry.add<c_hp>(id, 10, 10);
  registry.remove<c_hp>(id);
  registry.has<c_pos, c_vel>(id)            // bool
  registry.get<c_pos>(id).v                 // fan::vec2&

  // iteration (auto-deduces requested components from lambda args):
  registry.each([&](uint32_t id, c_pos& p, tag_enemy&) { ... });
  registry.each([&](c_pos& p, c_vel& v) { ... });  // id is optional
  registry.each_breakable([&](c_pos& p) { return false; }); // returns false to break

  // queries & conditional destruction:
  registry.any<tag_enemy>()                 // bool
  registry.destroy_if([&](c_pos& p) { return p.v.x < 0; });
  registry.destroy_dead<c_pos>([&](c_pos& p) { ... }); // destroys if c_hp.current <= 0
  registry.destroy_at<tag_enemy>(pos, [&]{ ... });     // exact fan::vec2 match

  // built-in systems (fan::ecs::systems):
  fan::ecs::systems::apply_drag<c_vel, tag_particle>(registry, 0.95f);
  fan::ecs::systems::kinematics<c_pos, c_vel>(registry, dt);
  fan::ecs::systems::lifetimes<c_life>(registry, dt);

---

io & files:
  // getting file names without string parsing:
  std::filesystem::path("path/to/file.ext").stem().string() // returns "file"
  std::filesystem::path("path/to/file.ext").filename().string() // returns "file.ext"
  
  // fan utilities (import fan.io.file / fan.io.directory)
  fan::io::file::strip_extension(path)                      // removes .ext
  fan::io::exclude_path(path)                               // removes directory
