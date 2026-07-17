import std;
import fan;

static constexpr f32_t cfg_bullet_speed = 4000.f;
static constexpr f32_t cfg_bullet_life  = 1.5f;
static constexpr f32_t cfg_shoot_cd     = 0.01f;

static constexpr auto player_size = fan::vec2(48.f);
static constexpr auto capsule_center0 = fan::vec2(0.f, -24.f);
static constexpr auto capsule_center1 = fan::vec2(0.f, 24.f);
static constexpr auto capsule_radius = player_size.x / 2.f;
static constexpr auto depth_background = 100;
static constexpr auto depth_tiles = 200;
static constexpr auto depth_player = 300;
static constexpr auto depth_trail = 350;
static constexpr auto depth_bullet = 400;
static constexpr auto depth_muzzle = 450;
static constexpr auto depth_casing = 500;
static constexpr auto depth_feet = 0xfffa;
static constexpr auto depth_flash = 0xfffe;
static constexpr auto feet_y_offset = capsule_center1.y + capsule_radius;
static constexpr auto feet_x_offset = capsule_radius;
static constexpr f32_t cfg_muzzle_flash_life = 0.01f;
static constexpr f32_t cfg_muzzle_sprite_size = 32.f;

struct player_t {
  void init(fan::vec2 spawn_point) {
    body = fan::graphics::physics::character_capsule_sprite(
      fan::vec3(spawn_point, depth_player),
      capsule_center0,
      capsule_center1,
      player_size,
      fan::graphics::get_default_texture(),
      {.friction = 0.6f, .fixed_rotation = true, .filter = {.categoryBits = 2, .maskBits = 0xFFFFFFFF}}
    );
    body.set_color(fan::color(0.2f, 0.6f, 1.0f));
    body.enable_default_movement();
    body.enable_double_jump();
    body.set_jump_height(48.f);
  }

  fan::graphics::physics::character2d_t body;
};

struct tag_muzzle_flash {};

using casing_body_t = fan::graphics::physics::sprite_t;

using registry_t = fan::ecs_t<
  fan::ecs::c_pos, fan::ecs::c_vel, fan::ecs::c_life, fan::ecs::c_line,
  fan::ecs::tag_bullet, 
  tag_muzzle_flash
>;

struct pile_t : fan::graphics::engine_t, fan::frame_task_t<pile_t> {

  struct example_stage_t : fan::stage_t<example_stage_t> {
    void open(void* sod) {
      map_id = renderer.open_map("map.json", {
        .position = fan::vec2(0),
        .size = fan::vec2i(64, 32),
        .depth_fn = [](auto&, auto&, auto&, f32_t) { return depth_tiles; },
      });

      tile_world.init(fan::vec2(-10000), fan::vec2(256), fan::vec2i(256));
      renderer.iterate_tiles(map_id, [&, i = 0u](const auto& tile) mutable {
        collisions.emplace_back(engine->get_physics_context().create_box(
          tile.position, tile.size, 0, fan::physics::body_type_e::static_body,
          fan::physics::shape_properties_t{.friction=1.f}
        ));
        tile_world.upsert(i++, fan::physics::aabb_t::from_center(tile.position, tile.size), fan::spatial::movement_static);
      });

      player.init(renderer.get_spawn(map_id));
      player.body.movement_state.jump_state.on_jump = [this](int jump_type) {
        auto squished = fan::vec2(player_size.x * 1.5f, player_size.y * 0.5f);
        player.body.squish(squished, player_size, 0.7f);
        auto pos = player.body.get_position();
        
        int count = jump_type == 1 ? 6 : 8;
        f32_t alive_min = jump_type == 1 ? 0.3f : 0.4f;
        f32_t alive_max = jump_type == 1 ? 0.6f : 0.8f;
        
        particles.spawn_footstep_dust(fan::vec3(pos.x, pos.y + feet_y_offset, depth_feet), feet_x_offset, count, smoke_image, alive_min, alive_max);
      };
      engine->camera_follow(player.body.get_position(), 0);
      ic.set_zoom(1.728f);
    }

    void close() {
      tile_world.reset();
      for (auto& c : collisions) { c.destroy(); }
      collisions.clear();
      casing_bodies.clear();
      renderer.close_map(map_id);
    }

    void fire_bullet() {
      shoot_task = start_shoot_cd();
      if (!is_hitstop) {
        hitstop_task = apply_hitstop();
      }

      auto pos = player.body.get_position();
      auto viewport_center = ic.get_viewport_size() * 0.5f;
      auto mouse_screen = fan::graphics::get_mouse_position();
      f32_t dir = mouse_screen.x > viewport_center.x ? 1.f : -1.f;

      // gun recoil movement via knockback system
      player.body.movement_state.knockback_initial_velocity.x = -dir * 300.f;
      player.body.movement_state.knockback_ticks_remaining = 3;
      player.body.movement_state.is_in_knockback = true;

      // random bullet spread
      f32_t spread = fan::random::value(-0.05f, 0.05f);
      f32_t speed = cfg_bullet_speed * fan::random::value(0.95f, 1.05f);
      f32_t dy = fan::random::value(-8.f, 8.f);
      fan::vec2 bullet_vel = fan::vec2(dir * speed, speed * spread + dy);
      fan::vec2 muzzle_pos = pos + fan::vec2(dir * 80.f, -4.f);
      
      registry.create_with(fan::ecs::tag_bullet{}, fan::ecs::c_pos{muzzle_pos},
        fan::ecs::c_vel{bullet_vel},
        fan::ecs::c_life{cfg_bullet_life},
        fan::ecs::c_line{bullet_vel.normalize() * 15.f, fan::color(1.0, 0.7, 0.1), 3.f, depth_bullet});

      // muzzle flash burst
      auto flash_col = fan::color(1.0, 0.6, 0.05);
      for (auto [offset, weight] : {
        std::pair{fan::vec2(dir * 28.f, 0.f), 2.5f},
        std::pair{fan::vec2(dir * 18.f, -1.f), 2.f},
        std::pair{fan::vec2(dir * 18.f, 1.f), 2.f}
      }) {
        registry.create_with(tag_muzzle_flash{}, fan::ecs::c_pos{muzzle_pos},
          fan::ecs::c_life{0.08f},
          fan::ecs::c_line{offset * 4.f, flash_col, weight, depth_muzzle});
      }

      muzzle_task = render_muzzle_flash(muzzle_pos);

      {
        fan::vec2 casing_vel = fan::vec2(-dir * 400.f, -450.f);
        casing_bodies.push_back(casing_body_t{{
          .position = fan::vec3(muzzle_pos, depth_casing),
          .size = fan::vec2(3.f, 2.f),
          .image = fan::graphics::image_t{fan::colors::orange},
          .body_type = fan::physics::body_type_e::dynamic_body,
          .shape_properties = {.friction=0.4f, .filter={4, ~std::uint32_t(2 | 4), 0}},
        }});
        casing_bodies.back().set_linear_velocity(casing_vel);
        fan::event::add_awaitable(fan::event::after(2000, [this, idx = casing_bodies.size() - 1]{
          casing_bodies[idx].set_filter({4, ~std::uint32_t(4), 0});
        }));
      }

      // camera effects
      ic.shake(0.01f, 0.01f);
      ic.bump(fan::vec2(-dir, 0), 0.3f, 0.06f);
      ic.bump_zoom(-0.01f, 0.04f);
      ic.flash(0.03f, 0.04f);
    }

    void update() {
      f32_t dt = pile.get_delta_time();

      if (is_hitstop) {
        dt *= 0.08f;
      }

      if (engine->are_keys_down(fan::key_left_control, fan::key_left_shift) && engine->is_key_clicked(fan::key_r)) {
        fan::image::async_cache().clear();
        engine->stage_restart<example_stage_t>();
      }

      bool on_ground = player.body.is_on_ground();
      if (!player.body.was_on_ground && on_ground) {
        auto pos = player.body.get_position();
        particles.spawn_footstep_dust(fan::vec3(pos.x, pos.y + feet_y_offset, depth_feet), feet_x_offset, 6, smoke_image, 0.3f, 0.6f);

        // land squish
        auto squished = fan::vec2(player_size.x * 1.2f, player_size.y * 0.7f);
        player.body.squish(squished, player_size, 0.5f);
      }
      player.body.was_on_ground = on_ground;

      particles.update(dt);

      if (engine->is_mouse_down(fan::mouse_left) && !fan::graphics::gui::want_io() && can_shoot) {
        fire_bullet();
      }

      // bullet trails
      registry.each<fan::ecs::c_pos, fan::ecs::c_vel, fan::ecs::tag_bullet>([&](std::uint32_t, fan::ecs::c_pos& p, fan::ecs::c_vel& v, fan::ecs::tag_bullet&) {
        registry.create_with(
          fan::ecs::c_pos{p.v - v.v * (dt * 0.5f)},
          fan::ecs::c_life{0.25f},
          fan::ecs::c_line{-v.v * (dt * 0.8f), fan::colors::yellow.set_alpha(0.45f), 2.5f, depth_trail}
        );
      });

      fan::physics::destroy_bullets_vs_tiles(registry, tile_world, 4.f);

      fan::ecs::systems::kinematics<fan::ecs::c_pos, fan::ecs::c_vel>(registry, dt);
      fan::ecs::systems::lifetimes<fan::ecs::c_life>(registry, dt);

      renderer.update(map_id, player.body.get_center());
      engine->camera_follow(player.body.get_position());
      ic.update_fx(dt);

      fan::graphics::systems::render2d(registry);

      // flash overlay
      f32_t flash_alpha = ic.get_flash_alpha();
      if (flash_alpha > 0) {
        fan::vec2 s = ic.get_viewport_size() / ic.get_zoom();
        fan::vec2 center = fan::graphics::camera_get_position(ic.render_view.camera);
        fan::graphics::rectangle(fan::vec3(center, depth_flash), s, fan::color(1, 1, 1, flash_alpha));
      }

      draw_gui();
    }

    void draw_gui() {
      if (auto h = fan::graphics::gui::hud{"##hud"}) {

      }
    }

    fan::event::task_t start_shoot_cd() {
      can_shoot = false;
      co_await fan::event::timer_t(cfg_shoot_cd * 1000.f);
      can_shoot = true;
    }

    fan::event::task_t apply_hitstop() {
      is_hitstop = true;
      co_await fan::event::timer_t(40);
      is_hitstop = false;
    }

    fan::event::task_t render_muzzle_flash(fan::vec2 pos) {
      auto t = fan::time::seconds_timer(cfg_muzzle_flash_life);
      while (!t) {
        f32_t muzzle_scale = std::sin(t.seconds() / cfg_muzzle_flash_life * fan::math::pi);
        fan::graphics::light(
          fan::vec3(pos, depth_muzzle), 
          fan::vec2(400.f) * muzzle_scale, 
          fan::color(1.0, 0.5, 0.05, 0.6) / 15.f
        );
        fan::graphics::circle(
          fan::vec3(pos, depth_muzzle),
          21.f * muzzle_scale,
          fan::color(1.0, 0.5, 0.05, 0.6) * 2.f
        );
        fan::graphics::sprite({
          .position = fan::vec3(pos, depth_muzzle),
          .size = cfg_muzzle_sprite_size * muzzle_scale,
          .color = fan::color(1.0, 0.1, 0.05, 0.6) * 2.f,
          .image = {"images/smoke.webp", fan::graphics::image_presets::pixel_art()}
        });
        co_await fan::graphics::co_next_frame();
      }
    }

    fan::graphics::tilemap_renderer_t::id_t map_id;
    std::vector<fan::physics::entity_t> collisions;
    fan::spatial::world_t<std::uint32_t> tile_world;
    fan::graphics::tilemap_renderer_t renderer;
    player_t player;
    fan::graphics::interactive_camera_t ic;
    fan::graphics::gpu_particle_system_t<> particles;
    fan::graphics::image_t smoke_image{"images/smoke.webp"};
    registry_t registry;
    std::deque<casing_body_t> casing_bodies;
    bool can_shoot = true;
    bool is_hitstop = false;
    fan::event::task_t shoot_task;
    fan::event::task_t hitstop_task;
    fan::event::task_t muzzle_task;
  };

  pile_t() {
    set_settings({
      .clear_color = fan::colors::black,
      .ambient_color = fan::colors::white,
    });
    update_physics(true);
    stage_open<example_stage_t>();
  }
} pile;

int main() {
  pile.loop();
}