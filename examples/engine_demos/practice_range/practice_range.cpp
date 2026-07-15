#include <cstdint>
#include <vector>
import std;
import fan;
import fan.physics.vehicle_controller;

using namespace fan::graphics;

static constexpr f32_t cfg_bullet_speed = 800.f;
static constexpr f32_t cfg_bullet_life  = 1.5f;
static constexpr f32_t cfg_shoot_cd     = 0.15f;

static constexpr auto player_size = fan::vec2(48.f);
static constexpr auto capsule_center0 = fan::vec2(0.f, -24.f);
static constexpr auto capsule_center1 = fan::vec2(0.f, 24.f);
static constexpr auto capsule_radius = player_size.x / 2.f;
static constexpr auto feet_z = 0xfffa;
static constexpr auto feet_y_offset = capsule_center1.y + capsule_radius;
static constexpr auto feet_x_offset = capsule_radius;

struct player_t {
  void init(fan::vec2 spawn_point) {
    body = fan::graphics::physics::character_capsule_sprite(
      fan::vec3(spawn_point, 10),
      capsule_center0,
      capsule_center1,
      player_size,
      fan::graphics::get_default_texture(),
      {.friction = 0.6f, .fixed_rotation = true}
    );
    body.set_color(fan::color(0.2f, 0.6f, 1.0f));
    body.enable_default_movement();
    body.enable_double_jump();
    body.set_jump_height(48.f);
  }

  fan::graphics::physics::character2d_t body;
  fan::tween::tween_manager_t tweens;
};

using registry_t = fan::ecs_t<
  fan::ecs::c_pos, fan::ecs::c_vel, fan::ecs::c_life, fan::ecs::c_line,
  fan::ecs::c_rectangle, fan::ecs::c_rectangle_bordered, fan::ecs::tag_bullet
>;

struct pile_t : engine_t, fan::frame_task_t<pile_t> {

  struct example_stage_t : fan::stage_t<example_stage_t> {
    void open(void* sod) {
      map_id = renderer.open_map("map.json", {
        .position = fan::vec2(0),
        .size = fan::vec2i(64, 32),
      });

      tile_world.init(fan::vec2(-10000), fan::vec2(256), fan::vec2i(256));
      renderer.iterate_tiles(map_id, [&, i = 0u](const auto& tile) mutable {
        fan::physics::shape_properties_t sp;
        sp.friction = 0;
        collisions.emplace_back(engine->get_physics_context().create_box(
          tile.position, tile.size, 0, fan::physics::body_type_e::static_body, sp
        ));
        tile_world.upsert(i++, fan::physics::aabb_t::from_center(tile.position, tile.size), fan::spatial::movement_static);
      });

      player.init(renderer.get_spawn(map_id));
      player.body.movement_state.jump_state.on_jump = [this](int jump_type) {
        auto squished = fan::vec2(player_size.x * 1.5f, player_size.y * 0.5f);
        player.body.set_size(squished);
        player.tweens.add<fan::vec2>(
          [this](fan::vec2 sz) { player.body.set_size(sz); },
          squished, player_size,
          0.7f, fan::tween::easing::out_elastic
        );
        auto pos = player.body.get_position();
        int count = 8;
        f32_t alive_min = 0.4f;
        f32_t alive_max = 0.8f;
        if (jump_type == 1) {
          count = 6;
          alive_min = 0.3f;
          alive_max = 0.6f;
        }
        auto c = decltype(particles)::smoke_config_t{};
        c.alive_time_min = alive_min;
        c.alive_time_max = alive_max;
        c.start_size = 15.f;
        c.end_size = 50.f;
        c.alpha = 0.25f;
        auto p = fan::vec3(pos.x, pos.y + feet_y_offset, feet_z);
        particles.spawn_smoke(p - fan::vec3(feet_x_offset, 0, 0), count, smoke_image, c);
        particles.spawn_smoke(p + fan::vec3(feet_x_offset, 0, 0), count, smoke_image, c);
      };
      engine->camera_follow(player.body.get_position(), 0);
      ic.set_zoom(1.728f);
    }

    void close() {
      tile_world.reset();
      for (auto& c : collisions) { c.destroy(); }
      collisions.clear();
      renderer.close_map(map_id);
    }

    void update() {
      f32_t dt = engine->get_delta_time();

      if (engine->are_keys_down(fan::key_left_control, fan::key_left_shift) && engine->is_key_clicked(fan::key_r)) {
        fan::image::async_cache().clear();
        engine->stage_restart<example_stage_t>();
      }

      bool on_ground = player.body.is_on_ground();
      if (!player.body.was_on_ground && on_ground) {
        auto pos = player.body.get_position();
        auto c = decltype(particles)::smoke_config_t{};
        c.alive_time_min = 0.3f;
        c.alive_time_max = 0.6f;
        c.start_size = 15.f;
        c.end_size = 50.f;
        c.alpha = 0.25f;
        auto p = fan::vec3(pos.x, pos.y + feet_y_offset, feet_z);
        particles.spawn_smoke(p - fan::vec3(feet_x_offset, 0, 0), 6, smoke_image, c);
        particles.spawn_smoke(p + fan::vec3(feet_x_offset, 0, 0), 6, smoke_image, c);
      }
      player.body.was_on_ground = on_ground;

      player.tweens.update(dt);
      particles.update(dt);

      shoot_cd.tick(dt);
      if (engine->is_mouse_down(fan::mouse_left) && !fan::graphics::gui::want_io() && shoot_cd.is_ready()) {
        shoot_cd = fan::cooldown_t::full(cfg_shoot_cd);
        f32_t dir = player.body.movement_state.desired_facing.x;
        if (dir == 0) { dir = 1; }
        auto pos = player.body.get_position();
        registry.create_with(fan::ecs::tag_bullet{}, fan::ecs::c_pos{pos}, fan::ecs::c_vel{fan::vec2(dir * cfg_bullet_speed, 0)},
          fan::ecs::c_life{cfg_bullet_life}, fan::ecs::c_line{fan::vec2(dir * 15.f, 0), fan::colors::yellow, 3.f});
      }

      fan::physics::destroy_bullets_vs_tiles(registry, tile_world, 4.f);

      fan::ecs::systems::kinematics<fan::ecs::c_pos, fan::ecs::c_vel>(registry, dt);
      fan::ecs::systems::lifetimes<fan::ecs::c_life>(registry, dt);

      renderer.update(map_id, player.body.get_center());
      ic.update();
      engine->camera_follow(player.body.get_position());

      fan::graphics::systems::render2d(registry);
      draw_gui();
    }

    void draw_gui() {
      if (auto h = fan::graphics::gui::hud{"##hud"}) {

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
    fan::cooldown_t shoot_cd = fan::cooldown_t::full(0.f);
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