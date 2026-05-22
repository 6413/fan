import std;
import fan;

using namespace fan::graphics;

struct pile_t : engine_t, fan::frame_task_t<pile_t> {

  struct example_stage_t : fan::stage_t<example_stage_t> {
    void open(void* sod) {
      id = pile.renderer.open_map("sample_level.fte", {
        .position = pile.player.body.get_position(),
        .size = fan::vec2i(16, 9),
      });

      pile.renderer.iterate_tiles(id, [&](const auto& tile) {
        collisions.emplace_back(pile.get_physics_context().create_box(tile.position, tile.size, 0, fan::physics::body_type_e::static_body, {}));
      });
    }
    void close() {
      for (auto& c : collisions) { c.destroy(); }
      pile.renderer.close_map(id);
    }

    void update() {
      pile.renderer.update(id, pile.player.body.get_position());
    }

    tilemap_renderer_t::id_t id;
    std::vector<fan::physics::entity_t> collisions;
  };

  struct player_t : fan::frame_task_t<player_t> {
    player_t() {
      body.enable_default_movement();
      body.add_child(light);
    }

    physics::character2d_t body = physics::character_capsule({
      .position = fan::vec3(fan::vec2(109, 123) * 64, 10),
      .center0 = {0.f, -24.f},
      .center1 = {0.f, 24.f},
      .radius = 12,
    }, {
      .friction = 0.6f, 
      .fixed_rotation = true
    });
    light_t light{body.get_position(), 200, fan::colors::white};
  };

  pile_t() {
    texture_pack.open_compiled("sample_texture_pack.ftp");
    camera_follow(player.body.get_position(), 0);
    update_physics(true);

    level_stage = stage_loader.open_stage<example_stage_t>();
  }

  void update() {
    camera_follow(player.body.get_position());
  }

  player_t player;
  tilemap_renderer_t renderer;
  fan::stage_loader_t stage_loader;
  fan::stage_loader_t::nr_t level_stage;
  interactive_camera_t ic{orthographic_render_view};
} pile;

int main() {
  pile.loop();
}