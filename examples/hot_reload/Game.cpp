// ALL gameplay lives here. This file becomes Game.dll / libGame.so.
// Edit it, run ./compile_main.sh --shared Game, and the running engine
// picks it up without restarting. GameState (owned by the engine)
// survives reloads, so never reset s->x / s->y / s->time in on_load.
//
// Draw with draw_rect/draw_circle/draw_line/draw_capsule using a stable
// id per shape. Re-emit every frame; the host creates on first sight
// and updates after.

#include "game_api.h"

import std;

static constexpr int game_version = 3;
static constexpr float speed = 400.f;
static constexpr float bound_x = 600.f;
static constexpr float bound_y = 400.f;
static constexpr float two_pi = 2.f * std::numbers::pi_v<float>;

enum shape_id_t : int {
  shape_player = 1,
  shape_orbiter_base = 10, // 10, 11, 12
  shape_halo = 20,
  shape_link = 30,
  shape_enemy = 40,
};

static void game_on_load(GameState* s, const EngineAPI* /*e*/) {
  // NOTE: do NOT reset s->x / s->y / s->time here.
  s->dll_version = game_version;
}

static void game_on_unload(GameState* /*s*/) {
  // Nothing to clean up. State + shapes stay alive in the engine.
}

static void game_update(GameState* s, const EngineAPI* e, float dt, float ix, float iy) {
  s->time += dt;
  s->tick++;
  s->dll_version = game_version;

  // Player rectangle, WASD / arrows. Try changing SPEED above.
  s->x = std::clamp(s->x + ix * speed * dt, -bound_x, bound_x);
  s->y = std::clamp(s->y + iy * speed * dt, -bound_y, bound_y);

  auto wave = [&](float phase) {
    return 0.5f + 0.5f * std::sin(s->time * 2.f + phase);
  };
  s->r = wave(0.f);
  s->g = wave(two_pi / 3.f);
  s->b = wave(two_pi * 2.f / 3.f);

  draw_rect(e, shape_player, s->x, s->y, 64.f, 64.f, s->r, s->g, s->b, 1.f);

  // Orbiters circling the player. Try count / radius / speed.
  float ox = s->x, oy = s->y;
  for (int i = 0; i < 3; ++i) {
    float a = s->time * 1.5f + i * two_pi / 3.f;
    ox = s->x + std::cos(a) * 140.f;
    oy = s->y + std::sin(a) * 140.f;
    draw_circle(e, shape_orbiter_base + i, ox, oy, 20.f,
      wave(i * 2.f), wave(i * 2.f + 2.f), wave(i * 2.f + 4.f), 1.f);
  }

  // Halo ring pulsing around the player.
  draw_circle(e, shape_halo, s->x, s->y,
    90.f + 20.f * std::sin(s->time * 3.f), 1.f, 1.f, 1.f, 0.35f);

  // Link line from the player to the last orbiter.
  draw_line(e, shape_link, s->x, s->y, ox, oy, 3.f, 1.f, 1.f, 1.f, 0.8f);

  // Enemy capsule sweeping across the screen.
  float ex = std::sin(s->time * 0.7f) * 500.f;
  float ey = 250.f + std::cos(s->time * 1.1f) * 150.f;
  draw_capsule(e, shape_enemy, ex, ey, -20.f, -40.f, 20.f, 40.f, 24.f,
    1.f, 0.3f, 0.2f, 1.f);
}

static const GameAPI k_api = {game_abi_version, game_on_load, game_on_unload, game_update};

extern "C" GAME_API_EXPORT const GameAPI* GetGameAPI() {
  return &k_api;
}
