// Hot-reload host. Thin on purpose: engine + persistent state + reload
// wiring. Shape storage lives in the fan library
// (fan::graphics::hot_shape_store_t); ALL gameplay is in Game.cpp.
//
// Run:
//   ./compile_main.sh --hot-reload   # engine host + Game.dll / libGame.so
//   ./a.exe
// Then edit examples/hot_reload/Game.cpp, run ./compile_main.sh --shared Game.

import std;
import fan;

#include "game_api.h"

using namespace fan::graphics;

#if defined(_WIN32)
static constexpr const char* k_game_file = "Game.dll";
#else
static constexpr const char* k_game_file = "libGame.so";
#endif

struct host_t {
  engine_t engine; // init first: window/renderer/world stay alive across reloads
  GameState state; // persistent, owned by ENGINE (never by the DLL)
  fan::hot_reload_t game{k_game_file}; // watcher + staging + swap (fan library)
  hot_shape_store_t shapes; // every shape type, keyed by game stable ids
  EngineAPI engine_api{};
  const GameAPI* api = nullptr;

  host_t() {
    // Three functions for every shape type. Non-capturing lambdas decay
    // to C function pointers; ctx carries `this`.
    engine_api.ctx = this;
    engine_api.shape = [](void* ctx, const ShapeDesc* d) {
      static_cast<host_t*>(ctx)->shapes.upsert(*d);
    };
    engine_api.drop = [](void* ctx, int id) {
      static_cast<host_t*>(ctx)->shapes.drop(id);
    };
    engine_api.drop_all = [](void* ctx) {
      static_cast<host_t*>(ctx)->shapes.clear();
    };

    // Adopt a freshly staged library: check ABI, move persistent
    // state over (never reset x/y), return true to commit the swap.
    game.on_reload = [&](dynamic_library_t& fresh) -> bool {
      auto fn = (GetGameAPIFn)fresh.symbol("GetGameAPI");
      const GameAPI* next = fn ? fn() : nullptr;
      if (!next || !next->update || next->abi_version != game_abi_version) {
        fan::print("hot-reload: bad GameAPI (missing exports or ABI mismatch)");
        return false; // keep old version running
      }
      if (api && api->on_unload) {
        api->on_unload(&state);
      }
      // Fresh visuals next frame: game code re-emits every shape in update().
      shapes.clear();
      api = next;
      if (api->on_load) {
        api->on_load(&state, &engine_api);
      }
      fan::print("hot-reload: loaded v", state.dll_version,
        "pos:", state.x, state.y);
      return true;
    };
    game.reload(); // boot load before the first frame
  }

  void update() {
    double dt = engine.get_delta_time();
    game.update(dt); // debounced watcher + fallback poll, swaps when verified

    if (engine.is_key_clicked(fan::key_r)) {
      game.reload(); // manual reload
    }

    if (api && api->update) {
      fan::vec2 in = engine.get_input_vector();
      api->update(&state, &engine_api, (float)dt, in.x, in.y);
    }

    gui::text("WASD/arrows - move (logic is in Game.cpp)");
    gui::text("dll v:", state.dll_version,
      " tick:", state.tick, " pos:", (int)state.x, (int)state.y,
      " shapes:", (int)shapes.size());
    gui::text("edit Game.cpp -> ./compile_main.sh --shared Game -> auto reload, no restart");
    gui::text("R - force reload");
  }
};

int main() {
  host_t host;
  host.engine.loop([&] {
    host.update();
  });
}
