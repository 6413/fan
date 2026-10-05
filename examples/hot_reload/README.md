# Hot reload demo (fan engine + Game.dll / libGame.so)

Engine stays running. Gameplay is reloaded live.

## Layout

- `main.cpp` — thin fan host. Owns `engine_t`, persistent `GameState`,
  shape storage, and reload wiring. No gameplay. Never restarts.
- `game_api.h` — two-way stable C ABI shared by host and DLL:
  `EngineAPI` (host→game: exactly 3 functions — `shape`/`drop`/`drop_all`
  — for every shape type, present and future) and `GameAPI`
  (game→host: `on_load`/`on_unload`/`update`). Shapes cross as plain-data
  `ShapeDesc` structs (`fan/graphics/shape_desc.h`); pretty `draw_rect` /
  `draw_circle` / `draw_line` / `draw_capsule` helpers cost zero ABI.
  No fan types, so the DLL stays dependency-free and rebuilds in ~1s.
- `fan/graphics/hot_shape_store.ixx` — host-side shape storage
  (`fan::graphics::hot_shape_store_t`): one `map<id, shape_t>` for all
  types, create-on-first-sight + generic setters, recreate only for
  structural params (line endpoints, capsule centers). New shape types
  are additive and never break old DLLs.
- `Game.cpp` — ALL gameplay: movement, shapes, colors. This is what
  you edit. Exports `GetGameAPI()`. Shapes are drawn by calling e.g.
  `e->rect(e->ctx, id, ...)` every frame with a stable id.
- `fan/system/hot_reload.ixx` — the reload machinery as a fan library
  component (`fan::hot_reload_t`, via `import fan`): libuv file watcher
  (`fan::event::fs_watcher_t`), debouncing, unique staging copies,
  verify-before-swap, and an mtime poll fallback. Game code only
  provides the `on_reload` adopt callback.

Loading uses fan's own abstraction (`fan/system/dynamic_library.ixx`:
`LoadLibrary`/`GetProcAddress`/`FreeLibrary` on Windows,
`dlopen`/`dlsym`/`dlclose` on Linux), now exported through `import fan`.

## Build

All through `compile_main.sh`:

```sh
./compile_main.sh --hot-reload
./a.exe            # linux
a.exe              # windows
```

`--hot-reload` sets `--main examples/hot_reload/main.cpp` and builds
both the engine host and the `Game` shared target (`set_kind("shared")`
in `xmake.lua`), so xmake emits `Game.dll` on Windows and `libGame.so`
on Linux into the project root, where the host looks.

## Hot reload loop

1. Change `Game.cpp` (e.g. `SPEED`, `GAME_VERSION`, color formula).
2. `./compile_main.sh --shared Game` (~1s, DLL only, engine untouched)
3. The fan file watcher fires on the rebuild; after a 0.35s debounce
   (coalesces linker bursts) the host stages the library to a unique
   file (`libGame_hot_<n>.so` / `Game_hot_<n>.dll`), verifies it via
   the `on_reload` adopt callback, and swaps it in — no engine restart.
   A 1s mtime poll runs behind the watcher as a safety net.
4. Player position survives because `GameState` lives in the host.

Notes:

- The staging copy is what makes overwrite safe: Windows locks a
  loaded `.dll`, and overwriting a mapped `.so` on Linux crashes.
  Loading the copy leaves the original free for the compiler.
- If a reload fails (e.g. compiler mid-write), the old version keeps
  running and the next poll retries.
- Press `R` in the running engine to force a reload check.
- `on_load` must never reset `x`/`y`/`time` — that is the whole point.
