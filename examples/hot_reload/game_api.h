#pragma once

import std;

#include <fan/graphics/shape_desc.h>

static constexpr std::uint32_t game_abi_version = 3;

// Persistent gameplay state. Owned by the ENGINE, never by the DLL,
// so it survives reloads.
struct GameState {
  float x = 0.0f;
  float y = 0.0f;
  float r = 1.0f;
  float g = 0.2f;
  float b = 0.3f;
  float time = 0.0f;
  int tick = 0;
  int dll_version = 0;
};

static_assert(std::is_standard_layout_v<GameState>);

// Services the ENGINE provides to game code. Exactly three functions
// for every shape type, present and future: shapes are plain-data
// descriptors (ShapeDesc), owned by the host, keyed by game-chosen
// stable ids. Function pointers stay valid across reloads
// (they point into the host).
struct EngineAPI {
  void* ctx;
  // Create-or-update. Call every frame for every visible shape.
  void (*shape)(void* ctx, const ShapeDesc* desc);
  void (*drop)(void* ctx, int id);
  void (*drop_all)(void* ctx);
};

// Pretty, zero-ABI-cost emitters for game code.
static inline void draw_rect(const EngineAPI* e, int id,
    float x, float y, float w, float h,
    float r, float g, float b, float a) {
  ShapeDesc d;
  d.id = id; d.type = SHAPE_RECT;
  d.x = x; d.y = y;
  d.r = r; d.g = g; d.b = b; d.a = a;
  d.u.rect.w = w; d.u.rect.h = h;
  e->shape(e->ctx, &d);
}

static inline void draw_circle(const EngineAPI* e, int id,
    float x, float y, float radius,
    float r, float g, float b, float a) {
  ShapeDesc d;
  d.id = id; d.type = SHAPE_CIRCLE;
  d.x = x; d.y = y;
  d.r = r; d.g = g; d.b = b; d.a = a;
  d.u.circle.radius = radius;
  e->shape(e->ctx, &d);
}

static inline void draw_line(const EngineAPI* e, int id,
    float x1, float y1, float x2, float y2, float thickness,
    float r, float g, float b, float a) {
  ShapeDesc d;
  d.id = id; d.type = SHAPE_LINE;
  d.x = x1; d.y = y1;
  d.r = r; d.g = g; d.b = b; d.a = a;
  d.u.line.x2 = x2; d.u.line.y2 = y2; d.u.line.thickness = thickness;
  e->shape(e->ctx, &d);
}

// Centers are relative to (x, y).
static inline void draw_capsule(const EngineAPI* e, int id,
    float x, float y, float x0, float y0, float x1, float y1, float radius,
    float r, float g, float b, float a) {
  ShapeDesc d;
  d.id = id; d.type = SHAPE_CAPSULE;
  d.x = x; d.y = y;
  d.r = r; d.g = g; d.b = b; d.a = a;
  d.u.capsule.x0 = x0; d.u.capsule.y0 = y0;
  d.u.capsule.x1 = x1; d.u.capsule.y1 = y1;
  d.u.capsule.radius = radius;
  e->shape(e->ctx, &d);
}

// Services the GAME module provides to the engine.
struct GameAPI {
  std::uint32_t abi_version;
  void (*on_load)(GameState* state, const EngineAPI* engine);
  void (*on_unload)(GameState* state);
  void (*update)(GameState* state, const EngineAPI* engine,
    float dt, float ix, float iy);
};

#if defined(_WIN32)
  #define GAME_API_EXPORT __declspec(dllexport)
#else
  #define GAME_API_EXPORT __attribute__((visibility("default")))
#endif

#if defined(__cplusplus)
extern "C" {
#endif

GAME_API_EXPORT const GameAPI* GetGameAPI();

#if defined(__cplusplus)
}
#endif

using GetGameAPIFn = const GameAPI* (*)();
