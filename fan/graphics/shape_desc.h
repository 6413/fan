
#pragma once

#include <cstdint>
#include <type_traits>

enum ShapeType : std::int32_t {
  SHAPE_RECT = 1,
  SHAPE_CIRCLE,
  SHAPE_LINE,
  SHAPE_CAPSULE
};

struct ShapeDesc {
  std::int32_t id = 0;   // stable, game-chosen. Re-emitting updates.
  std::int32_t type = 0; // ShapeType
  float x = 0, y = 0, z = 0;
  float r = 1, g = 1, b = 1, a = 1;
  union ShapeData {
    struct { float w, h; } rect;
    struct { float radius; } circle;
    // src = (x, y), dst absolute
    struct { float x2, y2, thickness; } line;
    // centers relative to (x, y)
    struct { float x0, y0, x1, y1, radius; } capsule;
  } u;
};

static_assert(std::is_standard_layout_v<ShapeDesc>);
static_assert(std::is_standard_layout_v<ShapeDesc::ShapeData>);
