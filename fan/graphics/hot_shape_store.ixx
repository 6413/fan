// Generic host-side shape storage for hot-reloadable game code.
//
// Game modules speak plain data (ShapeDesc, see shape_desc.h); this store
// owns the real fan shapes in the host process, keyed by game-chosen
// stable ids. One map covers every shape type, so the ABI never grows:
// adding a type only extends the creation switch below.
//
// New types are additive (new ShapeType + union member + case here);
// existing game DLLs keep working.

module;

#include <fan/graphics/shape_desc.h>

export module fan.graphics.hot_shape_store;

import std;
import fan.types.vector;
import fan.types.color;
import fan.graphics.shapes;
import fan.graphics;

export namespace fan::graphics {
struct hot_shape_store_t {
  // Create on first sight, update after. Unknown types are ignored.
  void upsert(const ShapeDesc& d) {
    auto it = m.find(d.id);
    if (it != m.end() && !structural_same(it->second.desc, d)) {
      m.erase(it);
      it = m.end();
    }
    if (it == m.end()) {
      auto s = make(d);
      if (!s) {
        return;
      }
      it = m.emplace(d.id, entry_t{std::move(*s), d}).first;
    }
    it->second.desc = d;
    apply(it->second.shape, d);
  }

  void drop(int id) {
    m.erase(id);
  }

  void clear() {
    m.clear();
  }

  std::size_t size() const {
    return m.size();
  }

private:
  struct entry_t {
    shape_t shape;
    ShapeDesc desc;
  };

  static std::optional<shape_t> make(const ShapeDesc& d) {
    fan::vec3 pos(d.x, d.y, d.z);
    fan::color col(d.r, d.g, d.b, d.a);
    switch (d.type) {
      case SHAPE_RECT:
        return shape_t(rectangle_t(pos, fan::vec2(d.u.rect.w, d.u.rect.h), col));
      case SHAPE_CIRCLE:
        return shape_t(circle_t(pos, d.u.circle.radius, col));
      case SHAPE_LINE:
        return shape_t(line_t(pos, fan::vec3(d.u.line.x2, d.u.line.y2, d.z), col, d.u.line.thickness));
      case SHAPE_CAPSULE:
        return shape_t(capsule_t(pos,
          fan::vec2(d.u.capsule.x0, d.u.capsule.y0),
          fan::vec2(d.u.capsule.x1, d.u.capsule.y1),
          d.u.capsule.radius, col));
      default:
        return std::nullopt;
    }
  }

  // Types without generic setters (line endpoints, capsule centers)
  // recreate when their structural params change.
  static bool structural_same(const ShapeDesc& a, const ShapeDesc& b) {
    if (a.type != b.type) {
      return false;
    }
    if (a.type == SHAPE_LINE) {
      return a.u.line.x2 == b.u.line.x2
        && a.u.line.y2 == b.u.line.y2
        && a.u.line.thickness == b.u.line.thickness;
    }
    if (a.type == SHAPE_CAPSULE) {
      return a.u.capsule.x0 == b.u.capsule.x0
        && a.u.capsule.y0 == b.u.capsule.y0
        && a.u.capsule.x1 == b.u.capsule.x1
        && a.u.capsule.y1 == b.u.capsule.y1
        && a.u.capsule.radius == b.u.capsule.radius;
    }
    return true;
  }

  static void apply(shape_t& s, const ShapeDesc& d) {
    s.set_position(fan::vec3(d.x, d.y, d.z));
    s.set_color(fan::color(d.r, d.g, d.b, d.a));
    switch (d.type) {
      case SHAPE_RECT:
        s.set_size(fan::vec2(d.u.rect.w, d.u.rect.h));
        break;
      case SHAPE_CIRCLE:
        s.set_radius(d.u.circle.radius);
        break;
      default:
        break;
    }
  }

  std::unordered_map<int, entry_t> m;
};
}
