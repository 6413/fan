#include <fan/pch.h>

using namespace fan::graphics;

std::vector<physics::polygon_t> m_pieces;

/*
equal division
std::vector<physics::polygon_t> create_pieces(physics::rectangle_t& rectangle, int n) {
  std::vector<physics::polygon_t> pieces;
  pieces.reserve(n);
  
  fan::vec2 center = rectangle.get_position();
  fan::vec2 size = rectangle.get_size();
  f32_t angle = rectangle.get_angle().z;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  
  std::vector<fan::vec2> perimeter_points;
  perimeter_points.reserve(n);
  
  for (int i = 0; i < n; i++) {
    f32_t point_angle = (2 * fan::math::pi * i) / n;
    
    fan::vec2 dir(std::cos(point_angle), std::sin(point_angle));
    
    f32_t scale_x = (dir.x != 0) ? ((dir.x > 0 ? size.x : -size.x) / dir.x) : std::numeric_limits<f32_t>::max();
    f32_t scale_y = (dir.y != 0) ? ((dir.y > 0 ? size.y : -size.y) / dir.y) : std::numeric_limits<f32_t>::max();
    
    f32_t scale = std::min(std::abs(scale_x), std::abs(scale_y));
    
    fan::vec2 local_point = dir * scale;
    fan::vec2 world_point = center + local_point.rotate(angle);
    
    perimeter_points.push_back(world_point);
  }
  
  for (int i = 0; i < n; i++) {
    std::vector<fan::graphics::vertex_t> vertices;
    vertices.reserve(3);
    
    fan::color random_color = fan::random::color();
    
    fan::graphics::vertex_t center_vertex;
    center_vertex.position = fan::vec3(center.x, center.y, 0);
    center_vertex.color = random_color;
    vertices.push_back(center_vertex);
    
    int idx1 = i;
    int idx2 = (i + 1) % n;
    
    fan::graphics::vertex_t vertex1;
    vertex1.position = fan::vec3(perimeter_points[idx1].x, perimeter_points[idx1].y, 0);
    vertex1.color = random_color;
    vertices.push_back(vertex1);
    
    fan::graphics::vertex_t vertex2;
    vertex2.position = fan::vec3(perimeter_points[idx2].x, perimeter_points[idx2].y, 0);
    vertex2.color = random_color;
    vertices.push_back(vertex2);
    
    physics::polygon_t poly({{
      .vertices = vertices,
      .body_type = fan::physics::body_type_e::dynamic_body
    }});
    
    std::uniform_real_distribution<f32_t> vel_dist(-10.0f, 10.0f);
    poly.set_linear_velocity({vel_dist(gen), vel_dist(gen)});
    
    pieces.push_back(poly);
  }
  
  return pieces;
}
*/

std::vector<physics::polygon_t> create_pieces(physics::rectangle_t& rectangle, int n) {
  std::vector<physics::polygon_t> pieces;
  pieces.reserve(n);
  
  fan::vec2 center = rectangle.get_position();
  fan::vec2 size = rectangle.get_size();
  f32_t angle = rectangle.get_angle().z;

  std::vector<fan::vec2> perimeter_points;
  perimeter_points.reserve(n);
  
  for (int i = 0; i < n; i++) {
    f32_t point_angle = (2 * fan::math::pi * i) / n;

    point_angle += fan::random::value(-fan::math::pi / (2 * n), fan::math::pi / (2 * n));
    
    fan::vec2 dir(std::cos(point_angle), std::sin(point_angle));
    
    f32_t scale_x = (dir.x != 0) ? ((dir.x > 0 ? size.x : -size.x) / dir.x) : std::numeric_limits<f32_t>::max();
    f32_t scale_y = (dir.y != 0) ? ((dir.y > 0 ? size.y : -size.y) / dir.y) : std::numeric_limits<f32_t>::max();
    
    f32_t scale = std::min(std::abs(scale_x), std::abs(scale_y));
    
    fan::vec2 local_point = dir * scale;
    fan::vec2 world_point = center + local_point.rotate(angle);
    
    perimeter_points.push_back(world_point);
  }
  
  std::vector<fan::vec2> internal_points;
  internal_points.reserve(n);
  
  for (int i = 0; i < n; i++) {
    f32_t factor = fan::random::value(0.2f, 0.7f);
    
    fan::vec2 internal_pt = center + (perimeter_points[i] - center) * factor;
    internal_points.push_back(internal_pt);
  }
  
  for (int i = 0; i < n; i++) {
    std::vector<fan::graphics::vertex_t> vertices;
    
    fan::color random_color = rectangle.get_color();
    
    fan::graphics::vertex_t center_vertex;
    center_vertex.position = fan::vec3(center.x, center.y, 0);
    center_vertex.color = random_color;
    
    fan::graphics::vertex_t internal_vertex;
    internal_vertex.position = fan::vec3(internal_points[i].x, internal_points[i].y, 0);
    internal_vertex.color = random_color;
    
    int idx1 = i;
    int idx2 = (i + 1) % n;
    
    fan::graphics::vertex_t vertex1;
    vertex1.position = fan::vec3(perimeter_points[idx1].x, perimeter_points[idx1].y, 0);
    vertex1.color = random_color;
    
    fan::graphics::vertex_t vertex2;
    vertex2.position = fan::vec3(perimeter_points[idx2].x, perimeter_points[idx2].y, 0);
    vertex2.color = random_color;
    
    std::uniform_int_distribution<int> dist_pattern(0, 2);
    int pattern = fan::random::value(0, 2);

    if (pattern == 0) {
      vertices.push_back(internal_vertex);
      vertices.push_back(vertex1);
      vertices.push_back(vertex2);
    }
    else if (pattern == 1) {
      vertices.push_back(center_vertex);
      vertices.push_back(internal_vertex);
      vertices.push_back(vertex1);
      vertices.push_back(vertex2);
    }
    else {
      vertices.push_back(center_vertex);
      vertices.push_back(vertex1);
      vertices.push_back(internal_vertex);
      vertices.push_back(vertex2);
    }

    physics::polygon_t poly({ {
      .vertices = vertices,
      .body_type = fan::physics::body_type_e::dynamic_body
    } });

    poly.set_linear_velocity(rectangle.get_linear_velocity());
    poly.set_angular_velocity(rectangle.get_angular_velocity());

    pieces.push_back(poly);
  }

  return pieces;
}

struct pile_t {
  engine_t engine;
  pile_t() {
    b2World_SetPreSolveCallback(engine.physics_context, presolve_static, this);

    for (int i = 0; i < 20; ++i) {
      physics::rectangle_t wall{ {
        .position = fan::random::vec2(100, fan::vec2(engine.window.get_size().x / 1.2, 600)),
        .size = fan::vec2(50),
        .color = fan::colors::red,
        .angle = 0,
        .body_type = fan::physics::body_type_e::dynamic_body,
        .shape_properties{.presolve_events = true}
      } };

      walls[wall.get_shape_id()] = std::move(wall);
    }
  }
  static bool presolve_static(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, void* context) {
    pile_t* pile = static_cast<pile_t*>(context);
    return pile->presolve(shapeIdA, shapeIdB, manifold);
  }

  bool presolve(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold) {
    std::unordered_map<b2ShapeId, physics::rectangle_t>::iterator found;
    if (found = walls.find(shapeIdA); found != walls.end()) {
      if (found->second.is_valid() == false) {
        return true;
      }
    }
    else if (found = walls.find(shapeIdB); found == walls.end()) {
      if (found->second.is_valid() == false) {
        return true;
      }
    }
    auto& wall = found->second;
    if (wall.get_shape_type() != engine_t::shape_type_t::rectangle) {
      return true;
    }

    gloco->single_queue.push_back([this, &wall, manifold, shapeIdA, shapeIdB] {
      if (wall.is_valid() == false) {
        return;
      }
      if (wall.get_shape_type() != engine_t::shape_type_t::rectangle) {
        return;
      }

      if (!b2Shape_IsValid(shapeIdA)) {
        return;
        }
      if (!b2Shape_IsValid(shapeIdB)) {
        return;
      }

      b2ShapeId wallShapeId = wall.get_shape_id();
      b2BodyId colliderBodyId = b2_nullBodyId;
      fan::vec2 impact_point;

      if (manifold->pointCount > 0) {
        impact_point = manifold->points[0].point;

        const int num_pieces = 8;
        auto pieces = create_pieces(wall, num_pieces);
        m_pieces.insert(m_pieces.end(), pieces.begin(), pieces.end());

        wall.erase();
      }
    });

    return true;
  }
  struct key_hasher_t {
    std::size_t operator()(const b2ShapeId& k) const {
      using std::hash;
      return ((hash<int32_t>()(k.index1)
        ^ (hash<uint16_t>()(k.revision) << 1)) >> 1)
        ^ (hash<uint16_t>()(k.world0) << 1);
    }
  };
  inline static auto equal = [](const b2ShapeId& l, const b2ShapeId& r){return l.index1 == r.index1 && l.revision == r.revision && l.world0 == r.world0;};
  std::unordered_map<b2ShapeId, physics::rectangle_t, key_hasher_t, decltype(equal)> walls;
};

int main() {
  pile_t pile;
 // pile.engine.physics_context.set_gravity(0);
  //b2SetLengthUnitsPerMeter(1 / 512.f);
  b2World_SetContactTuning(pile.engine.physics_context, 0, 0, 10);
  fan::vec2 ws = window_size();

  physics::rectangle_t ground{ {
    .position = fan::vec2(ws.x / 2, ws.y - 10),
    .size = fan::vec2(ws.x / 2, 10),
    .color = fan::colors::green
  } };


  physics::mouse_joint_t joint;

  bool play_physics = false;
  fan::graphics::physics::debug_draw(false);
  fan_window_loop{
    if (ImGui::Button("play physics")) {
      play_physics = !play_physics;
    }
    if (play_physics) {
      physics::step(pile.engine.delta_time);
    }
  };
}