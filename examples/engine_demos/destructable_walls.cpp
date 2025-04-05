#include <fan/pch.h>

using namespace fan::graphics;

std::vector< physics::polygon_t> m_pieces;

void SortVerticesClockwise(std::vector<fan::vec2>& vertices, const fan::vec2& center);
void CreatePieceFromPoint(
  fan::vec2 vertices[4],
  const std::vector<fan::vec2>& points,
  size_t pointIndex,
  float density,
  float friction,
  float restitution,
  const fan::vec2& impactPoint,
  float impulseForce,
  engine_t& engine,
  const std::vector<fan::vec2>& existingCenters,
  float minDistance
);

bool isTooClose(const fan::vec2& newPoint, const std::vector<fan::vec2>& existingPoints, float minDistance) {
  for (const auto& point : existingPoints) {
    if ((newPoint - point).length() < minDistance) {
      return true;
    }
  }
  return false;
}

void GenerateWallPieces(fan::vec2 vertices[4], const fan::vec2& impactPoint, int numPieces,
  float density, float friction, float restitution, float impulseForce, engine_t& engine) {

  std::vector<fan::vec2> points;

  points.push_back(impactPoint);

  float minDistance = 0.01f;

  while (points.size() < numPieces) {
    fan::vec2 newPoint(
      fan::random::f32(std::min(vertices[0].x, vertices[2].x), std::max(vertices[0].x, vertices[2].x)),
      fan::random::f32(std::min(vertices[0].y, vertices[2].y), std::max(vertices[0].y, vertices[2].y))
    );

    if (!isTooClose(newPoint, points, minDistance)) {
      points.push_back(newPoint);
    }
  }

  std::vector<fan::vec2> existingCenters;

  for (size_t i = 0; i < points.size(); i++) {
    CreatePieceFromPoint(vertices, points, i, density, friction, restitution, impactPoint, impulseForce, engine, existingCenters, 10.0f);
  }

}

void CreatePieceFromPoint(
  fan::vec2 vertices[4],
  const std::vector<fan::vec2>& points,
  size_t pointIndex,
  float density,
  float friction,
  float restitution,
  const fan::vec2& impactPoint,
  float impulseForce,
  engine_t& engine,
  const std::vector<fan::vec2>& existingCenters,
  float minDistance
) {
  fan::vec2 center = points[pointIndex];
  for (const auto& existingCenter : existingCenters) {
    if ((center - existingCenter).length() < minDistance) {
      return;
    }
  }

  int numVertices = fan::random::value(3, 8);
  std::vector<fan::vec2> pieceVertices;

  float minX = std::min({ vertices[0].x, vertices[1].x, vertices[2].x, vertices[3].x });
  float maxX = std::max({ vertices[0].x, vertices[1].x, vertices[2].x, vertices[3].x });
  float minY = std::min({ vertices[0].y, vertices[1].y, vertices[2].y, vertices[3].y });
  float maxY = std::max({ vertices[0].y, vertices[1].y, vertices[2].y, vertices[3].y });
  float width = maxX - minX;
  float height = maxY - minY;

  // Generate vertices for the polygon
  for (int i = 0; i < numVertices; i++) {
    float angle = 2.0f * fan::math::pi * i / numVertices;
    angle += fan::random::f32(-0.2f, 0.2f);

    float maxRadiusX = cosf(angle) > 0 ? (maxX - center.x) / cosf(angle) : (minX - center.x) / cosf(angle);
    float maxRadiusY = sinf(angle) > 0 ? (maxY - center.y) / sinf(angle) : (minY - center.y) / sinf(angle);
    float maxRadius = std::min(fabsf(maxRadiusX), fabsf(maxRadiusY));
    float radius = fan::random::f32(0.3f * maxRadius, 0.8f * maxRadius)*6;

    fan::vec2 vertex(
      center.x + radius * cosf(angle),
      center.y + radius * sinf(angle)
    );
    vertex.x = std::max(minX, std::min(vertex.x, maxX));
    vertex.y = std::max(minY, std::min(vertex.y, maxY));
    pieceVertices.push_back(vertex);
  }

  SortVerticesClockwise(pieceVertices, center);

  auto c = fan::random::color();
  std::vector<vertex_t> localVertices;
  for (const auto& v : pieceVertices) {
    localVertices.push_back({
        .position = fan::physics::physics_to_render(v - center)/2,
        .color = c
      });
  }

  physics::polygon_t piece{ {
      .position = fan::physics::physics_to_render(center),
      .vertices = localVertices,
      .body_type = fan::physics::body_type_e::dynamic_body,
      .shape_properties = {
          .friction = friction,
          .density = density,
          .restitution = restitution
      }
  } };

  // Apply impulse from impact point
  fan::vec2 impulseDir = center - impactPoint;
  float impulseDirLength = sqrt(impulseDir.x * impulseDir.x + impulseDir.y * impulseDir.y);
  if (impulseDirLength > 0) {
    impulseDir = impulseDir / impulseDirLength;
    float distanceFactor = 1.0f / (1.0f + impulseDirLength);
    //  piece.apply_linear_impulse_center(impulseDir * impulseForce * distanceFactor);
   //   piece.apply_angular_impulse(fan::random::f32(-impulseForce, impulseForce));
  }

  // Add to pieces list for management
  m_pieces.push_back(std::move(piece));
}


void SortVerticesClockwise(std::vector<fan::vec2>& vertices, const fan::vec2& center) {
  std::sort(vertices.begin(), vertices.end(), [center](const fan::vec2& a, const fan::vec2& b) {
    float angleA = atan2f(a.y - center.y, a.x - center.x);
    float angleB = atan2f(b.y - center.y, b.x - center.x);
    return angleA < angleB;
    });
}

void ShatterWall(physics::rectangle_t& wall, const fan::vec2& impactPoint, int numPieces, float impulseForce, engine_t& engine) {

  fan::vec2 position = wall.get_physics_position();
  fan::vec2 size = fan::physics::render_to_physics(wall.get_size());
  float angle = wall.get_angle().z;


  float density = wall.get_density();
  float friction = wall.get_friction();
  float restitution = wall.get_restitution();

  fan::vec2 vertices[4];
  float hw = size.x / 2;
  float hh = size.y / 2;

  fan::vec2 localVertices[4] = {
    {-hw, -hh},
    {hw, -hh},
    {hw, hh},
    {-hw, hh}
  };

  float c = cosf(angle);
  float s = sinf(angle);
  for (int i = 0; i < 4; i++) {
    float rotX = localVertices[i].x * c - localVertices[i].y * s;
    float rotY = localVertices[i].x * s + localVertices[i].y * c;

    vertices[i].x = rotX + position.x;
    vertices[i].y = rotY + position.y;
  }

  wall.erase();

  GenerateWallPieces(vertices, impactPoint, numPieces, density, friction, restitution, impulseForce, engine);
}

struct pile_t {
  engine_t engine;
  pile_t() {
    b2World_SetPreSolveCallback(engine.physics_context, presolve_static, this);
  }
  static bool presolve_static(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, void* context) {
    pile_t* pile = static_cast<pile_t*>(context);
    return pile->presolve(shapeIdA, shapeIdB, manifold);
  }

  bool presolve(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold) {

    if (wall.is_valid() == false) {
      return true;
    }

    if (wall.get_shape_type() != engine_t::shape_type_t::rectangle) {
      return true;
    }

    gloco->single_queue.push_back([this, manifold, shapeIdA, shapeIdB] {
      if (wall.is_valid() == false) {
        return;
      }

      if (wall.get_shape_type() != engine_t::shape_type_t::rectangle) {
        return;
      }
      fan::print("A");
      assert(b2Shape_IsValid(shapeIdA));
      assert(b2Shape_IsValid(shapeIdB));

      float sign = 0.0f;
      b2ShapeId wallShapeId = wall.get_shape_id();
      b2BodyId colliderBodyId = b2_nullBodyId;

      b2Vec2 impactPoint;

      if (manifold->pointCount > 0) {
        fan::print(manifold->points, manifold->pointCount);
        impactPoint = manifold->points[0].point;

        ShatterWall(wall, impactPoint, 8, 5.0f, engine);
        // wall.erase();
      }
      });
    return true;
  }

  physics::rectangle_t wall{ {
    .position = fan::vec2(500, 500),
    .size = fan::vec2(50),
    .color = fan::colors::white,
    .angle = 0,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .shape_properties{.presolve_events = true}
  } };
};

int main() {
  pile_t pile;
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