#include fan_pch

int main() {
  loco_t loco;

  static constexpr int wall_count = 4;
  fan::graphics::collider_static_t walls[wall_count];
  for (int i = 0; i < wall_count; ++i) {
    f32_t angle = 2 * fan::math::pi * i / wall_count;
    static constexpr int outline = 1.1;
    int x = std::cos(angle) / outline;
    int y = std::sin(angle) / outline;
    walls[i] = fan::graphics::rectangle_t{{
      .position = fan::vec2(x, y),
      .size = fan::vec2(
        std::abs(x) == 0 ? 1 : 0.1,
        std::abs(y) == 0 ? 1 : 0.1
      ),
      .color = fan::colors::red / 2
    }};
  }


  std::vector<fan::graphics::collider_dynamic_t> balls;
  static constexpr int ball_count = 100;
  balls.reserve(ball_count);
  for (int i = 0; i < ball_count; ++i) {
    balls.push_back(fan::graphics::circle_t{{
        .position = fan::vec3(fan::random::vec2(-0.8, 0.8), i),
        .radius = 0.05,
        .color = fan::random::color(),
        .blending = true
    }});
    balls.back().set_velocity(fan::random::vec2_direction(-1, 1) * 2);
  }

  fan::graphics::bcol.PreSolve_Shape_cb = [](
    bcol_t* bcol,
    const bcol_t::ShapeInfoPack_t* sip0,
    const bcol_t::ShapeInfoPack_t* sip1,
    bcol_t::Contact_Shape_t* Contact
    ) {
      if (sip0->ShapeEnum == sip1->ShapeEnum) {
        bcol->Contact_Shape_DisableContact(Contact);
        return;
      }

      fan::vec2 velocity = bcol->GetObject_Velocity(sip0->ObjectID);
      fan::vec2 p0 = bcol->GetObject_Position(sip0->ObjectID);
      fan::vec2 p1 = bcol->GetObject_Position(sip1->ObjectID);
      fan::vec2 wall_size = bcol->GetShape_Rectangle_Size(bcol->GetObjectExtraData(sip1->ObjectID)->shape_id);

      fan::vec2 reflection = fan::math::reflection_no_rot(velocity, p0, p1, wall_size);
      
      auto nr = gloco->m_write_queue.write_queue.NewNodeFirst();
      gloco->m_write_queue.write_queue[nr].cb = [oid = sip0->ObjectID, reflection] {
        fan::graphics::bcol.SetObject_Velocity(oid, reflection);
      };
  };

  loco.set_vsync(false);

  loco.loop([&] {
    for (auto& i : balls) {
      i.set_position(i.get_collider_position());
    }
  });
}