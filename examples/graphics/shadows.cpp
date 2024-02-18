#include fan_pch


// only supports circle for now
// enable release build

int main() {

  loco_t loco;
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, loco.window.get_size().x),
    fan::vec2(0, loco.window.get_size().y)
  );

  loco.lighting.ambient = fan::vec3(0.1);
  loco_t::shapes_t::sprite_t::properties_t p;

  p.size = fan::vec2(1);

  static constexpr int wall_count = 4;
  fan::graphics::collider_static_t walls[wall_count];
  for (int i = 0; i < wall_count; ++i) {
    f32_t angle = 2 * fan::math::pi * i / wall_count;
    static constexpr int outline = 1.1;
    int x = std::cos(angle) / outline;
    int y = std::sin(angle) / outline;
    walls[i] = fan::graphics::rectangle_t{ {
      .position = fan::vec3((fan::vec2(1.0) + fan::vec2(x, y)) * fan::vec2(1000, 600), 100),
      .size = fan::vec2(
        std::abs(x) == 0 ? 1000 : 50,
        std::abs(y) == 0 ? 1000 : 50
      ),
      .color = fan::colors::red / 2
    } };
  }


  loco_t::image_t image;
  //image.load("images/brick.webp");
  image.create(fan::color(1, 1, 1), 1);
  p.image = &image;
  p.position = fan::vec3(loco.window.get_size() / 2, 0);
  p.size = loco.window.get_size() / 2;
  p.color.a = 1;
  p.blending = true;
  loco_t::shape_t s0 = p;

  //lp.size = 1000;
  //lp.color = fan::colors::green * 10;
  //loco_t::shape_t l1 = lp;

  std::vector<fan::vec2> positions;

  std::vector<fan::graphics::collider_dynamic_t> balls;
  std::vector<fan::graphics::collider_dynamic_t> lights;
  {
    static constexpr int ball_count = 50;
    balls.reserve(ball_count);
    for (int i = 0; i < ball_count; ++i) {
      positions.push_back(fan::random::vec2(fan::vec2(100), fan::vec2(2000, 1000)));
      balls.push_back(fan::graphics::circle_t{ {
          .position = fan::vec3(positions[i], i + 1),
          .radius = 25,
          .color = fan::random::color(),
          .blending = true,
      } });
      balls.back().set_velocity(fan::random::vec2_direction(-1000, 1000) * 50);
    }
  }
  {
    static constexpr int ball_count = 10;
    lights.reserve(ball_count);
    for (int i = 0; i < ball_count; ++i) {
      lights.push_back(fan::graphics::light_t{ {
          .position = fan::vec3(fan::random::vec2(fan::vec2(100), fan::vec2(2000, 1000)), 0),
          .size = 300,
          .color = fan::colors::white / 2,
          .blending = true,
      } });
      lights.back().set_velocity(fan::random::vec2_direction(-1000, 1000) * 200);
      lights.back().set_collider_size(10);
    }
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


  loco.loop([&] {
    loco.get_fps();

    int idx = 0;
    for (auto& i : balls) {
      positions[idx] = i.get_collider_position();
      i.set_position(i.get_collider_position());
      idx++;
    }

    for (auto& i : lights) {
      i.set_position(i.get_collider_position());
      idx++;
    }

    loco.shapes.light.m_current_shader->use();
    loco.shapes.light.m_current_shader->set_vec2_array("un_positions", positions);
  });

  return 0;
}