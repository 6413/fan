#include <fan/pch.h>
#include <fan/system.h>

int main() {

  std::vector<loco_t*> locos;
  for (int i = 0; i < 2; ++i) {
    locos.push_back(new loco_t{ {.window_size = 400} });
  }

  static constexpr int wall_count = 4;
  fan::graphics::collider_static_t walls[wall_count];
  for (int i = 0; i < wall_count; ++i) {
    f32_t angle = 2 * fan::math::pi * i / wall_count;
    static constexpr int outline = 1;
    int x = std::cos(angle) / outline;
    int y = std::sin(angle) / outline;
    walls[i] = fan::graphics::rectangle_t{ {
      .position = fan::vec2(x, y),
      .size = fan::vec2(
        std::abs(x) == 0 ? 1 : 0.001,
        std::abs(y) == 0 ? 1 : 0.001
      ),
      .color = fan::colors::red / 2
    } };
  }

  fan::vec2 screen_size = fan::sys::get_screen_resolution();
  auto screen_to_ndc = [&](const fan::vec2& screen_pos) {
    return screen_pos / screen_size * 2 - 1;
    };

  auto ndc_to_screen = [&](const fan::vec2& ndc_position) {
    fan::vec2 normalized_position = (ndc_position + 1) / 2;
    return normalized_position * screen_size;
    };

  std::vector<fan::graphics::collider_dynamic_hidden_t> balls;
  balls.reserve(locos.size());

  std::vector<loco_t::image_t*> images;
  std::vector<loco_t::shape_t> sprites;

  for (int i = 0; i < 2; ++i) {
    balls.push_back(fan::graphics::collider_dynamic_hidden_t(0, fan::vec2(locos[i]->window.get_size().y / screen_size.x, locos[i]->window.get_size().x / screen_size.y)));
    balls.back().set_velocity(fan::random::vec2_direction(-1, 1) / 2);
    locos[i]->use();
    images.push_back(new loco_t::image_t(locos[i]->image_load("images/folder.webp")));
    sprites.push_back(fan::graphics::sprite_t{ {
  .position = fan::vec3(200, 200, 0),
  .size = 50,
  .image = *images.back()
} });

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

      gloco->single_queue.push_back([oid = sip0->ObjectID, reflection] {
        fan::graphics::bcol.SetObject_Velocity(oid, reflection);
      });
    };

  f32_t angle = 0, angle2 = 0;
  while (1) {

    int index = 0;
    for (auto& i : balls) {
      fan::vec2 p = i.get_collider_position();
      fan::vec2 scrn = ndc_to_screen(p);

      locos[index]->window.set_position(scrn - locos[index]->window.get_size() / 2);
   //   sprites[index].set_angle(fan::vec3(0, 0, index == 0 ? angle : angle2));
      index++;
    }

    angle += locos[0]->delta_time;
    angle2 += locos[0]->delta_time * 5;
    for (auto& i : locos) {
      i->use();
      i->process_loop([] {});
    }
    locos[0]->get_fps();
  }

}