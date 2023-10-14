// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_post_process
#define loco_rectangle
#define loco_button
#include _FAN_PATH(graphics/loco.h)

int main() {

  loco_t loco;

  loco_t::viewport_divider_t vd(loco.get_window()->get_size());

  struct viewport_t {
    loco_t::viewport_divider_t::viewport_nr_t nr;
    loco_t::simple_rectangle_t rectangle;
  };

  std::vector<viewport_t> viewports;
 

  f32_t max_viewports = 1;

  auto it = loco.get_window()->add_buttons_callback([&](const auto& d) {
    if (d.button == fan::mouse_left && d.state == fan::mouse_state::press) {
      vd.division_direction = loco_t::viewport_divider_t::division_mode::right;
      viewports.push_back({ vd.add() });
      //for (int i = 0; i < max_viewports; ++i) {
      //  viewports.push_back({ vd.add() });
      //  for (int j = 0; j < viewports.size(); ++j) {
      //    auto viewport = vd.get(viewports[j].nr);
      //    // fan::print((int)viewports[j].nr.NRI, viewport.position, viewport.size);
      //  }
      //  //fan::print("\n\n");
      //  //vd.division_direction = (loco_t::viewport_divider_t::division_mode)(fan::random::value_i64(0, 1));
      //}
      for (int i = 0; i < viewports.size(); ++i) {
        auto viewport = vd.get(viewports[i].nr);

        viewports[i].rectangle = loco_t::simple_rectangle_t{ {
            .position = fan::vec3(viewport.position * 2 - 1/*(1.f / viewports.size()) * i * 2 - 1 + (1.f / viewports.size()), 0*/, i),
            .size = viewport.size * 2, // because coordinate is -1 -> 1 so * 2 when viewport size is 0-1
            .color = /*fan::random::color() - fan::vec4(0, 0, 0, 0.5)*/
              fan::color::hsv(i * 10, 100, 100),
            .blending = true
        } };
      }
    }
  });

  loco.get_window()->m_buttons_callback[it].data({.button = fan::mouse_left, .state = fan::mouse_state::press});

  loco.loop([&] {

  });

  return 0;
}
