#include <fan/pch.h>

#include <fan/graphics/gui/model_maker/loader.h>

int main() {
  loco_t loco;

  loco_t::texturepack_t tp;
  tp.open_compiled("tp_controller");
  model_list_t m;

  model_list_t::cm_t cm;
  cm.import_from("fmm_controller.json", &tp);

  model_list_t::properties_t p;
  p.position = fan::vec3(fan::vec2(500), 5);

  auto model_id = m.push_model(&tp, &cm, p);

  loco.loop([&] {

    for (int i = 1; i < 17; ++i) {
      m.iterate(model_id, i, [&](model_list_t::group_data_t& group_data) {
        if (i < 13) {
          int key_state = 0;
          int key = fan::gamepad_a + (i - 1);
          // skip left and right thumb
          if (i >= 7) {
            key += 2;
          }
          key_state = loco.window.key_state(key);
          if (key_state == (int)fan::keyboard_state::press || key_state == (int)fan::keyboard_state::repeat) {
            group_data.shape.set_color(fan::color(0, 0, 0, 1));
          }
          else if (key_state == (int)fan::keyboard_state::release) {
            group_data.shape.set_color(1);
          }
        }
        else {
          int axis;
          if (i == 13) {
              axis = fan::gamepad_l2;
          } else if (i == 14) {
              axis = fan::gamepad_r2;
          } else if (i == 15) {
              axis = fan::gamepad_left_thumb;
          } else if (i == 16) {
              axis = fan::gamepad_right_thumb;
          }

          fan::vec2 v = loco.window.get_gamepad_axis(axis);
          fan::color c = fan::color(1) - ((i <= 14) ? ((v.x + 1) / 2) : v.length());
          c.a = 1;
          group_data.shape.set_color(c);

          if (i == 15) {
              static fan::vec2 initial_pos = group_data.shape.get_position();
              group_data.shape.set_position(initial_pos + v * 10);
          }
          else if (i == 16) {
            static fan::vec2 initial_pos = group_data.shape.get_position();
            group_data.shape.set_position(initial_pos + v * 10);
          }
        }
      });
    }
  });
}