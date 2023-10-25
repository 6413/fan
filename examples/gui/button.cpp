#include fan_pch

int main() {
  loco_t loco;

  auto mouse_move_cb = [] (const loco_t::mouse_move_data_t& mm_d) -> int {
    //fan::print(mm_d.position, (int)mm_d.mouse_stage);
    return 0;
  };
  auto mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> int {
    if (ii_d.button_state != fan::mouse_state::press) {
      return 0;
    }
    fan::print("clicked");
    return 0;
  };

  loco_t::theme_t theme = loco_t::themes::gray();

  fan::graphics::button_t button0{{
    .theme = &theme,
    .position = 0,
    .size = fan::vec2(0.3, 0.1),
    .text = " button",
    .mouse_move_cb = mouse_move_cb,
    .mouse_button_cb = mouse_button_cb
  }};

  loco.loop([&] {

  });

  return 0;
}
