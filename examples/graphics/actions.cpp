#include <fan/pch.h>

int main() {
  loco_t loco;

  loco.input_action.add({ fan::key_space }, "jump");
  loco.input_action.add_keycombo({ fan::key_space, fan::key_a }, "combo_test");


  loco.loop([&] {

    if (loco.input_action.is_active("jump")) {
      fan::print("jump");
    }
    if (loco.input_action.is_active("combo_test")) {
      fan::print("combo_test");
    }

  });
}