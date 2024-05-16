#include <fan/pch.h>

int main() {
  loco_t loco;

  loco.console.commands.add("cmd", [&](const fan::commands_t::arg_t& args) {
    fan::print("test");
  });

  loco.set_vsync(0);

  loco.toggle_console = true;

  loco.loop([&] {
    //loco.console.print(fan::random::string(32)+"\n");
    loco.console.print_colored(fan::random::string(32) + "\n", fan::colors::green);
  });
}