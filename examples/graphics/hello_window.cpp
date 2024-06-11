#include <fan/pch.h>

int main() {
  loco_t loco;

  loco.set_vsync(0);

  glfwSwapInterval(0);

  while (1) {
    loco.get_fps();
    loco.window.handle_events();
    glfwSwapBuffers(loco.window.glfw_window);
  }

  /*loco.loop([&] {
    
  });*/
  return 0;
}