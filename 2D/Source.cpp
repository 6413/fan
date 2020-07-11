#ifdef FAN_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <FAN/Graphics.hpp>
#include <GLFW/glfw3native.h>

using namespace fan_gui;
using namespace fan_gui::text_button;

int main() {

    callbacks::key_callback.add(GLFW_KEY_ESCAPE, true, [&] {
        glfwSetWindowShouldClose(window, true);
    });

#ifdef FAN_WINDOWS
#ifdef FAN_CUSTOM_WINDOW
    SetWindowLong(glfwGetWin32Window(window), GWL_STYLE, WS_SIZEBOX);
    ShowWindow(glfwGetWin32Window(window), SW_SHOW);
#endif
#endif

    fan_2d::line line;
    fan_2d::square square;

    line.set_color(Color(0, 0, 1));
    square.set_color(Color(1, 0, 0));
    square.set_size(vec2(100, 100));
    
    while (!glfwWindowShouldClose(window)) {
        GetFps(true, true);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        square.set_position(cursor_position);
        line.set_position(vec2(), cursor_position);

        square.draw();
        line.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();

    return 0;
}