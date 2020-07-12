#ifdef FAN_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <FAN/Graphics.hpp>
#include <GLFW/glfw3native.h>

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

    callbacks::cursor_move_callback.add(std::bind(&Camera::rotate_camera, fan_3d::camera));

    fan_3d::square_vector s("sides_05.png", 32);
    fan_3d::line_vector l(mat2x3(vec3(), vec3(100)), Color(1, 0, 0));

    l.push_back(mat2x3(vec3(), vec3(-100)), Color(0, 0, 1));

    s.push_back(vec3(1, 0, 0), vec3(0.5), vec2());
    s.push_back(vec3(0, 0, 3), vec3(0.5), vec2(0, 1));

    s.set_texture(0, vec2(1, 0));

    while (!glfwWindowShouldClose(window)) {
        GetFps(true, true);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        fan_3d::camera.move(true, 200);

        l.draw();

        s.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();

    return 0;
}