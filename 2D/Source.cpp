#ifdef FAN_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <FAN/Graphics.hpp>
#include <GLFW/glfw3native.h>
#include <cstdint>

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

    fan_3d::square_vector s("sides_05.png", 32);
    fan_3d::line_vector l(mat2x3(vec3(), vec3(100)), Color(1, 0, 0));

    l.push_back(mat2x3(vec3(), vec3(-100)), Color(0, 0, 1));

    s.push_back(vec3(2, 1, 1), vec3(0.5), vec2());
    s.push_back(vec3(0, 0, 3), vec3(0.5), vec2(0, 1));

    s.set_texture(0, vec2(1, 0));

    fan_gui::text_button::text_button_vector b(L"x+", vec2(100, 0), Color(1, 0, 0), 32, vec2(100));

    b.add(L"x-", b.get_position(0) + vec2(0, b.get_size(0).y * 2), Color(1, 0, 0), 32, vec2(100));
    b.add(L"z-", b.get_position(1) - vec2(b.get_size(1).x,  b.get_size(0).y), Color(1, 0, 0), 32, vec2(100));
    b.add(L"z+", b.get_position(1) + vec2(b.get_size(1).x, -b.get_size(0).y), Color(1, 0, 0), 32, vec2(100));

    fan_3d::model m("models/untitled.obj", vec3(2, 0.5, 1), vec3(1));

    bool allow_mouse = false;

    callbacks::key_callback.add(GLFW_KEY_R, true, [&] {
        allow_mouse = !allow_mouse;
        glfwSetInputMode(window, GLFW_CURSOR, allow_mouse ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
    });

    callbacks::cursor_move_callback.add(std::bind(&Camera::rotate_camera, fan_3d::camera, allow_mouse));

    b.set_press_callback(GLFW_MOUSE_BUTTON_LEFT, [&] {
        if (!allow_mouse) {
            return;
        }
        if (b.inside(0)) {
            m.set_position(m.get_position() + vec3(1, 0, 0) / 2.f);
            return;
        }
        if (b.inside(1)) {
            m.set_position(m.get_position() - vec3(1, 0, 0) / 2.f);
            return;
        }
        if (b.inside(2)) {
            m.set_position(m.get_position() - vec3(0, 0, 1) / 2.f);
        }
        if (b.inside(3)) {
            m.set_position(m.get_position() + vec3(0, 0, 1) / 2.f);
        }
    });

    while (!glfwWindowShouldClose(window)) {
        GetFps(true, true);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!allow_mouse) {
            fan_3d::camera.move(true, 200);
        }

        m.draw();
        
        l.draw();

        s.draw();

        b.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();

    return 0;
}