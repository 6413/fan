#include <FAN/Graphics.hpp>

using namespace fan_gui;

int main() {

    key_callback.add(GLFW_KEY_ESCAPE, true, [&] {
        glfwSetWindowShouldClose(window, true);
    });

    glfwSetWindowPos(window, window_size.x / 2, window_size.y / 2 - window_size.y / 4);

    button_vector b(vec2(100, window_size.y / 2), 200, Color(1, 0, 0));
   
    b.add(vec2(500, 500), Color(0, 0, 1));

    character_callback.add(b.get_character_callback(0), 0, 0);
    character_callback.add(b.get_character_callback(1), 1, 0);

    key_callback.add(GLFW_KEY_ENTER, true, b.get_newline_callback(), 0);
    key_callback.add(GLFW_KEY_ENTER, true, b.get_newline_callback(), 1);

    key_callback.add(GLFW_KEY_BACKSPACE, true, b.get_erase_callback(), 0);
    key_callback.add(GLFW_KEY_BACKSPACE, true, b.get_erase_callback(), 1);

    while (!glfwWindowShouldClose(window)) {
        GetFps();

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        b.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}