#include <FAN/Graphics.hpp>

int main() {
    bool noclip = true;
    vec3& position = camera3d.position;
    key_callback.add(GLFW_KEY_LEFT_CONTROL, true, [&] {
        noclip = !noclip;
    });

    key_callback.add(GLFW_KEY_ESCAPE, true, [&] {
        glfwSetWindowShouldClose(window, true);
    });

    glfwSetWindowPos(window, window_size.x / 2, window_size.y / 2 - window_size.y / 4);

    float crosshair_size = 3;
    CircleVector crosshair(window_size / 2, crosshair_size, 60, Color(1, 1, 1));

    window_resize_callback.add([&] {
        crosshair.set_position(0, window_size / 2);
    });

    cursor_move_callback.add(rotate_camera);

    glEnable(GL_CULL_FACE);

    square_vector2d s(window_size / 2 - 50, vec2(100), Color(0, 1, 0));

    s.push_back(vec2(100, 100), vec2(10), Color(1, 0, 0));
    s.push_back(vec2(200, 200), vec2(20), Color(0, 0, 1));


    s.set_size(0, vec2(100));
    s.set_color(0, Color(1, 1, 1), false);
    s.set_color(1, Color(0.5, 0.5, 0.5), false);
    s.set_position(1, window_size - s.get_size(1));
    s.get_position(1).print();


    while (!glfwWindowShouldClose(window)) {
        GetFps();

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        move_camera(noclip, 2000);

        s.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}