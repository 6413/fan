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

    cursor_move_callback.add(rotate_camera);

    glEnable(GL_CULL_FACE);

    square_vector2d s(window_size / 2, vec2(100), Color(0, 0, 1));

    SquareVector3D test("sides_05.png");

    test.push_back(vec3(1, 0, 0), vec3(1), vec2());

    while (!glfwWindowShouldClose(window)) {
        GetFps();

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        move_camera(noclip, 200);

        test.draw();

        s.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}