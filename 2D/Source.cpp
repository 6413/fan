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

    cursor_move_callback.add(std::bind(&Camera::rotate_camera, camera3d));

    glEnable(GL_CULL_FACE);

    TextRenderer r;

    while (!glfwWindowShouldClose(window)) {
        GetFps();

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera3d.move(noclip, 200);
        r.render("bbbbb", vec2(100, window_size.y / 2 + 200), 0.5, Color(1, 0, 0));
      //  r.render("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", vec2(100, window_size.y / 2+10), 1, Color(1, 0, 0));
      //  r.render("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", vec2(100, window_size.y / 2+20), 1, Color(1, 0, 0));
      //  r.render("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", vec2(100, window_size.y / 2+30), 1, Color(1, 0, 0));
      //  r.render("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", vec2(100, window_size.y / 2+40), 1, Color(1, 0, 0));

      /*  r.render("yofsdaasdfflsökdjasdlköfjasdölkjfdaskjafsdkfasdlöjafsdkjlafsdlalkfsjdhfadlsjhafdsl", vec2(0, window_size.y / 2+10), 0.4, Color(1, 0, 0));
        r.render("yofsdaasdfflsökdjasdlköfjasdölkjfdaskjafsdkfasdlöjafsdkjlafsdlalkfsjdhfadlsjhafdsl", vec2(0, window_size.y / 2+20), 0.4, Color(1, 0, 0));
        r.render("yofsdaasdfflsökdjasdlköfjasdölkjfdaskjafsdkfasdlöjafsdkjlafsdlalkfsjdhfadlsjhafdsl", vec2(0, window_size.y / 2+30), 0.4, Color(1, 0, 0));
        r.render("yofsdaasdfflsökdjasdlköfjasdölkjfdaskjafsdkfasdlöjafsdkjlafsdlalkfsjdhfadlsjhafdsl", vec2(0, window_size.y / 2+40), 0.4, Color(1, 0, 0));*/

       // sp.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}