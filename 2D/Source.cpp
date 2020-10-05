#include <FAN/Graphics.hpp>

int main() {
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    callback::cursor_move.add(std::bind(&Camera::rotate_camera, fan_3d::camera, 0));
    //square amount x * square amount y
    constexpr vec2i map_size(150, 150);
    constexpr vec2i vertices_size(map_size + 1);

    constexpr int triangle_size(10);

    fan_3d::terrain_generator tg(map_size);

    std::vector<fan_3d::triangle_vertices_t> vertices(vertices_size.x * vertices_size.y);
    std::vector<Color> colors(vertices_size.x * vertices_size.y);

    f_t y_off = 0;
    int index = 0;
    for (int j = 0; j < vertices_size.y * triangle_size; j += triangle_size) {
        f_t x_off = 0;
        for (int i = 0; i < vertices_size.x * triangle_size; i += triangle_size) {
            vertices[index] = vec3((float)i, (f_t)ValueNoise_2D(x_off, y_off) * 100, (float)j);
            colors[index] = Color(1, 1, 1);
            x_off += 10;
            index++;
        }
        y_off += 10;
    }

    tg.insert(vertices, colors);
    vertices.clear();
    colors.clear();

    fan_window_loop() {
        begin_render(Color::hex(0));
        fan_3d::camera.move(true, 1000);

        tg.draw();

        end_render();
    }
    return 0;
}