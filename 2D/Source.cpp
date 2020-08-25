#include <FAN/Graphics.hpp>

int main() {
    fan_2d::square player(window_size / 2, vec2(50), Color(0, 0, 1));
    fan_2d::square_vector walls;

    walls.push_back(vec2(35, window_size.y / 2), vec2(50, 50), Color(1, 0, 0));

    walls.push_back(vec2(25, window_size.y / 2), vec2(50, window_size.y), Color(1, 0, 0));
    walls.push_back(vec2(window_size.x - 25, window_size.y / 2), vec2(50, window_size.y), Color(1, 0, 0));

    walls.push_back(vec2(window_size.x / 2, window_size.y - 25), vec2(window_size.x, 50), Color(1, 0, 0));
    walls.push_back(vec2(window_size.x / 2, 25), vec2(window_size.x, 50), Color(1, 0, 0));

    walls.push_back(window_size / 2 + vec2(100, 0), 10, Color(1, 0, 0));
    walls.push_back(window_size / 2 + vec2(100, 10 + player.get_size().y), 10, Color(1, 0, 0));

    fan_2d::line l;

    l.set_color(Color(0, 1, 0));

    while (!glfwWindowShouldClose(window)) {
        begin_render(0);

        l.set_position(mat2x2(player.get_position(), player.get_position() + player.get_velocity() * 10000));

        vec2 old_position = player.get_position();

        player.move(1000.f, 0);

        vec2 new_position = player.get_position();

        auto result = rectangle_collision_2d(old_position, new_position, player.get_size(), player.get_velocity(), walls);

        if (colliding(result)) {
            player.set_position(result.position);
            player.set_velocity(result.velocity);
        }

        walls.draw();
        player.draw();
        l.draw();

        end_render();
    }

    glfwTerminate();
    return 0;
}