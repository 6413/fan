#include <FAN/Graphics.hpp>

int main() {

    fan_2d::square_vector s;

    s.push_back(vec2(10, 10), vec2(10, 10), Color(1, 0, 0));
    s.push_back(vec2(100, 100), vec2(100, 100), Color(0, 0, 1));

    fan_window_loop() {
        begin_render(Color::rgb(0, 0, 0));

        s.draw(0);

        end_render();
    }

    return 0;
}