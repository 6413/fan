#include <FAN/graphics.hpp>

int main() {

    fan_2d::square s(window_size / 2, 100, 1);

    fan_2d::square center(fan::vec2(), fan::vec2(5), fan::color(1, 0, 0));
    center.set_position(s.center() - center.get_size() / 2);

    window_loop(fan::color(0), [&] {
        s.draw();
        center.draw();
    });
    return 0;
}