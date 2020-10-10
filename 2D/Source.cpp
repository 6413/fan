#include <FAN/graphics.hpp>

int main() {

    fan_2d::square s(window_size / 2, 100, 1);

    window_loop(fan::color(0), [&] {
        s.draw();
        s.set_position(cursor_position - s.get_size() / 2);
    });
    return 0;
}