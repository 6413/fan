#include <fan/graphics.hpp>

int main() {

    fan::window window("", fan::window::resolutions::r_1600x1024);

    window.vsync(0);

    window.add_key_callback(fan::key_escape, [&] {
        window.close();
    });

    fan::camera camera(window);

    fan_2d::gui::text_box tb(camera, " ", 16, 0, fan::colors::red, fan::vec2(20, 20));

    tb.on_touch([&] {
        tb.set_box_color(0, fan::colors::green);
    });

    tb.on_click([&] {
        tb.set_box_color(0, fan::colors::white);
    });

    tb.on_release([&] {
        tb.set_box_color(0, fan::colors::green);
    });

    tb.on_exit([&] {
        tb.set_box_color(0, fan::colors::red);
    });

    window.loop(0, [&] {

        window.get_fps();

        tb.draw();

    });

}