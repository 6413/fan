#include <fan/graphics.hpp>

int main() {
    fan::window window("", fan::window::resolutions::r_1680x1050);

    window.vsync(0);

    window.add_key_callback(fan::key_escape, [&] {
        window.close();
    });

    fan::camera camera(window);

    window.loop(0, [&] {

        window.get_fps();

    });

}