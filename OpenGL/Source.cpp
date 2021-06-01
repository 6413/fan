#include <fan/graphics/gui.hpp>

#include <fan/audio/audio.hpp>

int main() {

    fan::window window; 

    window.set_max_fps(fan::get_screen_refresh_rate());

    fan::camera camera(&window);

    window.add_keys_callback([&](uint16_t key) {
        switch (key) {
            case fan::key_up:
            {
                fan::print("up");
                break;
            }
            case fan::key_right:
            {
                fan::print("right");
                break;
            }
            case fan::key_down:
            {
                fan::print("down");
                break;
            }
            case fan::key_left:
            {
                fan::print("left");
                break;
            }
        }
    });

    window.set_text_callback([&](wchar_t key) {
        fan::wprint((int)key);
    });

    fan_2d::graphics::gui::text_renderer tr(&camera);

    tr.push_back(L"test", 0, fan::colors::white, 32);

    tr.set_text_color(0, 2, fan::colors::red);

    window.loop([&] {
       
        tr.draw();

    });

    return 0;
}