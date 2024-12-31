// old
#include <fan/pch.h>

int main() {
    loco_t loco;

    fan::vec2 window_size = loco.window.get_size();

    fan::vec2 platform_size{window_size.x / 3, 10};

    fan::graphics::collider_static_t walls[4];
    // floor
    walls[0] = fan::graphics::rectangle_t{{
        .position = fan::vec2(window_size.x / 2, window_size.y),
        .size = fan::vec2(window_size.x, platform_size.y / 2),
        .color = fan::colors::green
    }};
    // ceiling
    walls[1] = fan::graphics::rectangle_t{{
        .position = fan::vec2(window_size.x / 2, 0),
        .size = fan::vec2(window_size.x, platform_size.y / 2),
        .color = fan::colors::green
    }};
    // left
    walls[2] = fan::graphics::rectangle_t{{
        .position = fan::vec2(0, window_size.y / 2),
        .size = fan::vec2(platform_size.y, window_size.y),
        .color = fan::colors::green
    }};
    // right
    walls[3] = fan::graphics::rectangle_t{{
        .position = fan::vec2(window_size.x, window_size.y / 2),
        .size = fan::vec2(platform_size.y, window_size.y),
        .color = fan::colors::green
    }};

    fan::graphics::collider_static_t platforms[3];
    // left
    platforms[0] = fan::graphics::rectangle_t{{
        .position = fan::vec2(platform_size.x / 2, window_size.y / 1.5),
        .size = platform_size / 2,
        .color = fan::colors::blue
    }};
    // center
    platforms[1] = fan::graphics::rectangle_t{{
        .position = fan::vec2(window_size.x / 2, window_size.y / 2.5),
        .size = platform_size / 2,
        .color = fan::colors::blue
    }};
    //right
    platforms[2] = fan::graphics::rectangle_t{{
        .position = fan::vec2(window_size.x - platform_size.x / 2, window_size.y / 1.5),
        .size = platform_size / 2,
        .color = fan::colors::blue
    }};

    fan::graphics::collider_dynamic_t player = fan::graphics::rectangle_t{{
        .position = fan::vec2(window_size.x / 2, window_size.y / 2),
        .size = fan::vec2(10, 10),
        .color = fan::colors::red
    }};
    
    fan::graphics::set_gravity(fan::vec2(0, 2));
    fan::graphics::set_bump_friction(100);
    //fan::graphics::set_constant_friction(10);

    loco.input_action.add({ fan::key_space }, "jump");


    loco.loop([&]{
        player.move(fan::vec2(10, 0));

        if (loco.input_action.is_active("jump")) {
            fan::vec2 vel = player.get_velocity();
            player.set_velocity(fan::vec2(vel.x, -400));
        }

        fan::vec2 vel = player.get_velocity();
        fan::graphics::bcol_update.push_back([&, vel]{
            player.set_velocity(fan::vec2(vel.x / 1.01, vel.y));
        });
    });
}