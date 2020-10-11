#include <FAN/graphics.hpp>

int main() {
    constexpr fan::vec2i map_size(150, 150); // squares
    constexpr uint32_t triangle_size(10);
    constexpr fan::vec2i mesh_size(5);

    fan_3d::terrain_generator tg("grass.jpg", map_size, triangle_size, mesh_size);

    callback::cursor_move.add(std::bind(&Camera::rotate_camera, fan_3d::camera, 0));

    window_loop(fan::color::hex(0x87ceeb), [&] {
        
        fan_3d::camera.move(true, 500);

        tg.draw();

    });
    return 0;
}