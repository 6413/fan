#include <FAN/graphics.hpp>

constexpr fan::vec2i map_size(300);
constexpr f_t triangle_size(10);
constexpr f_t player_height = 5;

void get_collision_point(const fan::vec2i& grid_position, const std::vector<fan::vec3>& vertices, bool& did_collide) {

	unsigned int grid_index = grid_position.y * map_size.x + grid_position.x;

	auto vdt = fan_3d::camera.get_velocity() * fan::delta_time;

	auto line = fan::da_t<f32_t, 2, 3>{ fan_3d::camera.get_position() - vdt, vdt };

	auto p = fan_3d::line_triangle_intersection(
		line,
		fan::da_t<f32_t, 3, 3>{
		vertices[1 + grid_index], 
			vertices[0 + grid_index], 
			vertices[map_size.x + grid_index]} + fan::da3(0, 0, player_height)
	);

	auto p2 = fan_3d::line_triangle_intersection(
		line,
		fan::da_t<f32_t, 3, 3>{
		vertices[1 + grid_index], 
		vertices[map_size.x + 1 + grid_index], 
		vertices[map_size.x + grid_index]} + fan::da3(0, 0, player_height)
	);

	if(p != INFINITY || p2 != INFINITY){
		fan::vec3 ppoint = (p != INFINITY ? p : p2) + fan::vec3(0, 0, 0.05);
		auto velocity = fan_3d::camera.get_velocity();
		fan_3d::camera.set_position(ppoint);
		fan_3d::camera.set_velocity(fan::vec3(velocity.x, velocity.y, fan_3d::camera.jumping ? velocity.z : 0));
		fan::is_colliding = true;
		did_collide = true;
	}
}

void terrain_collision(const std::vector<fan::vec3>& vertices) {
	bool did_collide = false;

		auto position = fan_3d::camera.get_position() - fan_3d::camera.get_velocity() * fan::delta_time;

		get_collision_point((position / triangle_size).floored(), vertices, did_collide);

		if (!did_collide) {
			d_grid_raycast_2d(fan::vec2(position.x, position.y), fan::vec2(fan_3d::camera.get_position().x, fan_3d::camera.get_position().y), r, triangle_size) {
				if (r.grid.x < 0 || r.grid.y < 0) {
					break;
				}
				if (r.grid.x >= map_size.x || r.grid.y >= map_size.y) {
					break;
				}
				get_collision_point(r.grid, vertices, did_collide);
				break;
			}
		}
		if (!did_collide) {
			fan::is_colliding = false;
		}
}

int main() {
	fan_3d::add_camera_rotation_callback();

	fan_2d::sprite crosshair("images/crosshair.png", fan::window_size / 2 - 2, 4);

	fan::callback::window_resize.add([&]() {
		crosshair.set_position(fan::window_size / 2 - 2);
	});

	fan_3d::terrain_generator tg("images/grass.jpg", 50, 0, map_size, triangle_size, 2);

	fan_3d::camera.set_position(fan::vec3(20, 20, 50));

	auto vertices = tg.get_vertices();

	fan::window_loop(fan::color::hex(0), [&] {
		fan_3d::camera.move(3000, false);

		terrain_collision(vertices);

		tg.draw();
		
		fan::gui_draw([&] {
			crosshair.draw();
		});
	});
	return 0;
}