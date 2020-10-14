#include <FAN/graphics.hpp>

int main() {

	fan_3d::add_camera_rotation_callback();

	fan_3d::square_vector s("sides_05.png", 32);

	s.push_back(fan::vec3(2, 0, 0), 2, fan::vec2(0, 1));

	fan::da_t<f32_t, 2, 3> a{ fan::da_t<f32_t, 3>{1, 2, 3}, fan::da_t<f32_t, 3>{ 4, 5, 6 } };
	fan::da_t<f32_t, 2, 3> b = a;

	fan::window_loop(fan::color::hex(0x87ceeb), [&] {

		fan_3d::camera.move(500);

		auto corners = s.get_corners(0);


		LOG(fan_3d::line_plane_intersection(
			fan::da_t<f32_t, 2, 3>{
				fan::da3(fan_3d::camera.get_position()) + fan::da_t<f_t, 3>{ 0, 0, 0 },
				fan::da3(fan_3d::camera.get_position()) + fan::da_t<f_t, 3>{ 100, 0, 0 }
		},
			fan::da_t<f_t, 4, 3>{
				corners.front.top_left,
				corners.front.top_right,
				corners.front.bottom_left,
				corners.front.bottom_right
			}
		));


		s.draw();

	});

	return 0;
}