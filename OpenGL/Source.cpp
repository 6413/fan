#include <fan/engine/graphics/graphics.hpp>

int main() {

	fan_2d::engine::engine_t e(fan::vec2(0, 30));

	e.window.set_size(fan::window::resolutions::r_1280x900);

	fan_2d::engine::rectangle car_body(&e);

	fan::vec2 car_position = fan::vec2(500, 400);
	fan::vec2 car_size = fan::vec2(300, 150);

	car_body.push_back(car_position, car_size, fan::colors::red, fan_2d::physics::body_type::dynamic_body);

	fan_2d::engine::motor motor(&e);

	fan::vec2 wheel_size(200, 10);

	fan_2d::engine::circle balls(&e);

	for (int i = 0; i < 50; i++) {
		balls.push_back(i + 300, 30, fan::colors::yellow, fan_2d::physics::body_type::dynamic_body);
	}

	fan_2d::engine::rectangle wheels(&e);

	wheels.push_back(car_position - fan::vec2(wheel_size.x / 2, wheel_size.y / 2), wheel_size, fan::colors::white, fan_2d::physics::body_type::dynamic_body);
	wheels.push_back(car_position + fan::vec2(car_size.x, 0) - fan::vec2(wheel_size.x / 2, wheel_size.y / 2), wheel_size, fan::colors::white, fan_2d::physics::body_type::dynamic_body);

	wheels.push_back(car_position + fan::vec2(0, car_size.y) - fan::vec2(wheel_size.x / 2, wheel_size.y / 2), wheel_size, fan::colors::white, fan_2d::physics::body_type::dynamic_body);
	wheels.push_back(car_position + fan::vec2(car_size.x, car_size.y) - fan::vec2(wheel_size.x / 2, wheel_size.y / 2), wheel_size, fan::colors::white, fan_2d::physics::body_type::dynamic_body);

	for (int i = 0; i < wheels.size(); i++) {
		motor.push_back(car_body.get_body(0), wheels.get_body(i));
	}

	fan_2d::engine::rectangle ground(&e);

	ground.push_back(fan::vec2(200, 800), fan::vec2(800, 30), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(200, 50), fan::vec2(400, 30), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(0, 150), fan::vec2(300, 30), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(50, 150), fan::vec2(30, 300), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(75, 400), fan::vec2(30, 300), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(200, 600), fan::vec2(30, 300), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(950, 600), fan::vec2(30, 300), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(1090, 400), fan::vec2(30, 300), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(1100, 200), fan::vec2(30, 300), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(1000, 0), fan::vec2(30, 300), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(200, 800), fan::vec2(800, 30), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.push_back(fan::vec2(600, 50), fan::vec2(400, 30), fan::colors::green, fan_2d::physics::body_type::static_body);

	ground.set_rotation(0, fan::math::radians(10));

	ground.set_rotation(1, fan::math::radians(10));

	ground.set_rotation(2, fan::math::radians(45));
	ground.set_rotation(3, fan::math::radians(-10));
	ground.set_rotation(4, fan::math::radians(20));
	ground.set_rotation(5, fan::math::radians(45));

	ground.set_rotation(6, fan::math::radians(-45));

	ground.set_rotation(7, fan::math::radians(-20));

	ground.set_rotation(8, fan::math::radians(10));

	ground.set_rotation(9, fan::math::radians(45));

	ground.set_rotation(10, fan::math::radians(-10));

	ground.set_rotation(11, fan::math::radians(-10));

	constexpr f32_t speed = 100;

	uint32_t hz = fan::get_screen_refresh_rate();

	e.window.loop([&] {

		car_body.set_position(0, fan_2d::get_body_position(motor.get_body_a(0)));
		car_body.set_rotation(0, fan_2d::get_body_angle(motor.get_body_a(0)));

		car_body.draw();

		for (int i = 0; i < wheels.size(); i++) {
			wheels.set_position(i, fan_2d::get_body_position(motor.get_body_b(i)));
			wheels.set_rotation(i, fan_2d::get_body_angle(motor.get_body_b(i)));
		}

		wheels.draw();

		ground.draw();
		
		balls.update_position();

		balls.draw();

		if (e.window.key_press(fan::key_d)) {
			for (int i = 0; i < motor.size(); i++) {
				motor.set_speed(i, speed);
			}
		}
		else if (e.window.key_press(fan::key_a)) {
			for (int i = 0; i < motor.size(); i++) {
				motor.set_speed(i, -speed);
			}
		}
		else {
			for (int i = 0; i < motor.size(); i++) {
				motor.set_speed(i, 0);
			}
		}

		e.step(1.0 / hz);
	});

	return 0;
} 