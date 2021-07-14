#pragma once

#include <fan/types/types.hpp>
#include <fan/types/matrix.hpp>
#include <fan/window/window.hpp>

namespace fan {
	class window;

	class camera {
	public:
		camera(fan::window* window);

		camera(const fan::camera& camera);
		camera(fan::camera&& camera);

		fan::camera& operator=(const fan::camera& camera);
		fan::camera& operator=(fan::camera&& camera) noexcept;

		void move(f_t movement_speed, bool noclip = true, f_t friction = 12);
		void rotate_camera(bool when);

		fan::mat4 get_view_matrix() const;
		fan::mat4 get_view_matrix(const fan::mat4& m) const;

		fan::vec3 get_position() const;
		void set_position(const fan::vec3& position);

		fan::vec3 get_front() const;
		void set_front(const fan::vec3 front);

		fan::vec3 get_velocity() const;
		void set_velocity(const fan::vec3& velocity);

		f_t get_yaw() const;
		void set_yaw(f_t angle);

		f_t get_pitch() const;
		void set_pitch(f_t angle);

		bool first_movement = true;

		void update_view();

		static constexpr f_t sensitivity = 0.1;

		static constexpr f_t max_yaw = 180;
		static constexpr f_t max_pitch = 89;

		static constexpr f_t gravity = 500;
		static constexpr f_t jump_force = 100;

		static constexpr fan::vec3 world_up = fan::vec3(0, 1, 0);

		fan::window* m_window;

	protected:

		fan::vec3 m_position;
		fan::vec3 m_front;

		f32_t m_yaw;
		f32_t m_pitch;
		fan::vec3 m_right;
		fan::vec3 m_up;
		fan::vec3 m_velocity;

	};
}