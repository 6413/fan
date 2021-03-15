#include <fan/graphics/camera.hpp>

fan::camera::camera(fan::window* window) : m_window(window), m_yaw(0), m_pitch(0) {
	this->update_view();
}

fan::camera::camera(const fan::camera& camera)
	: m_window(camera.m_window)
{
	this->operator=(camera);
}

fan::camera::camera(fan::camera&& camera)
	: m_window(camera.m_window)
{
	this->operator=(std::move(camera));
}

fan::camera& fan::camera::operator=(const fan::camera& camera)
{
	this->m_front = camera.m_front;
	this->m_pitch = camera.m_pitch;
	this->m_position = camera.m_position;
	this->m_right = camera.m_right;
	this->m_up = camera.m_up;
	this->m_velocity = camera.m_velocity;
	this->m_window = camera.m_window;
	this->m_yaw = camera.m_yaw;

	return *this;
}

fan::camera& fan::camera::operator=(fan::camera&& camera) noexcept
{
	this->m_front = std::move(camera.m_front);
	this->m_pitch = std::move(camera.m_pitch);
	this->m_position = std::move(camera.m_position);
	this->m_right = std::move(camera.m_right);
	this->m_up = std::move(camera.m_up);
	this->m_velocity = std::move(camera.m_velocity);
	this->m_window = std::move(camera.m_window);
	this->m_yaw = std::move(camera.m_yaw);

	return *this;
}

void fan::camera::move(f_t movement_speed, bool noclip, f_t friction)
{
	if (!noclip) {
		//if (fan::is_colliding) {
		this->m_velocity.x /= friction * m_window->get_delta_time() + 1;
		this->m_velocity.y /= friction * m_window->get_delta_time() + 1;
		//}
	}
	else {
		this->m_velocity /= friction * m_window->get_delta_time() + 1;
	}
	static constexpr auto minimum_velocity = 0.001;
	if (this->m_velocity.x < minimum_velocity && this->m_velocity.x > -minimum_velocity) {
		this->m_velocity.x = 0;
	}
	if (this->m_velocity.y < minimum_velocity && this->m_velocity.y > -minimum_velocity) {
		this->m_velocity.y = 0;
	}
	if (this->m_velocity.z < minimum_velocity && this->m_velocity.z > -minimum_velocity) {
		this->m_velocity.z = 0;
	}
	if (m_window->key_press(fan::input::key_w)) {
		this->m_velocity += this->m_front * (movement_speed * m_window->get_delta_time());
	}
	if (m_window->key_press(fan::input::key_s)) {
		this->m_velocity -= this->m_front * (movement_speed * m_window->get_delta_time());
	}
	if (m_window->key_press(fan::input::key_a)) {
		this->m_velocity -= this->m_right * (movement_speed * m_window->get_delta_time());
	}
	if (m_window->key_press(fan::input::key_d)) {
		this->m_velocity += this->m_right * (movement_speed * m_window->get_delta_time());
	}
	if (!noclip) {
		// is COLLIDING
		if (m_window->key_press(fan::input::key_space/*, true*/)) { // FIX THISSSSSS
			this->m_velocity.z += jump_force;
			//jumping = true;
		}
		else {
			//jumping = false;
		}
		this->m_velocity.z += -gravity * m_window->get_delta_time();
	}
	else {
		if (m_window->key_press(fan::input::key_space)) {
			this->m_velocity.y += movement_speed * m_window->get_delta_time();
		}
		// IS COLLIDING
		if (m_window->key_press(fan::input::key_left_shift)) {
			this->m_velocity.y -= movement_speed * m_window->get_delta_time();
		}
	}

	if (m_window->key_press(fan::input::key_left)) {
		this->set_yaw(this->get_yaw() - sensitivity * 5000 * m_window->get_delta_time());
	}
	if (m_window->key_press(fan::input::key_right)) {
		this->set_yaw(this->get_yaw() + sensitivity * 5000 * m_window->get_delta_time());
	}
	if (m_window->key_press(fan::input::key_up)) {
		this->set_pitch(this->get_pitch() + sensitivity * 5000 * m_window->get_delta_time());
	}
	if (m_window->key_press(fan::input::key_down)) {
		this->set_pitch(this->get_pitch() - sensitivity * 5000 * m_window->get_delta_time());
	}

	this->m_position += this->m_velocity * m_window->get_delta_time();
	this->update_view();
}

void fan::camera::rotate_camera(bool when) // this->updateCameraVectors(); move function updates
{
	if (when) {
		return;
	}

	f32_t xoffset = m_window->get_raw_mouse_offset().x;
	f32_t yoffset = -m_window->get_raw_mouse_offset().y;

	xoffset *= sensitivity;
	yoffset *= sensitivity;

	this->set_yaw(this->get_yaw() + xoffset);
	this->set_pitch(this->get_pitch() + yoffset);

	this->update_view();
}

fan::mat4 fan::camera::get_view_matrix() const {
	return fan::look_at_left<fan::mat4>(this->m_position, m_position + m_front, this->m_up);
}

fan::mat4 fan::camera::get_view_matrix(fan::mat4 m) const {
	//																	 to prevent extra trash in camera class
	return m * fan::look_at_left<fan::mat4>(this->m_position, this->m_position + m_front, this->world_up);
}

fan::vec3 fan::camera::get_position() const {
	return this->m_position;
}

void fan::camera::set_position(const fan::vec3& position) {
	this->m_position = position;
}

fan::vec3 fan::camera::get_velocity() const
{
	return fan::camera::m_velocity;
}

void fan::camera::set_velocity(const fan::vec3& velocity)
{
	fan::camera::m_velocity = velocity;
}

f_t fan::camera::get_yaw() const
{
	return this->m_yaw;
}

f_t fan::camera::get_pitch() const
{
	return this->m_pitch;
}

void fan::camera::set_yaw(f_t angle)
{
	this->m_yaw = angle;
	if (m_yaw > fan::camera::max_yaw) {
		m_yaw = -fan::camera::max_yaw;
	}
	if (m_yaw < -fan::camera::max_yaw) {
		m_yaw = fan::camera::max_yaw;
	}
}

void fan::camera::set_pitch(f_t angle)
{
	this->m_pitch = angle;
	if (this->m_pitch > fan::camera::max_pitch) {
		this->m_pitch = fan::camera::max_pitch;
	}
	if (this->m_pitch < -fan::camera::max_pitch) {
		this->m_pitch = -fan::camera::max_pitch;
	} 
}

void fan::camera::update_view() {
	this->m_front = fan_3d::normalize(fan::direction_vector<fan::vec3>(this->m_yaw, this->m_pitch));
	this->m_right = fan_3d::normalize(cross(this->world_up, this->m_front)); 
	this->m_up = fan_3d::normalize(cross(this->m_front, this->m_right));
}