#include <fan/graphics/camera.hpp>

fan::camera::camera() : m_yaw(0), m_pitch(0) {
	this->update_view();
}

//void fan::camera::rotate_camera(bool when)
//{
//	if (when) {
//		return;
//	}
//
//	f32_t xoffset = m_window->get_raw_mouse_offset().x;
//	f32_t yoffset = -m_window->get_raw_mouse_offset().y;
//
//	xoffset *= sensitivity;
//	yoffset *= sensitivity;
//
//	this->set_yaw(this->get_yaw() + xoffset);
//	this->set_pitch(this->get_pitch() + yoffset);
//
//	this->update_view();
//}

fan::mat4 fan::camera::get_view_matrix() const {
	return fan::math::look_at_left<fan::mat4>(this->m_position, m_position + m_front, this->m_up);
}

fan::mat4 fan::camera::get_view_matrix(const fan::mat4& m) const {
	return m * fan::math::look_at_left<fan::mat4>(this->m_position, this->m_position + m_front, this->world_up);
}

fan::vec3 fan::camera::get_position() const {
	return this->m_position;
}

void fan::camera::set_position(const fan::vec3& position) {
	this->m_position = position;
}

fan::vec3 fan::camera::get_front() const
{
	return this->m_front;
}

void fan::camera::set_front(const fan::vec3 front)
{
	this->m_front = front;
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
	this->m_front = fan_3d::math::normalize(fan::math::direction_vector<fan::vec3>(this->m_yaw, this->m_pitch));
	this->m_right = fan_3d::math::normalize(fan::math::cross(this->world_up, this->m_front)); 
	this->m_up = fan_3d::math::normalize(fan::math::cross(this->m_front, this->m_right));
}