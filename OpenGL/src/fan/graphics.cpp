#include <fan/graphics.hpp>

#include <functional>
#include <numeric>

#include <fan/fast_noise.hpp>
#include <fan/collision/rectangle.hpp>

fan::camera::camera(fan::window& window) : m_window(window), m_yaw(0), m_pitch(0) {
	this->update_view();
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

void fan::camera::move(f32_t movement_speed, bool noclip, f32_t friction)
{
	if (!noclip) {
		//if (fan::is_colliding) {
			this->m_velocity.x /= friction * m_window.get_delta_time() + 1;
			this->m_velocity.y /= friction * m_window.get_delta_time() + 1;
		//}
	}
	else {
		this->m_velocity /= friction * m_window.get_delta_time() + 1;
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
	if (m_window.key_press(fan::input::key_w)) {
		this->m_velocity += this->m_front * (movement_speed * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_s)) {
		this->m_velocity -= this->m_front * (movement_speed * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_a)) {
		this->m_velocity -= this->m_right * (movement_speed * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_d)) {
		this->m_velocity += this->m_right * (movement_speed * m_window.get_delta_time());
	}
	if (!noclip) {
		// is COLLIDING
		if (m_window.key_press(fan::input::key_space/*, true*/)) { // FIX THISSSSSS
			this->m_velocity.z += jump_force;
			//jumping = true;
		}
		else {
			//jumping = false;
		}
		this->m_velocity.z += -gravity * m_window.get_delta_time();
	}
	else {
		if (m_window.key_press(fan::input::key_space)) {
			this->m_velocity.y += movement_speed * m_window.get_delta_time();
		}
		// IS COLLIDING
		if (m_window.key_press(fan::input::key_left_shift)) {
			this->m_velocity.y -= movement_speed * m_window.get_delta_time();
		}
	}

	if (m_window.key_press(fan::input::key_left)) {
		this->set_yaw(this->get_yaw() - sensitivity * 5000 * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_right)) {
		this->set_yaw(this->get_yaw() + sensitivity * 5000 * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_up)) {
		this->set_pitch(this->get_pitch() + sensitivity * 5000 * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_down)) {
		this->set_pitch(this->get_pitch() - sensitivity * 5000 * m_window.get_delta_time());
	}

	this->m_position += this->m_velocity * m_window.get_delta_time();
	this->update_view();
}

void fan::camera::rotate_camera(bool when) // this->updateCameraVectors(); move function updates
{
	if (when) {
		return;
	}

	static f32_t lastX = 0, lastY = 0;

	f32_t xpos = m_window.get_mouse_position().x;
	f32_t ypos = m_window.get_mouse_position().y;

	f32_t xoffset = xpos - lastX;
	f32_t yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

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

f32_t fan::camera::get_yaw() const
{
	return this->m_yaw;
}

f32_t fan::camera::get_pitch() const
{
	return this->m_pitch;
}

void fan::camera::set_yaw(f32_t angle)
{
	this->m_yaw = angle;
	if (m_yaw > fan::camera::max_yaw) {
		m_yaw = -fan::camera::max_yaw;
	}
	if (m_yaw < -fan::camera::max_yaw) {
		m_yaw = fan::camera::max_yaw;
	}
}

void fan::camera::set_pitch(f32_t angle)
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

uint32_t load_texture(const std::string_view path, const std::string& directory, bool flip_image) {

	std::string file_name = std::string(directory + (directory.empty() ? "" : "/") + path.data());

	uint32_t texture_id = 0;

	fan_2d::load_image(texture_id, file_name, flip_image);

	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	return texture_id;
}

fan::mat4 fan_2d::get_projection(const fan::vec2i& window_size) {

	return fan::ortho<fan::mat4>((f_t)window_size.x / 2, window_size.x + (f_t)window_size.x * 0.5f, window_size.y + (f_t)window_size.y * 0.5f, (f_t)window_size.y / 2.f, 0.1f, 1000.0f);
}

fan::mat4 fan_2d::get_view_translation(const fan::vec2i& window_size, const fan::mat4& view)
{
	auto m = fan::translate(view, fan::vec3((f_t)window_size.x / 2, (f_t)window_size.y / 2, -700.0f));

	return m;
}

fan::vec2 fan_2d::move_object(fan::window& window, fan::vec2& position, fan::vec2& velocity, f32_t speed, f32_t gravity, f32_t jump_force, f32_t friction) {
	const f_t delta_time = window.get_delta_time();

	if (gravity != 0) {
		if (window.key_press(fan::input::key_space)) { // AND COLLIDING
			velocity.y = jump_force;
		}
		else {
			velocity.y += gravity * delta_time;
		}
	}

	speed *= 100;

	static constexpr auto minimum_velocity = 0.001;

	if (gravity != 0) {
		velocity.x /= friction * delta_time + 1;
	}
	else {
		velocity /= friction * delta_time + 1;
	}

	if (window.key_press(fan::input::key_w)) {
		velocity.y -= speed * delta_time;
	}
	if (window.key_press(fan::input::key_s)) {
		velocity.y += speed * delta_time;
	}
	if (window.key_press(fan::input::key_a)) {
		velocity.x -= speed * delta_time;
	}
	if (window.key_press(fan::input::key_d)) {
		velocity.x += speed * delta_time;
	}

	if (velocity.x < minimum_velocity && velocity.x > -minimum_velocity) {
		velocity.x = 0;
	}
	if (velocity.y < minimum_velocity && velocity.y > -minimum_velocity) {
		velocity.y = 0;
	}

	if constexpr (std::is_same<decltype(velocity.x), f32_t>::value) {
		if (velocity.x >= FLT_MAX) {
			velocity.x = FLT_MAX;
		}
		if (velocity.y >= FLT_MAX) {
			velocity.y = FLT_MAX;
		}
	}
	else {
		throw std::runtime_error("no double support");
	}

	position += velocity * delta_time;

	return velocity * delta_time;
}

fan_2d::line::line(fan::camera& camera) : 
	line::basic_shape(camera, fan::shader(shader_paths::single_shapes_vs, shader_paths::single_shapes_fs), fan::vec2(), fan::vec2()), 
	line::basic_shape_color_vector() {}

fan_2d::line::line(fan::camera& camera, const fan::mat2& begin_end, const fan::color& color)
	: line::basic_shape(camera, fan::shader(shader_paths::single_shapes_vs, shader_paths::single_shapes_fs), begin_end[0], begin_end[1]),
	  line::basic_shape_color_vector(color) {}

void fan_2d::line::draw()
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size());

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->m_shader.use();

	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_vec4("shape_color", get_color());
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::LINE));
	this->m_shader.set_vec2("begin", basic_shape::get_position());
	this->m_shader.set_vec2("end", get_size());

	line::basic_shape::basic_draw(GL_LINES, 2);
}

fan::mat2 fan_2d::line::get_position() const
{
	return fan::mat2(m_position, m_size);
}

void fan_2d::line::set_position(const fan::mat2& begin_end)
{
	line::basic_shape::set_position(fan::vec2(begin_end[0]));
	set_size(begin_end[1]);
}

fan_2d::rectangle::rectangle(fan::camera& camera)
	: basic_shape(
		camera,
		fan::shader(shader_paths::single_shapes_vs, shader_paths::single_shapes_fs),
		fan::vec2(),
		fan::vec2()
	), basic_shape_color_vector(), m_rotation(0) {
}

fan_2d::rectangle::rectangle(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color)
	: basic_shape(
		camera,
		fan::shader(shader_paths::single_shapes_vs, shader_paths::single_shapes_fs),
		position, size
	), basic_shape_color_vector(color), m_rotation(0) {}

fan_2d::rectangle::rectangle(const rectangle& single)
	:  rectangle::basic_shape(single), 
		rectangle::basic_shape_color_vector(single), 
		rectangle::basic_shape_velocity(single)
{
	m_rotation = single.m_rotation;
	m_corners = single.m_corners;
}

fan_2d::rectangle::rectangle(rectangle&& single) noexcept
	:  rectangle::basic_shape(std::move(single)), 
		rectangle::basic_shape_color_vector(std::move(single)), 
		rectangle::basic_shape_velocity(std::move(single))
{
	m_rotation = std::move(single.m_rotation);
	m_corners = std::move(single.m_corners);
}

fan_2d::rectangle_corners_t fan_2d::rectangle::get_corners() const
{
	if (this->m_rotation) {
		return this->m_corners;
	}
	return fan_2d::get_rectangle_corners(this->m_position, this->m_size);
}

fan::vec2 fan_2d::rectangle::get_center() const
{
	return fan_2d::rectangle::m_position + fan_2d::rectangle::m_size / 2;
}

f_t fan_2d::rectangle::get_rotation() const
{
	return m_rotation;
}

void fan_2d::rectangle::set_rotation(f_t angle)
{
	m_rotation = fmod(angle, 360);

	angle = fan::radians(-angle);

	constexpr f_t offset = 3 * fan::PI / 4;
	const fan::vec2 position(this->get_position());
	const fan::vec2 radius(this->get_size() / 2);

	f_t r = fan_2d::distance(position, position + radius);

	f_t x1 = r * cos(angle + offset - fan::PI * 3.0 / 2.0);
	f_t y1 = r * sin(angle + offset - fan::PI * 3.0 / 2.0);

	f_t x2 = r * cos(angle + offset - fan::PI);
	f_t y2 = r * sin(angle + offset - fan::PI);

	f_t x3 = r * cos(angle + offset);
	f_t y3 = r * sin(angle + offset);

	f_t x4 = r * cos(angle + offset - fan::PI / 2);
	f_t y4 = r * sin(angle + offset - fan::PI / 2);

	m_corners = {
		fan::vec2(x1 + position[0] + radius[0], y1 + position[1] + radius[1]),
		fan::vec2(x2 + position[0] + radius[0], y2 + position[1] + radius[1]),
		fan::vec2(x3 + position[0] + radius[0], y3 + position[1] + radius[1]),
		fan::vec2(x4 + position[0] + radius[0], y4 + position[1] + radius[1])
	};
}

void fan_2d::rectangle::draw() const
{

	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size());

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	fan::mat4 model(1);
	model = translate(model, get_position() + get_size() / 2);
	model = fan::rotate(model, fan::radians(m_rotation), fan::vec3(0, 0, 1));
	model = translate(model, -get_size() / 2);
	model = scale(model, get_size());

	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_mat4("model", model);
	this->m_shader.set_vec4("shape_color", get_color());
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::SQUARE));

	rectangle::basic_shape::basic_draw(GL_TRIANGLES, 6);
}

fan::vec2 fan_2d::rectangle::move(f32_t speed, f32_t gravity, f32_t jump_force, f32_t friction)
{
	fan::vec2 offset = fan_2d::move_object(m_window, m_position, m_velocity, speed, gravity, jump_force, friction);
	for (std::size_t i = 0; i < sizeof(m_corners) / sizeof(m_corners[0]); i++) {
		m_corners[i] += offset;
	}
	return offset;
}

//void fan_2d::rectangle::set_position(const fan::vec2& position)
//{
//	const fan::vec2 old_position = this->get_position();
//	basic_single_shape::set_position(position);
//	for (int i = 0; i < sizeof(m_corners) / sizeof(m_corners[0]); i++) {
//		m_corners[i] += position - old_position;
//	}
//}

fan::vec2 fan_2d::load_image(uint32_t& texture_id, const std::string& path, bool flip_image)
{
	std::ifstream file(path);
	if (!file.good()) {
		fan::print("sprite loading error: File path does not exist for", path.c_str());
		exit(1);
	}

	fan::vec2i image_size;

	if ((int)texture_id == fan::uninitialized) {
		texture_id = SOIL_load_OGL_texture(path.c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, (flip_image ? SOIL_FLAG_INVERT_Y : 0));

		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, image_size.begin());
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, image_size.begin() + 1);

		glBindTexture(GL_TEXTURE_2D, texture_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

	}
	else {

		glBindTexture(GL_TEXTURE_2D, texture_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		auto pixels = SOIL_load_image(path.c_str(), image_size.begin(), image_size.begin() + 1, 0, 0);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_size.x, image_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		glGenerateMipmap(GL_TEXTURE_2D);

		SOIL_free_image_data(pixels);

		glBindTexture(GL_TEXTURE_2D, 0);

	}

	return image_size;
}

fan_2d::image_info fan_2d::load_image(unsigned char* pixels, const fan::vec2i& size)
{
	unsigned int texture_id = 0;

	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	if (pixels != nullptr) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	return { size, texture_id };
}

fan_2d::line_vector fan_2d::create_grid(fan::camera& camera, const fan::vec2i& block_size, const fan::vec2i& grid_size, const fan::color& color)
{
	fan_2d::line_vector lv(camera);
	
	const fan::vec2 view = (fan::cast<f_t>(grid_size) / block_size).ceiled();

	for (int i = 0; i < view.x; i++) {
		lv.push_back(fan::mat2(fan::vec2(i * block_size.x, 0), fan::vec2(i * block_size.x, grid_size.y)), color, true);
	}

	for (int i = 0; i < view.y; i++) {
		lv.push_back(fan::mat2(fan::vec2(0, i * block_size.y), fan::vec2(grid_size.x, i * block_size.y)), color, true);
	}

	lv.release_queue(true, true);

	return lv;
}

fan_2d::sprite::sprite(fan::camera& camera) :
	sprite::basic_shape<0, fan::vec2>(camera, fan::shader(shader_paths::single_sprite_vs, shader_paths::single_sprite_fs), fan::vec2(), fan::vec2()), 
	m_rotation(0), m_transparency(1) {}

fan_2d::sprite::sprite(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size, f_t transparency)
	: sprite::basic_shape<0, fan::vec2>(camera, fan::shader(shader_paths::single_sprite_vs, shader_paths::single_sprite_fs), position, size), 
	  m_rotation(0), m_transparency(transparency), m_path(path)
{
	this->load_sprite(path, size);
}

fan_2d::sprite::sprite(fan::camera& camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size, f_t transparency)
	: sprite::basic_shape<0, fan::vec2>(camera, fan::shader(shader_paths::single_sprite_vs, shader_paths::single_sprite_fs), position, size), 
	  m_rotation(0), m_transparency(transparency)
{
	this->reload_sprite(pixels, size);
}

fan_2d::sprite::sprite(const fan_2d::sprite& sprite)
	: sprite::basic_shape(sprite),
	  sprite::basic_shape_velocity(sprite),
	  texture_handler(sprite), m_rotation(sprite.m_rotation), 
	  m_transparency(sprite.m_transparency), m_path(sprite.m_path) { 
	this->load_sprite(sprite.m_path);
}

fan_2d::sprite::sprite(fan_2d::sprite&& sprite) noexcept
	: sprite::basic_shape(std::move(sprite)),
	  sprite::basic_shape_velocity(std::move(sprite)),
	  texture_handler(std::move(sprite)), m_rotation(sprite.m_rotation), 
	  m_transparency(sprite.m_transparency), m_path(std::move(sprite.m_path)) { }

void fan_2d::sprite::load_sprite(const std::string& path, const fan::vec2i& size, bool flip_image)
{
	this->reload_sprite(path, size, flip_image);
}

void fan_2d::sprite::reload_sprite(unsigned char* pixels, const fan::vec2i& size)
{
	glBindTexture(GL_TEXTURE_2D, this->m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void fan_2d::sprite::reload_sprite(const std::string& path, const fan::vec2i& size, bool flip_image)
{
	this->m_path = path;

	fan::vec2i image_size = fan_2d::load_image(m_texture, path, flip_image);

	if (size == 0) {
		this->set_size(image_size);
	}
	else {
		this->set_size(size);
	}
}

void fan_2d::sprite::draw()
{

	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size());

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	fan::mat4 model(1);
	model = translate(model, get_position() + get_size() / 2);
	model = fan::rotate(model, fan::radians(get_rotation()), fan::vec3(0, 0, 1));
	model = fan::translate(model, -get_size() / 2);
	model = scale(model, get_size());

	m_shader.use();
	m_shader.set_mat4("projection", projection);
	m_shader.set_mat4("view", view);
	m_shader.set_mat4("model", model);
	m_shader.set_int("texture_sampler", 0);
	m_shader.set_float("transparency", m_transparency);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	basic_shape::basic_draw(GL_TRIANGLES, 6);
}

f32_t fan_2d::sprite::get_rotation()
{
	return this->m_rotation;
}

void fan_2d::sprite::set_rotation(f32_t degrees)
{
	this->m_rotation = degrees;
}

//fan_2d::animation::animation(const fan::vec2& position, const fan::vec2& size) : basic_single_shape(fan::shader(shader_paths::single_sprite_vs, shader_paths::single_sprite_fs), position, size) {}
//
//void fan_2d::animation::add(const std::string& path)
//{
//	auto texture_info = fan_2d::sprite::load_image(path);
//	this->m_textures.emplace_back(texture_info.texture_id);
//	fan::vec2 image_size = texture_info.image_size;
//	if (size != 0) {
//		image_size = size;
//	}
//	this->set_size(image_size);
//}
//
//void fan_2d::animation::draw(uint_t m_texture)
//{
//	shader.use();
//
//	fan::mat4 model(1);
//	model = translate(model, get_position());
//
//	model = scale(model, get_size());
//
//	shader.set_mat4("projection", fan_2d::frame_projection);
//	shader.set_mat4("view", fan_2d::frame_view);
//	shader.set_mat4("model", model);
//	shader.set_int("texture_sampler", 0);
//
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, m_textures[m_texture]);
//
//	fan_2d::basic_single_shape::basic_draw(GL_TRIANGLES, 6);
//}

void fan::bind_vao(uint32_t vao, const std::function<void()>& function)
{
	glBindVertexArray(vao);
	function();
	glBindVertexArray(0);
}

void fan::write_glbuffer(unsigned int buffer, void* data, uint_t size, uint_t target, uint_t location)
{
	glBindBuffer(target, buffer);
	glBufferData(target, size, data, GL_STATIC_DRAW);
	glBindBuffer(target, 0);
	if (target == GL_SHADER_STORAGE_BUFFER) {
		glBindBufferBase(target, location, buffer);
	}
}

void fan::edit_glbuffer(unsigned int buffer, void* data, uint_t offset, uint_t size, uint_t target, uint_t location)
{
	glBindBuffer(target, buffer);
	glBufferSubData(target, offset, size, data);
	glBindBuffer(target, 0);
	if (target == GL_SHADER_STORAGE_BUFFER) {
		glBindBufferBase(target, location, buffer);
	}
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::basic_shape_position() : glsl_location_handler<layout_location, buffer_type>() {}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::basic_shape_position(const _Vector& position) 
	: basic_shape_position()
{
	if constexpr (enable_vector) {
		this->basic_push_back(position, true);
	}
	else {
		this->m_position = position;
	}
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::basic_shape_position(const basic_shape_position& vector)
	: glsl_location_handler<layout_location, buffer_type>(vector) {
	this->m_position = vector.m_position;
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::basic_shape_position(basic_shape_position&& vector) noexcept
	: glsl_location_handler<layout_location, buffer_type>(std::move(vector)) {
	this->m_position = std::move(vector.m_position);
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>& fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::operator=(
	const fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>& vector) {

	glsl_location_handler<layout_location, buffer_type>::operator=(vector);

	this->m_position = vector.m_position;

	return *this;
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>& fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::operator=(
	fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>&& vector) {

	glsl_location_handler<layout_location, buffer_type>::operator=(std::move(vector));

	this->m_position = std::move(vector.m_position);

	return *this;
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::reserve(uint_t new_size)
{
	m_position.reserve(new_size);
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::resize(uint_t new_size)
{
	m_position.resize(new_size);
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp std::vector<_Vector> fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::get_positions() const 
{
	return this->m_position;
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::set_positions(const std::vector<_Vector>& positions) {
	this->m_position.clear();
	this->m_position.insert(this->m_position.begin(), positions.begin(), positions.end());
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp _Vector fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::get_position(uint_t i) const {
	return this->m_position[i];
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::set_position(uint_t i, const _Vector& position, bool queue) {
	this->m_position[i] = position;

	if (!queue) {
		this->edit_data(i);
	}
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::erase(uint_t i, bool queue)
{
	this->m_position.erase(this->m_position.begin() + i);

	if (!queue) {
		this->write_data();
	}
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::erase(uint_t begin, uint_t end, bool queue)
{
	if (!begin && end == this->m_position.size()) {
		this->m_position.clear();
	}
	else {
		this->m_position.erase(this->m_position.begin() + begin, this->m_position.begin() + end);
	}

	if (!queue) {
		this->write_data();
	}
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp _Vector fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::get_position() const
{
	return m_position;
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::set_position(const _Vector& position)
{
	this->m_position = position;
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::initialize_buffers(bool divisor)
{
	glsl_location_handler<layout_location, buffer_type>::initialize_buffers(m_position.data(), sizeof(_Vector) * m_position.size(), divisor, _Vector::size());
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::basic_push_back(const _Vector& position, bool queue)
{
	this->m_position.emplace_back(position);
	if (!queue) {
		this->write_data();
	}
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::edit_data(uint_t i)
{
	glsl_location_handler<layout_location, buffer_type>::edit_data(i, m_position.data() + i, sizeof(_Vector));
}

template <bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_position<enable_vector, _Vector, layout_location, buffer_type>::write_data()
{
	glsl_location_handler<layout_location, buffer_type>::write_data(m_position.data(), sizeof(_Vector) * m_position.size());
}

template class fan::basic_shape_position<false, fan::vec2>;
template class fan::basic_shape_position<false, fan::vec3>;
template class fan::basic_shape_position<false, fan::vec4>;

template class fan::basic_shape_position<true, fan::vec2>;
template class fan::basic_shape_position<true, fan::vec3>;
template class fan::basic_shape_position<true, fan::vec4>;

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::basic_shape_size() : glsl_location_handler<layout_location, buffer_type>() {}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::basic_shape_size(const _Vector& size) : fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>() {
	if constexpr (enable_vector) {
		this->basic_push_back(size, true);
	}
	else {
		this->m_size = size;
	}
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::basic_shape_size(const basic_shape_size& vector) 
	: glsl_location_handler<layout_location, buffer_type>(vector) {
	this->m_size = vector.m_size;
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::basic_shape_size(basic_shape_size&& vector) noexcept
	: glsl_location_handler<layout_location, buffer_type>(std::move(vector)) {
	this->m_size = std::move(vector.m_size);
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>& fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::operator=(
	const basic_shape_size& vector) {

	glsl_location_handler<layout_location, buffer_type>::operator=(vector);

	this->m_size = vector.m_size;

	return *this;
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>& fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::operator=(
	basic_shape_size&& vector) noexcept {

	glsl_location_handler<layout_location, buffer_type>::operator=(std::move(vector));

	this->m_size = std::move(m_size);

	return *this;
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::reserve(uint_t new_size)
{
	m_size.reserve(new_size);
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::resize(uint_t new_size)
{
	m_size.resize(new_size);
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp std::vector<_Vector> fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::get_sizes() const
{
	return m_size;
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp _Vector fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::get_size(uint_t i) const
{
	return this->m_size[i];
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::set_size(uint_t i, const _Vector& size, bool queue)
{
	this->m_size[i] = size;

	if (!queue) {
		this->edit_data(i);
	}
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::erase(uint_t i, bool queue)
{
	this->m_size.erase(this->m_size.begin() + i);

	if (!queue) {
		this->write_data();
	}
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::erase(uint_t begin, uint_t end, bool queue)
{
	this->m_size.erase(this->m_size.begin() + begin, this->m_size.begin() + end);

	if (!queue) {
		this->write_data();
	}
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp _Vector fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::get_size() const
{
	return m_size;
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::set_size(const _Vector& size)
{
	this->m_size = size;
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::basic_push_back(const _Vector& size, bool queue)
{
	this->m_size.emplace_back(size);

	if (!queue) {
		this->write_data();
	}
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::edit_data(uint_t i)
{
	glsl_location_handler<layout_location, buffer_type>::edit_data(i, m_size.data() + i, sizeof(_Vector));
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::write_data()
{
	glsl_location_handler<layout_location, buffer_type>::write_data(m_size.data(), sizeof(_Vector) * m_size.size());
}

template<bool enable_vector, typename _Vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>::initialize_buffers(bool divisor)
{
	glsl_location_handler<layout_location, buffer_type>::initialize_buffers(m_size.data(), vector_byte_size(m_size), divisor, _Vector::size());
}

template <bool enable_vector, typename _Vector>
fan::basic_shape_velocity<enable_vector, _Vector>::basic_shape_velocity() : m_velocity(1) {}

template <bool enable_vector, typename _Vector>
fan::basic_shape_velocity<enable_vector, _Vector>::basic_shape_velocity(const _Vector& velocity)
{
	this->m_velocity.emplace_back(velocity);
}

template <bool enable_vector, typename _Vector>
fan::basic_shape_velocity<enable_vector, _Vector>::basic_shape_velocity(const basic_shape_velocity& vector)
{
	this->operator=(vector);
}

template <bool enable_vector, typename _Vector>
fan::basic_shape_velocity<enable_vector, _Vector>::basic_shape_velocity(basic_shape_velocity&& vector) noexcept
{
	this->operator=(std::move(vector));
}

template <bool enable_vector, typename _Vector>
fan::basic_shape_velocity<enable_vector, _Vector>& fan::basic_shape_velocity<enable_vector, _Vector>::operator=(const basic_shape_velocity& vector)
{
	this->m_velocity = vector.m_velocity;

	return *this;
}

template <bool enable_vector, typename _Vector>
fan::basic_shape_velocity<enable_vector, _Vector>& fan::basic_shape_velocity<enable_vector, _Vector>::operator=(basic_shape_velocity&& vector) noexcept
{
	this->m_velocity = std::move(vector.m_velocity);

	return *this;
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp _Vector fan::basic_shape_velocity<enable_vector, _Vector>::get_velocity(uint_t i) const
{
	return this->m_velocity[i];
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape_velocity<enable_vector, _Vector>::set_velocity(uint_t i, const _Vector& velocity)
{
	this->m_velocity[i] = velocity;
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape_velocity<enable_vector, _Vector>::reserve(uint_t new_size)
{
	this->m_velocity.reserve(new_size);
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape_velocity<enable_vector, _Vector>::resize(uint_t new_size)
{
	this->m_velocity.resize(new_size);
}

template<bool enable_vector, typename _Vector>
enable_function_for_vector_cpp _Vector fan::basic_shape_velocity<enable_vector, _Vector>::get_velocity() const
{
	return this->m_velocity;
}

template<bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape_velocity<enable_vector, _Vector>::set_velocity(const _Vector& velocity)
{
	this->m_velocity = velocity;
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::basic_shape_color_vector()
 : glsl_location_handler<layout_location, buffer_type>() {}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::basic_shape_color_vector(const basic_shape_color_vector& vector)
: glsl_location_handler<layout_location, buffer_type>(vector) {
	this->m_color = vector.m_color;
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::basic_shape_color_vector(basic_shape_color_vector&& vector) noexcept
: glsl_location_handler<layout_location, buffer_type>(std::move(vector)) {
	this->m_color = std::move(vector.m_color);
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>& fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::operator=(
	const fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>& vector) {

	glsl_location_handler<layout_location, buffer_type>::operator=(vector);

	this->m_color = vector.m_color;

	return *this;
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>& fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::operator=(
	fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>&& vector) noexcept {

	glsl_location_handler<layout_location, buffer_type>::operator=(std::move(vector));

	this->m_color = std::move(vector.m_color);

	return *this;
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::basic_shape_color_vector(const fan::color& color)
 : basic_shape_color_vector() {
	if constexpr (enable_vector) {
		basic_push_back(color, true);
	}
	else {
		this->m_color = color;
	}
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::reserve(uint_t new_size)
{
	m_color.reserve(new_size);
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::resize(uint_t new_size, const fan::color& color)
{
	m_color.resize(new_size, color);
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::resize(uint_t new_size)
{
	m_color.resize(new_size);
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp fan::color fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::get_color(uint_t i)
{
	return this->m_color[i];
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::erase(uint_t i, bool queue)
{
	this->m_color.erase(this->m_color.begin() + i);

	if (!queue) {
		this->write_data();
	}
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::erase(uint_t begin, uint_t end, bool queue)
{
	this->m_color.erase(this->m_color.begin() + begin, this->m_color.begin() + end);

	if (!queue) {
		this->write_data();
	}
}

template<bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp fan::color fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::get_color() const
{
	return this->m_color;
}

template<bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::set_color(const fan::color& color)
{
	return this->m_color = color;
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::basic_push_back(const fan::color& color, bool queue)
{
	this->m_color.emplace_back(color);

	if (!queue) {
		this->write_data();
	}
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::edit_data(uint_t i)
{
	glsl_location_handler<layout_location, buffer_type>::edit_data(i, m_color.data() + i, sizeof(fan::color));
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::write_data()
{
	glsl_location_handler<layout_location, buffer_type>::write_data(m_color.data(), sizeof(fan::color) * m_color.size());
}

template <bool enable_vector, uint_t layout_location, fan::opengl_buffer_type buffer_type>
enable_function_for_vector_cpp void fan::basic_shape_color_vector<enable_vector, layout_location, buffer_type>::initialize_buffers(bool divisor)
{
	glsl_location_handler<layout_location, buffer_type>::initialize_buffers(m_color.data(), sizeof(fan::color) * m_color.size(), divisor, fan::color::size());
}

template <bool enable_vector, typename _Vector>
fan::basic_shape<enable_vector, _Vector>::basic_shape(fan::camera& camera)
	: m_camera(camera), m_window(camera.m_window) {}

template <bool enable_vector, typename _Vector>
fan::basic_shape<enable_vector, _Vector>::basic_shape(fan::camera& camera, const fan::shader& shader)
	: m_shader(shader), m_camera(camera), m_window(camera.m_window) { }

template <bool enable_vector, typename _Vector>
fan::basic_shape<enable_vector, _Vector>::basic_shape(fan::camera& camera, const fan::shader& shader, const _Vector& position, const _Vector& size)
	: basic_shape::basic_shape_position(position), basic_shape::basic_shape_size(size), m_shader(shader), m_camera(camera), m_window(camera.m_window) { }

template <bool enable_vector, typename _Vector>
fan::basic_shape<enable_vector, _Vector>::basic_shape(const basic_shape& vector)
	: basic_shape::basic_shape_position(vector),
	  basic_shape::basic_shape_size(vector), vao_handler(vector),
	  m_shader(vector.m_shader), m_camera(vector.m_camera), m_window(vector.m_window) { }

template <bool enable_vector, typename _Vector>
fan::basic_shape<enable_vector, _Vector>::basic_shape(basic_shape&& vector) noexcept
	:  basic_shape::basic_shape_position(std::move(vector)), 
	   basic_shape::basic_shape_size(std::move(vector)), vao_handler(std::move(vector)),
	   m_shader(std::move(vector.m_shader)), m_camera(vector.m_camera), m_window(vector.m_window) { }

template <bool enable_vector, typename _Vector>
fan::basic_shape<enable_vector, _Vector>& fan::basic_shape<enable_vector, _Vector>::operator=(const basic_shape& vector)
{
	basic_shape::basic_shape_position::operator=(vector);
	basic_shape::basic_shape_size::operator=(vector);
	vao_handler::operator=(vector);

	m_camera.operator=(vector.m_camera);
	m_shader.operator=(vector.m_shader);
	m_window = vector.m_window;

	return *this;
}

template <bool enable_vector, typename _Vector>
fan::basic_shape<enable_vector, _Vector>& fan::basic_shape<enable_vector, _Vector>::operator=(basic_shape&& vector) noexcept
{
	basic_shape::basic_shape_position::operator=(std::move(vector));
	basic_shape::basic_shape_size::operator=(std::move(vector));
	vao_handler::operator=(std::move(vector));

	this->m_camera.operator=(std::move(vector.m_camera));
	this->m_shader.operator=(std::move(vector.m_shader));
	this->m_window = std::move(vector.m_window);

	return *this;
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::reserve(uint_t new_size)
{
	basic_shape::basic_shape_position::reserve(new_size);
	basic_shape::basic_shape_size::reserve(new_size);
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::resize(uint_t new_size)
 {
	basic_shape::basic_shape_position::resize(new_size);
	basic_shape::basic_shape_size::resize(new_size);
}

template class fan::basic_shape<false, fan::vec2>;
template class fan::basic_shape<false, fan::vec3>;
template class fan::basic_shape<false, fan::vec4>;

template class fan::basic_shape<true, fan::vec2>;
template class fan::basic_shape<true, fan::vec3>;
template class fan::basic_shape<true, fan::vec4>;

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp uint_t fan::basic_shape<enable_vector, _Vector>::size() const
{
	return this->m_position.size();
}

template<bool enable_vector, typename _Vector>
f_t fan::basic_shape<enable_vector, _Vector>::get_delta_time() const
{
	return m_window.get_delta_time();
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::basic_push_back(const _Vector& position, const _Vector& size, bool queue)
{
	basic_shape::basic_shape_position::basic_push_back(position, queue);
	basic_shape::basic_shape_size::basic_push_back(size, queue);
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::erase(uint_t i, bool queue)
{
	basic_shape::basic_shape_position::erase(i, queue);
	basic_shape::basic_shape_size::erase(i, queue);
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::erase(uint_t begin, uint_t end, bool queue)
{
	basic_shape::basic_shape_position::erase(begin, end, queue);
	basic_shape::basic_shape_size::erase(begin, end, queue);
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::edit_data(uint_t i, bool position, bool size)
{
	if (position) {
		basic_shape::basic_shape_position::edit_data(i);
	}
	if (size) {
		basic_shape::basic_shape_size::edit_data(i);
	}
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::write_data(bool position, bool size)
{
	if (position) {
		basic_shape::basic_shape_position::write_data();
	}
	if (size) {
		basic_shape::basic_shape_size::write_data();
	}
}

template <bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::basic_draw(unsigned int mode, uint_t count, uint_t primcount, uint_t i) const
{
	glBindVertexArray(m_vao);
	if (i != (uint_t)-1) {
		glDrawArraysInstancedBaseInstance(mode, 0, count, 1, i);
	}
	else {
		glDrawArraysInstanced(mode, 0, count, primcount);
	}

	glBindVertexArray(0);
}

template<bool enable_vector, typename _Vector>
enable_function_for_vector_cpp void fan::basic_shape<enable_vector, _Vector>::basic_draw(unsigned int mode, uint_t count) const
{
	glBindVertexArray(m_vao);
	glDrawArrays(mode, 0, count);
	glBindVertexArray(0);
}

template <typename _Vector>
fan::basic_vertice_vector<_Vector>::basic_vertice_vector(fan::camera& camera, const fan::shader& shader)
	: m_shader(shader), m_window(camera.m_window), m_camera(camera) {}

template<typename _Vector>
fan::basic_vertice_vector<_Vector>::basic_vertice_vector(fan::camera& camera, const fan::shader& shader, const fan::vec2& position, const fan::color& color)
	:  basic_vertice_vector::basic_shape_position(position), basic_vertice_vector::basic_shape_color_vector(color), 
	   m_shader(shader), m_window(camera.m_window), m_camera(camera) { }

template<typename _Vector>
fan::basic_vertice_vector<_Vector>::basic_vertice_vector(const basic_vertice_vector& vector)
	: basic_vertice_vector::basic_shape_position(vector), 
	  basic_vertice_vector::basic_shape_color_vector(vector),
	  basic_vertice_vector::basic_shape_velocity(vector), 
	  vao_handler(vector), m_shader(vector.m_shader), 
	  m_window(vector.m_window), m_camera(vector.m_camera) { }

template<typename _Vector>
fan::basic_vertice_vector<_Vector>::basic_vertice_vector(basic_vertice_vector&& vector) noexcept
	: basic_vertice_vector::basic_shape_position(std::move(vector)), 
	  basic_vertice_vector::basic_shape_color_vector(std::move(vector)),
	  basic_vertice_vector::basic_shape_velocity(std::move(vector)),
	  vao_handler(std::move(vector)), m_shader(std::move(vector.m_shader)), 
	  m_window(vector.m_window), m_camera(vector.m_camera) { }

template<typename _Vector>
fan::basic_vertice_vector<_Vector>& fan::basic_vertice_vector<_Vector>::operator=(const basic_vertice_vector& vector)
{
	basic_vertice_vector::basic_shape_position::operator=(vector);
	basic_vertice_vector::basic_shape_color_vector::operator=(vector);
	basic_vertice_vector::basic_shape_velocity::operator=(vector);
	vao_handler::operator=(vector);

	m_shader = vector.m_shader;
	m_window = vector.m_window;
	m_camera = vector.m_camera;

	return *this;
}

template<typename _Vector>
fan::basic_vertice_vector<_Vector>& fan::basic_vertice_vector<_Vector>::operator=(basic_vertice_vector&& vector)
{
	basic_vertice_vector::basic_shape_position::operator=(std::move(vector));
	basic_vertice_vector::basic_shape_color_vector::operator=(std::move(vector));
	basic_vertice_vector::basic_shape_velocity::operator=(std::move(vector));
	vao_handler::operator=(std::move(vector));

	m_shader = std::move(vector.m_shader);
	m_window = std::move(vector.m_window);
	m_camera = std::move(vector.m_camera);

	return *this;
}

template<typename _Vector>
uint_t fan::basic_vertice_vector<_Vector>::size() const
{
	return basic_vertice_vector::basic_shape_position::m_position.size();
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::basic_push_back(const _Vector& position, const fan::color& color, bool queue)
{
	basic_vertice_vector::basic_shape_position::basic_push_back(position, queue);
	basic_vertice_vector::basic_shape_color_vector::basic_push_back(color, queue);
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::erase(uint_t i, bool queue)
{
	basic_vertice_vector::basic_shape_position::erase(i, queue);
	basic_vertice_vector::basic_shape_color_vector::erase(i, queue);
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::erase(uint_t begin, uint_t end, bool queue)
{
	basic_vertice_vector::basic_shape_position::erase(begin, end, queue);
	basic_vertice_vector::basic_shape_color_vector::erase(begin, end, queue);
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::edit_data(uint_t i, bool position, bool color)
{
	if (position) {
		basic_vertice_vector::basic_shape_position::edit_data(i);
	}
	if (color) {
		basic_vertice_vector::basic_shape_color_vector::edit_data(i);
	}
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::write_data(bool position, bool color)
{
	if (position) {
		basic_vertice_vector::basic_shape_position::write_data();
	}
	if (color) {
		basic_vertice_vector::basic_shape_color_vector::write_data();
	}
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::basic_draw(unsigned int mode, uint_t count)
{
	fan::bind_vao(this->m_vao, [&] {
		glDrawElements(mode, count, GL_UNSIGNED_INT, 0);
	});
}

template <uint_t layout_location, uint_t gl_buffer>
fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::basic_shape_color_vector_vector() : m_color_vbo(-1) {
	glGenBuffers(1, &m_color_vbo);
}

template <uint_t layout_location, uint_t gl_buffer>
fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::basic_shape_color_vector_vector(const std::vector<fan::color>& color)
	: basic_shape_color_vector_vector()
{
	this->m_color.emplace_back(color);

	this->write_data();
}

template <uint_t layout_location, uint_t gl_buffer>
fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::~basic_shape_color_vector_vector()
{
	fan_validate_buffer(m_color_vbo, glDeleteBuffers(1, &m_color_vbo));
}

template <uint_t layout_location, uint_t gl_buffer>
std::vector<fan::color> fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::get_color(uint_t i)
{
	return this->m_color[i];
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::set_color(uint_t i, const std::vector<fan::color>& color, bool queue)
{
	this->m_color[i] = color;
	if (!queue) {
		write_data();
	}
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::erase(uint_t i, bool queue)
{
	this->m_color.erase(this->m_color.begin() + i);
	if (!queue) {
		this->write_data();
	}
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::basic_push_back(const std::vector<fan::color>& color, bool queue)
{
	this->m_color.emplace_back(color);
	if (!queue) {
		write_data();
	}
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::edit_data(uint_t i)
{
	std::vector<fan::color> vector(this->m_color[i].size());
	std::copy(this->m_color[i].begin(), this->m_color[i].end(), vector.begin());
	uint_t offset = 0;
	for (uint_t j = 0; j < i; j++) {
		offset += m_color[j].size();
	}
	fan::edit_glbuffer(this->m_color_vbo, vector.data(), sizeof(fan::color) * offset, sizeof(fan::color) * vector.size(), gl_buffer, layout_location);
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::initialize_buffers(bool divisor)
{
	glBindBuffer(gl_buffer, m_color_vbo);
	glBufferData(gl_buffer, 0, nullptr, GL_STATIC_DRAW);
	if constexpr (gl_buffer == GL_SHADER_STORAGE_BUFFER) {
		glBindBufferBase(gl_buffer, layout_location, m_color_vbo);
	}
	else {
		glEnableVertexAttribArray(layout_location);
		glVertexAttribPointer(layout_location, 4, fan::GL_FLOAT_T, GL_FALSE, sizeof(fan::color), 0);
	}
	if (divisor) {
		glVertexAttribDivisor(0, 1);
	}
}


fan_2d::vertice_vector::vertice_vector(fan::camera& camera, uint_t index_restart)
	: basic_vertice_vector(camera, fan::shader(fan_2d::shader_paths::shape_vector_vs, fan_2d::shader_paths::shape_vector_fs)), m_index_restart(index_restart), m_offset(0)
{
	fan::bind_vao(this->m_vao, [&] {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_position<true, fan::vec2>::initialize_buffers(false);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::initialize_buffers(false);
	});
}

fan_2d::vertice_vector::vertice_vector(fan::camera& camera, const fan::vec2& position, const fan::color& color, uint_t index_restart)
	: basic_vertice_vector(camera, fan::shader(fan_2d::shader_paths::shape_vector_vs, fan_2d::shader_paths::shape_vector_fs), position, color), m_index_restart(index_restart), m_offset(0)
{
	this->initialize_buffers();
}

fan_2d::vertice_vector::vertice_vector(const vertice_vector& vector)
	: fan::basic_vertice_vector<fan::vec2>(vector), ebo_handler(vector) {

	this->m_indices = vector.m_indices;
	this->m_offset = vector.m_offset;
	this->m_index_restart = vector.m_index_restart;

	this->initialize_buffers();
}

fan_2d::vertice_vector::vertice_vector(vertice_vector&& vector) noexcept
: fan::basic_vertice_vector<fan::vec2>(std::move(vector)), ebo_handler(std::move(vector)) { 

	this->m_indices = std::move(vector.m_indices);
	this->m_offset = vector.m_offset;
	this->m_index_restart = vector.m_index_restart;
}

fan_2d::vertice_vector& fan_2d::vertice_vector::operator=(const vertice_vector& vector)
{
	this->m_indices = vector.m_indices;
	this->m_offset = vector.m_offset;
	this->m_index_restart = vector.m_index_restart;

	ebo_handler::operator=(vector);
	basic_vertice_vector::operator=(vector);

	this->initialize_buffers();

	return *this;
}

fan_2d::vertice_vector& fan_2d::vertice_vector::operator=(vertice_vector&& vector) noexcept
{
	this->m_indices = std::move(vector.m_indices);
	this->m_offset = vector.m_offset;
	this->m_index_restart = vector.m_index_restart;

	ebo_handler::operator=(std::move(vector));
	basic_vertice_vector::operator=(std::move(vector));

	return *this;
}


void fan_2d::vertice_vector::release_queue(bool position, bool color, bool indices)
{
	if (position) {
		fan::basic_vertice_vector<fan::vec2>::basic_shape_position<true, fan::vec2>::write_data();
	}
	if (color) {
		fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::write_data();
	}
	if (indices) {
		this->write_data();
	}
}

void fan_2d::vertice_vector::push_back(const fan::vec2& position, const fan::color& color, bool queue)
{
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position<true, fan::vec2>::basic_push_back(position, queue);
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::basic_push_back(color, queue);

	m_indices.emplace_back(m_offset);
	m_offset++;

	if (!(m_offset % this->m_index_restart) && !m_indices.empty()) {
		m_indices.emplace_back(UINT32_MAX);
	}
	if (!queue) {
		this->write_data();
	}
}

void fan_2d::vertice_vector::draw(uint32_t mode)
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size());

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->m_shader.use();

	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::VERTICE));

	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(this->m_index_restart);

//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	vertice_vector::basic_vertice_vector::basic_draw(mode, size());

	glDisable(GL_PRIMITIVE_RESTART);
}

void fan_2d::vertice_vector::erase(uint_t i, bool queue)
{
	m_indices.erase(m_indices.begin() + i);

	fan::basic_vertice_vector<fan::vec2>::erase(i, queue);

	m_offset--; // ? FIX

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::vertice_vector::erase(uint_t begin, uint_t end, bool queue)
{
	if (!begin && end == m_indices.size()) {
		this->m_indices.clear();
	}
	else {
		m_indices.erase(m_indices.begin() + begin, m_indices.begin() + end);
	}

	fan::basic_vertice_vector<fan::vec2>::erase(begin, end, queue);
	
	m_offset = 0; // ? FIX

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::vertice_vector::initialize_buffers()
{
	fan::bind_vao(this->m_vao, [&] {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_position<true, fan::vec2>::initialize_buffers(false);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::initialize_buffers(false);
	});
}

void fan_2d::vertice_vector::write_data()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

fan_2d::line_vector::line_vector(fan::camera& camera)
	: basic_shape(camera, fan::shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs))
{
	this->initialize_buffers();
}

fan_2d::line_vector::line_vector(fan::camera& camera, const fan::mat2& begin_end, const fan::color& color)
	: basic_shape(camera, fan::shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), begin_end[0], begin_end[1]), basic_shape_color_vector(color)
{
	this->initialize_buffers();
}

fan_2d::line_vector::line_vector(const line_vector& vector)
	:	line_vector::basic_shape(vector), 
		line_vector::basic_shape_color_vector(vector)
{
	this->initialize_buffers();
}

fan_2d::line_vector::line_vector(line_vector&& vector) noexcept
	:	line_vector::basic_shape(std::move(vector)), 
		line_vector::basic_shape_color_vector(std::move(vector)) { }

fan_2d::line_vector& fan_2d::line_vector::operator=(const line_vector& vector)
{

	line_vector::basic_shape::operator=(vector);
	line_vector::basic_shape_color_vector::operator=(vector);

	this->initialize_buffers();

	return *this;
}

fan_2d::line_vector& fan_2d::line_vector::operator=(line_vector&& vector) noexcept
{
	line_vector::basic_shape::operator=(std::move(vector));
	line_vector::basic_shape_color_vector::operator=(std::move(vector));

	return *this;
}

void fan_2d::line_vector::reserve(uint_t size)
{
	line_vector::basic_shape::reserve(size);
	basic_shape_color_vector::reserve(size);
}

void fan_2d::line_vector::resize(uint_t size, const fan::color& color)
{
	line_vector::basic_shape::resize(size);
	basic_shape_color_vector::resize(size, color);
	this->release_queue(true, true);
}

void fan_2d::line_vector::push_back(const fan::mat2& begin_end, const fan::color& color, bool queue)
{
	basic_shape::basic_push_back(begin_end[0], begin_end[1], queue);
	m_color.emplace_back(color);

	if (!queue) {
		release_queue(false, true);
	}
}

void fan_2d::line_vector::draw()
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size());

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->m_shader.use();

	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::LINE));

	basic_shape::basic_draw(GL_LINES, 2, size());
}

void fan_2d::line_vector::set_position(uint_t i, const fan::mat2& begin_end, bool queue)
{
	basic_shape::set_position(i, begin_end[0], true);
	basic_shape::set_size(i, begin_end[1], true);

	if (!queue) {
		release_queue(true, false);
	}
}

void fan_2d::line_vector::release_queue(bool position, bool color)
{
	if (position) {
		basic_shape::write_data(true, true);
	}
	if (color) {
		basic_shape_color_vector::write_data();
	}
}

void fan_2d::line_vector::initialize_buffers()
{
	fan::bind_vao(this->m_vao, [&] {
		line_vector::basic_shape_position::initialize_buffers(true);
		line_vector::basic_shape_size::initialize_buffers(true);
		line_vector::basic_shape_color_vector::initialize_buffers(true);
	});
}


//fan_2d::triangle_vector::triangle_vector(const fan::mat3x2& corners, const fan::color& color)
//{
//
//}
//
//void fan_2d::triangle_vector::set_position(uint_t i, const fan::mat3x2& corners)
//{
//
//}
//
//void fan_2d::triangle_vector::push_back(const fan::mat3x2& corners, const fan::color& color)
//{
//
//}

//void fan_2d::triangle_vector::draw()
//{
//	this->m_shader.use();
//
//	this->m_shader.set_mat4("projection", fan_2d::frame_projection);
//	this->m_shader.set_mat4("view", fan_2d::frame_view);
//	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::TRIANGLE));
//
//	//basic_shape_vector::basic_draw(GL_TRIANGLES, 3, m_lcorners.size());
//}

fan_2d::rectangle_vector::rectangle_vector(fan::camera& camera)
	: basic_shape(camera, fan::shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs)) {
	this->initialize_buffers();
}

fan_2d::rectangle_vector::rectangle_vector(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color)
	: basic_shape(camera, fan::shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), position, size), basic_shape_color_vector(color)
{
	this->initialize_buffers();

	this->m_corners.resize(1);
	this->m_rotation.resize(1);
	this->m_velocity.resize(1);
}

fan_2d::rectangle_vector::rectangle_vector(const rectangle_vector& vector)
	:	fan_2d::rectangle_vector::basic_shape(vector),
		fan_2d::rectangle_vector::basic_shape_color_vector(vector),
		fan_2d::rectangle_vector::basic_shape_velocity(vector),
		m_corners(vector.m_corners), m_rotation(vector.m_rotation)
{
	this->initialize_buffers();
}

fan_2d::rectangle_vector::rectangle_vector(rectangle_vector&& vector) noexcept
	:	fan_2d::rectangle_vector::basic_shape(std::move(vector)),
		fan_2d::rectangle_vector::basic_shape_color_vector(std::move(vector)),
		fan_2d::rectangle_vector::basic_shape_velocity(std::move(vector)),
		m_corners(std::move(vector.m_corners)), m_rotation(std::move(vector.m_rotation)) { }

fan_2d::rectangle_vector& fan_2d::rectangle_vector::operator=(const rectangle_vector& vector)
{
	fan_2d::rectangle_vector::basic_shape::operator=(vector);
	fan_2d::rectangle_vector::basic_shape_color_vector::operator=(vector);
	fan_2d::rectangle_vector::basic_shape_velocity::operator=(vector);

	this->m_corners = vector.m_corners;
	this->m_rotation = vector.m_rotation;

	this->initialize_buffers();

	return *this;
}

fan_2d::rectangle_vector& fan_2d::rectangle_vector::operator=(rectangle_vector&& vector) noexcept
{
	fan_2d::rectangle_vector::basic_shape::operator=(std::move(vector));
	fan_2d::rectangle_vector::basic_shape_color_vector::operator=(std::move(vector));
	fan_2d::rectangle_vector::basic_shape_velocity::operator=(std::move(vector));

	this->m_corners = std::move(vector.m_corners);
	this->m_rotation = std::move(vector.m_rotation);

	return *this;
}

void fan_2d::rectangle_vector::initialize_buffers()
{
	fan::bind_vao(this->m_vao, [&] {
		rectangle_vector::basic_shape_color_vector::initialize_buffers(true);
		rectangle_vector::basic_shape_position::initialize_buffers(true);
		rectangle_vector::basic_shape_size::initialize_buffers(true);
	});
}

fan_2d::rectangle fan_2d::rectangle_vector::construct(uint_t i)
{
	return fan_2d::rectangle(
		m_camera,
		fan_2d::rectangle_vector::m_position[i], 
		fan_2d::rectangle_vector::m_size[i], 
		fan_2d::rectangle_vector::m_color[i]
	);
}

void fan_2d::rectangle_vector::release_queue(bool position, bool size, bool color)
{
	basic_shape::write_data(position, size);

	if (color) {
		basic_shape_color_vector::write_data();
	}
}

void fan_2d::rectangle_vector::push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, bool queue)
{
	basic_shape::basic_push_back(position, size, queue);
	basic_shape_color_vector::basic_push_back(color, queue);

	this->m_velocity.emplace_back(fan::vec2());
	this->m_corners.resize(this->m_corners.size() + 1);
	this->m_rotation.resize(this->m_rotation.size() + 1);
}

void fan_2d::rectangle_vector::erase(uint_t i, bool queue)
{
	basic_shape::erase(i, queue);
	rectangle_vector::basic_shape_color_vector::erase(i, queue);
	this->m_corners.erase(this->m_corners.begin() + i);
	this->m_velocity.erase(this->m_velocity.begin() + i);
	this->m_rotation.erase(this->m_rotation.begin() + i);
}

void fan_2d::rectangle_vector::erase(uint_t begin, uint_t end, bool queue)
{
	basic_shape::erase(begin, end, queue);
	rectangle_vector::basic_shape_color_vector::erase(begin, end, queue);
	this->m_corners.erase(this->m_corners.begin() + begin, this->m_corners.begin() + end);
	this->m_velocity.erase(this->m_velocity.begin() + begin, this->m_velocity.begin() + end);
	this->m_rotation.erase(this->m_rotation.begin() + begin, this->m_rotation.begin() + end);
}

void fan_2d::rectangle_vector::draw(uint_t i) const
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size());

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->m_shader.use();

	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::SQUARE));
	basic_shape::basic_draw(GL_TRIANGLES, 6, size(), i);
}

fan::vec2 fan_2d::rectangle_vector::get_center(uint_t i) const
{
	return fan_2d::rectangle_vector::m_position[i] + fan_2d::rectangle_vector::m_size[i] * 0.5;
}

fan_2d::rectangle_corners_t fan_2d::rectangle_vector::get_corners(uint_t i) const
{
	if (m_rotation[i]) {
		return fan_2d::rectangle_vector::m_corners[i];
	}
	return fan_2d::get_rectangle_corners(this->get_position(i), this->get_size(i));
}

void fan_2d::rectangle_vector::move(uint_t i, f32_t speed, f32_t gravity, f32_t jump_force, f32_t friction)
{
	move_object(m_window, this->m_position[i], this->m_velocity[i], speed, gravity, jump_force, friction);
	glBindBuffer(GL_ARRAY_BUFFER, basic_shape::basic_shape_position::glsl_location_handler::m_buffer_object);
	fan::vec2 data;
	glGetBufferSubData(GL_ARRAY_BUFFER, sizeof(fan::vec2) * i, sizeof(fan::vec2), data.data());
	if (data != this->m_position[i]) {
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(fan::vec2) * i, sizeof(fan::vec2), this->m_position[i].data());
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool fan_2d::rectangle_vector::inside(uint_t i) const
{
	return fan_2d::collision::rectangle::point_inside_boundary(m_window.get_mouse_position(), this->get_position(i), this->get_size(i));
}

f_t fan_2d::rectangle_vector::get_rotation(uint_t i) const
{
	return m_rotation[i];
}

void fan_2d::rectangle_vector::set_rotation(uint_t i, f_t angle)
{
	m_rotation[i] = angle;
	angle = fan::radians(angle);
	constexpr f_t offset = 3 * fan::PI / 4;
	const fan::vec2 position(this->get_position(i));
	const fan::vec2 radius(this->get_size(i) / 2);
	f_t r = fan_2d::distance(position, position + radius);

	f_t x1 = r * cos(angle + offset);
	f_t y1 = r * sin(angle + offset);

	f_t x2 = r * cos(angle + offset - fan::PI / 2);
	f_t y2 = r * sin(angle + offset - fan::PI / 2);

	f_t x3 = r * cos(angle + offset - fan::PI);
	f_t y3 = r * sin(angle + offset - fan::PI);

	f_t x4 = r * cos(angle + offset - fan::PI * 3);
	f_t y4 = r * sin(angle + offset - fan::PI * 3);

	m_corners[i] = {
		fan::vec2(x1, y1),
		fan::vec2(x2, y2),
		fan::vec2(x3, y3),
		fan::vec2(x4, y4)
	};
}

fan_2d::sprite_vector::sprite_vector(fan::camera& camera, const std::string& path)
	: basic_shape(camera, fan::shader(shader_paths::sprite_vector_vs, shader_paths::sprite_vector_fs)), m_path(path)
{
	this->initialize_buffers();

	this->load_sprite(path);
}

fan_2d::sprite_vector::sprite_vector(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size)
	: basic_shape(camera, fan::shader(shader_paths::sprite_vector_vs, shader_paths::sprite_vector_fs), position, size), m_path(path)
{
	this->initialize_buffers();
	this->load_sprite(path, size);
}

fan_2d::sprite_vector::sprite_vector(const sprite_vector& vector)
	: fan_2d::sprite_vector::basic_shape(vector), sprite_vector::basic_shape_velocity(vector), texture_handler(vector),
		m_rotation(vector.m_rotation), m_original_image_size(vector.m_original_image_size), m_path(vector.m_path) { 

	this->initialize_buffers();
	this->load_sprite(vector.m_path);
}

fan_2d::sprite_vector::sprite_vector(sprite_vector&& vector) noexcept
	: fan_2d::sprite_vector::basic_shape(std::move(vector)), sprite_vector::basic_shape_velocity(std::move(vector)), texture_handler(std::move(vector)),
		m_rotation(std::move(vector.m_rotation)), m_original_image_size(std::move(vector.m_original_image_size)), m_path(std::move(vector.m_path)) { }

fan_2d::sprite_vector& fan_2d::sprite_vector::operator=(const sprite_vector& vector)
{
	sprite_vector::basic_shape::operator=(vector);
	sprite_vector::basic_shape_velocity::operator=(vector);
	sprite_vector::texture_handler::operator=(vector);

	m_path = vector.m_path;
	m_rotation = vector.m_rotation;
	m_original_image_size = vector.m_original_image_size;

	return *this;
}

fan_2d::sprite_vector& fan_2d::sprite_vector::operator=(sprite_vector&& vector) noexcept
{
	sprite_vector::basic_shape::operator=(std::move(vector));
	sprite_vector::basic_shape_velocity::operator=(std::move(vector));
	sprite_vector::texture_handler::operator=(std::move(vector));

	m_path = std::move(vector.m_path);
	m_rotation = std::move(vector.m_rotation);
	m_original_image_size = std::move(vector.m_original_image_size);

	return *this;
}

void fan_2d::sprite_vector::initialize_buffers()
{
	fan::bind_vao(this->m_vao, [&] {
		fan_2d::sprite_vector::basic_shape_position::initialize_buffers(true);
		fan_2d::sprite_vector::basic_shape_size::initialize_buffers(true);
	});
}

void fan_2d::sprite_vector::push_back(const fan::vec2& position, const fan::vec2& size, bool queue)
{
	this->m_position.emplace_back(position);
	if (size == 0) {
		this->m_size.emplace_back(m_original_image_size);
	}
	else {
		this->m_size.emplace_back(size);
	}
	if (!queue) {
		release_queue(true, true);
	}
}

void fan_2d::sprite_vector::reserve(uint_t new_size)
{
	sprite_vector::basic_shape::reserve(new_size);
	sprite_vector::basic_shape_velocity::reserve(new_size);
}

void fan_2d::sprite_vector::resize(uint_t new_size)
{
	sprite_vector::basic_shape::resize(new_size);
	sprite_vector::basic_shape_velocity::resize(new_size);

	basic_shape::write_data(true, true);
}

void fan_2d::sprite_vector::draw()
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size());

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("texture_sampler", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->m_texture);

	basic_shape::basic_draw(GL_TRIANGLES, 6, size());
}

void fan_2d::sprite_vector::release_queue(bool position, bool size)
{
	if (position) {
		basic_shape::write_data(true, false);
	}
	if (size) {
		basic_shape::write_data(false, true);
	}
}

void fan_2d::sprite_vector::erase(uint_t i, bool queue)
{
	sprite_vector::basic_shape::erase(i, queue);
}

void fan_2d::sprite_vector::load_sprite(const std::string& path, const fan::vec2 size)
{
	m_original_image_size = fan_2d::load_image(m_texture, path);
	if (size == 0) {
		if (!m_size.empty()) {
			this->m_size[this->m_size.size() - 1] = m_original_image_size;
			basic_shape::write_data(false, true);
		}
	}
}

void fan_2d::sprite_vector::allocate_texture()
{
	glGenTextures(1, &m_texture);
}

fan_2d::particles::particles(fan::camera& camera)
	: rectangle_vector(camera) { }

void fan_2d::particles::add(const fan::vec2& position, const fan::vec2& size, const fan::vec2& velocity, const fan::color& color, uint_t time)
{
	this->push_back(position, size, color);
	this->m_particles.push_back({ velocity, fan::timer(fan::timer<>::start(), time) });
}

void fan_2d::particles::update()
{
	for (uint_t i = 0; i < this->size(); i++) {
		if (this->m_particles[i].m_timer.finished()) {
			this->erase(i);
			this->m_particles.erase(this->m_particles.begin() + i);
			continue;
		}
		this->set_position(i, this->get_position(i) + this->m_particles[i].m_velocity * m_window.get_delta_time(), true);
	}
	this->release_queue(true, false, false);
}

fan::vec2 fan_2d::gui::get_resize_movement_offset(fan::window& window)
{
	return fan::cast<f_t>(window.get_size() - window.get_previous_size()) * 0.5;
}

void fan_2d::gui::add_resize_callback(fan::window& window, fan::vec2& position) {
	window.add_resize_callback([&] {
		position += fan_2d::gui::get_resize_movement_offset(window);
	});
}

fan_2d::gui::rectangle::rectangle(fan::camera& camera) : fan_2d::rectangle(camera) {
	add_resize_callback(m_window, this->m_position);
}

fan_2d::gui::rectangle::rectangle(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color) 
	: fan_2d::rectangle(camera, position, size, color)
{
	add_resize_callback(m_window, this->m_position);
}

fan_2d::gui::rectangle_vector::rectangle_vector(fan::camera& camera) : fan_2d::rectangle_vector(camera) {
	m_window.add_resize_callback([&] {
		for (auto& i : this->m_position) {
			i += fan_2d::gui::get_resize_movement_offset(m_window);
		}
		fan_2d::gui::rectangle_vector::basic_shape::write_data(true, false);
	});
}

fan_2d::gui::rectangle_vector::rectangle_vector(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color) 
	: fan_2d::rectangle_vector(camera, position, size, color)
{
	m_window.add_resize_callback([&] {
		for (auto& i : this->m_position) {
			i += fan_2d::gui::get_resize_movement_offset(m_window);
		}
		fan_2d::gui::rectangle_vector::basic_shape::write_data(true, false);
	});
}


fan_2d::gui::sprite::sprite(fan::camera& camera) : fan_2d::sprite(camera) {
	add_resize_callback(m_window, this->m_position);
}

fan_2d::gui::sprite::sprite(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size, f_t transparency)
	: fan_2d::sprite(camera, path, position, size, transparency) 
{
	add_resize_callback(m_window, this->m_position);
}

fan_2d::gui::sprite::sprite(fan::camera& camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size, f_t transparency) 
	: fan_2d::sprite(camera, pixels, position, size, transparency) 
{
	add_resize_callback(m_window, this->m_position);
}

fan_2d::gui::sprite_vector::sprite_vector(fan::camera& camera, const std::string& path) : fan_2d::sprite_vector(camera, path) {
	m_window.add_resize_callback([&] {
		for (auto& i : this->m_position) {
			i += fan_2d::gui::get_resize_movement_offset(m_window);
		}
		fan_2d::gui::sprite_vector::basic_shape::write_data(true, false);
	});
}

fan_2d::gui::sprite_vector::sprite_vector(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size) 
	: fan_2d::sprite_vector(camera, path, position, size) 
{
	m_window.add_resize_callback([&] {
		for (auto& i : this->m_position) {
			i += fan_2d::gui::get_resize_movement_offset(m_window);
		}
		fan_2d::gui::sprite_vector::basic_shape::write_data(true, false);
	});
}

fan_2d::gui::rounded_rectangle::rounded_rectangle(fan::camera& camera) : fan_2d::vertice_vector(camera) { }

fan_2d::gui::rounded_rectangle::rounded_rectangle(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color)
	: fan_2d::vertice_vector(camera)
{
	this->push_back(position, size, color);
}

void fan_2d::gui::rounded_rectangle::push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, bool queue)
{
	fan::vec2 offset;

	for (f_t i = 0; i < segments; i++) {
		f_t t = 2 * fan::PI * i / segments - fan::PI / 2;
		if (i == (int)(segments / 2)) {
			offset.x -= (size.x - size.y);
			fan_2d::vertice_vector::push_back(fan::vec2(cos(t), sin(t)) * size.y / 2 + position + size / 2, color, true);
		}
		else if (i == (int)(segments - 1)) {
			offset.x += (size.x - size.y);
			fan_2d::vertice_vector::push_back(fan::vec2(cos(t), sin(t)) * size.y / 2 + position + size / 2, color, true);
		}

		fan_2d::vertice_vector::push_back(fan::vec2(cos(t), sin(t)) * size.y / 2 + position + size / 2 + offset, color, true);
	}
	if (!queue) {
		this->release_queue(true, true, true);
	}
	if (fan_2d::vertice_vector::m_index_restart == UINT32_MAX) {
		fan_2d::vertice_vector::m_index_restart = this->size();
	}
	fan_2d::gui::rounded_rectangle::m_position.emplace_back(position);
	fan_2d::gui::rounded_rectangle::m_size.emplace_back(fan::vec2(size.x, size.y));
	data_offset.emplace_back(this->size());
}

fan::vec2 fan_2d::gui::rounded_rectangle::get_position(uint_t i) const
{
	return fan_2d::gui::rounded_rectangle::m_position[i];
}

void fan_2d::gui::rounded_rectangle::set_position(uint_t i, const fan::vec2& position)
{
	const auto offset = fan_2d::gui::rounded_rectangle::data_offset[i];
	const auto previous_offset = !i ? 0 : fan_2d::gui::rounded_rectangle::data_offset[i - 1];
	const auto distance = position - fan_2d::gui::rounded_rectangle::get_position(i);
	for (uint_t j = 0; j < offset - previous_offset; j++) {
		fan_2d::vertice_vector::m_position[previous_offset + j] += distance;
	}

	basic_shape_position::glsl_location_handler::edit_data(fan_2d::vertice_vector::m_position.data() + previous_offset, sizeof(fan::vec2) * previous_offset, sizeof(fan::vec2) * (offset - previous_offset));

	fan_2d::gui::rounded_rectangle::m_position[i] = position;
}

fan::vec2 fan_2d::gui::rounded_rectangle::get_size(uint_t i) const
{
	return fan_2d::gui::rounded_rectangle::m_size[i];
}

void fan_2d::gui::rounded_rectangle::draw()
{
	fan_2d::vertice_vector::draw(GL_TRIANGLE_FAN);
}

fan_2d::gui::text_renderer::text_renderer(fan::camera& camera) 
	: text_color_t(), outline_color_t(), m_shader(fan_2d::shader_paths::text_renderer_vs, fan_2d::shader_paths::text_renderer_fs), m_window(camera.m_window), m_camera(camera)
{

	this->m_original_image_size = fan_2d::load_image(m_texture, "fonts/consolas.png");

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_texture_vbo);
	glGenBuffers(1, &m_vertex_vbo);
	glGenBuffers(1, &m_letter_ssbo);

	auto font_info = fan::io::file::parse_font("fonts/consolas.fnt");
	m_font = font_info.m_font;
	m_original_font_size = font_info.m_size;
	
	glBindVertexArray(m_vao);

	text_color_t::initialize_buffers(false);
	outline_color_t::initialize_buffers(false);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_vbo);
	glVertexAttribPointer(1, fan::vec2::size(), fan::GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, m_texture_vbo);
	glVertexAttribPointer(2, fan::vec2::size(), fan::GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_letter_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_letter_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

}

fan_2d::gui::text_renderer::text_renderer(fan::camera& camera, const std::string& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color, bool queue)
	: fan_2d::gui::text_renderer::text_renderer(camera)
{
	this->push_back(text, position, text_color, font_size, outline_color, queue);
}

fan_2d::gui::text_renderer::~text_renderer() {

	fan_validate_buffer(m_vao, glDeleteVertexArrays(1, &m_vao));
	fan_validate_buffer(m_texture_vbo, glDeleteBuffers(1, &m_texture_vbo));
	fan_validate_buffer(m_vertex_vbo, glDeleteBuffers(1, &m_vertex_vbo));
	fan_validate_buffer(m_letter_ssbo, glDeleteBuffers(1, &m_letter_ssbo));

	fan_validate_buffer(m_texture, glDeleteTextures(1, &m_texture));
}

fan::vec2 fan_2d::gui::text_renderer::get_position(uint_t i) const {
	return this->m_position[i];
}

void fan_2d::gui::text_renderer::set_position(uint_t i, const fan::vec2& position, bool queue)
{

	if (this->get_position(i) == position) {
		return;
	}
	auto current_position = this->get_position(i);

	for (uint_t j = 0; j < this->m_vertices[i].size(); j++) {
		this->m_vertices[i][j] -= current_position;
		this->m_vertices[i][j] += position;
	}

	this->m_position[i] = position;

	if (!queue) {
		this->write_vertices();
	}
}

f_t fan_2d::gui::text_renderer::get_font_size(uint_t i) const
{
	return m_font_size[i][0];
}

void fan_2d::gui::text_renderer::set_font_size(uint_t i, f_t font_size, bool queue) {
	if (this->get_font_size(i) == font_size) {
		return;
	}

	std::fill(m_font_size[i].begin(), m_font_size[i].end(), font_size);

	this->load_characters(i, this->get_position(i), m_text[i], true, false);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::gui::text_renderer::set_text(uint_t i, const std::string& text, bool queue) {
	if (m_text.size() && m_text[i] == text) {
		return;
	}

	const auto position = this->get_position(i);
	// possibly a bit unsafe
	const auto text_color = text_color_t::get_color(i)[0];
	const auto outline_color = outline_color_t::get_color(i)[0];
	const auto font_size = this->m_font_size[i][0];
	this->erase(i, true);
	this->insert(i, text, position, text_color, font_size, outline_color, true);
	if (!queue) {
		this->write_data();
	}
}

void fan_2d::gui::text_renderer::set_text_color(uint_t i, const fan::color& color, bool queue)
{
	if (text_color_t::m_color[i][0] == color) {
		return;
	}
	std::fill(text_color_t::m_color[i].begin(), text_color_t::m_color[i].end(), color);
	if (!queue) {
		text_color_t::edit_data(i);
	}
}

void fan_2d::gui::text_renderer::set_outline_color(uint_t i, const fan::color& color, bool queue)
{
	if (outline_color_t::m_color[i][0] == color) {
		return;
	}
	std::fill(outline_color_t::m_color[i].begin(), outline_color_t::m_color[i].end(), color);
	if (!queue) {
		outline_color_t::edit_data(i);
	}
}

fan::io::file::font_t fan_2d::gui::text_renderer::get_letter_info(char c, f_t font_size) const
{
	auto found = m_font.find(c);

	if (found == m_font.end()) {
		throw std::runtime_error("failed to find character: " + std::to_string(c));
	}

	f_t converted_size = this->convert_font_size(font_size);

	return fan::io::file::font_t{
		found->second.m_position * converted_size,
		found->second.m_size * converted_size,
		found->second.m_offset * converted_size,
		(fan::vec2::type)(found->second.m_advance * converted_size)
	};
}

fan::vec2 fan_2d::gui::text_renderer::get_text_size(const std::string& text, f_t font_size) const
{
	fan::vec2 length;

	f_t current = 0;

	int new_lines = 0;

	for (const auto& i : text) {

		if (i == '\n') {
			length.x = std::max((f_t)length.x, current);
			length.y += fan_2d::gui::font_properties::new_line;
			new_lines++;
			current = 0;
		}

		auto found = m_font.find(i);
		if (found == m_font.end()) {
			throw std::runtime_error("failed to find character: " + std::to_string(i));
		}

		current += found->second.m_advance;
		length.y = std::max((f_t)length.y, fan_2d::gui::font_properties::new_line * new_lines + (f_t)found->second.m_size.y + std::abs(found->second.m_offset.y));
	}

	length.x = std::max((f_t)length.x, current);

	if (text.size()) {
		length.x -= m_font.find(text[text.size() - 1])->second.m_offset.x;
	}

	return length * convert_font_size(font_size);
}

fan::vec2 fan_2d::gui::text_renderer::get_text_size_original(const std::string& text, f_t font_size) const {
	fan::vec2 length;

	const f_t converted_font_size(this->convert_font_size(font_size));

	for (const auto& i : text) {
		auto found = m_font.find(i);
		if (found == m_font.end()) {
			throw std::runtime_error("failed to find character: " + std::to_string(i));
		}
		length.x += found->second.m_advance * converted_font_size;
		length.y = std::max((f_t)length.y, found->second.m_size.y * converted_font_size);
	}

	return length;
}

f_t fan_2d::gui::text_renderer::get_font_height_max(uint32_t font_size)
{
	f_t height = 0;
	const f_t converted = this->convert_font_size(font_size) / (1.439024390243902 / 1.06);

	for (const auto& i : m_font) {
		height = std::max(height, (i.second.m_size.y + i.second.m_offset.y) * converted);
	}
	return height;
}

fan::color fan_2d::gui::text_renderer::get_color(uint_t i, uint_t j) const
{
	return text_color_t::m_color[i][j];
}

std::string fan_2d::gui::text_renderer::get_text(uint_t i) const
{
	return this->m_text[i];
}

f_t fan_2d::gui::text_renderer::convert_font_size(f_t font_size) const
{
	return (1.0 / m_original_font_size * font_size);
}

void fan_2d::gui::text_renderer::free_queue() {
	this->write_data();
}

void fan_2d::gui::text_renderer::push_back(const std::string& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color, bool queue) {
	if (text.empty()) {
		fan::print("text renderer string is empty");
		exit(1);
	}
	m_text.emplace_back(text);
	this->m_position.emplace_back(position);

	m_font_size.resize(m_font_size.size() + 1);
	text_color_t::m_color.resize(text_color_t::m_color.size() + 1);
	outline_color_t::m_color.resize(outline_color_t::m_color.size() + 1);

	m_vertices.resize(m_vertices.size() + 1);
	m_texture_coordinates.resize(m_texture_coordinates.size() + 1);

	m_font_size[m_font_size.size() - 1].insert(m_font_size[m_font_size.size() - 1].end(), text.size(), font_size);

	text_color_t::m_color[text_color_t::m_color.size() - 1].insert(text_color_t::m_color[text_color_t::m_color.size() - 1].end(), text.size(), text_color);
	outline_color_t::m_color[outline_color_t::m_color.size() - 1].insert(outline_color_t::m_color[outline_color_t::m_color.size() - 1].end(), text.size(), outline_color);

	this->load_characters(m_font_size.size() - 1, position, text, false, false);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::gui::text_renderer::insert(uint_t i, const std::string& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color, bool queue)
{
	m_font_size.insert(m_font_size.begin() + i, std::vector<f32_t>(text.size(), font_size));

	m_text.insert(m_text.begin() + i, text);
	this->m_position.insert(m_position.begin() + i, position);

	text_color_t::m_color.insert(text_color_t::m_color.begin() + i, std::vector<fan::color>(text.size(), text_color));

	outline_color_t::m_color.insert(outline_color_t::m_color.begin() + i, std::vector<fan::color>(text.size(), outline_color));

	m_vertices.insert(m_vertices.begin() + i, std::vector<fan::vec2>(text.size() * 6));

	m_texture_coordinates.insert(m_texture_coordinates.begin() + i, std::vector<fan::vec2>(text.size() * 6));
	
	this->load_characters(i, position, text, true, false);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::gui::text_renderer::draw() const {
	const fan::vec2i window_size = m_window.get_size();
	fan::mat4 projection = fan::ortho<fan::mat4>(0, window_size.x, window_size.y, 0);

	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);

	this->m_shader.set_int("texture_sampler", 0);
	this->m_shader.set_float("original_font_size", m_original_font_size);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->m_texture);

	text_color_t::bind_gl_storage_buffer();
	outline_color_t::bind_gl_storage_buffer();

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_letter_ssbo);

	fan::bind_vao(m_vao, [&] {
		glDisable(GL_DEPTH_TEST);
		glDrawArrays(GL_TRIANGLES, 0, fan::vector_size(m_vertices));
		glEnable(GL_DEPTH_TEST);
	});
}

void fan_2d::gui::text_renderer::erase(uint_t i, bool queue) {
	if (i >= m_vertices.size()) {
		return;
	}
	m_vertices.erase(m_vertices.begin() + i);
	m_texture_coordinates.erase(m_texture_coordinates.begin() + i);
	m_font_size.erase(m_font_size.begin() + i);
	m_position.erase(m_position.begin() + i);
	text_color_t::m_color.erase(text_color_t::m_color.begin() + i);
	outline_color_t::m_color.erase(outline_color_t::m_color.begin() + i);
	m_text.erase(m_text.begin() + i);

	if (!queue) {
		this->write_data();
	}
}

uint_t fan_2d::gui::text_renderer::size() const
{
	return this->m_text.size();
}

uint_t fan_2d::gui::text_renderer::get_character_offset(uint_t i, bool special) {
	uint_t offset = 0;
	for (uint_t j = 0; j < i; j++) {
		offset += m_text[j].size() - (special ? std::count(m_text[j].begin(), m_text[j].end(), '\n') : 0);
	}
	return offset;
}
//
//std::vector<fan::vec2> fan_2d::gui::text_renderer::get_vertices(uint_t i) {
//	uint_t offset = this->get_character_offset(i, true) * 6;
//	return std::vector<fan::vec2>(m_vertices.begin() + offset, m_vertices.begin() + offset + m_text[i].size() * 6);
//}
//
//std::vector<fan::vec2> fan_2d::gui::text_renderer::get_texture_coordinates(uint_t i) {
//	uint_t offset = this->get_character_offset(i, true) * 6;
//	return std::vector<fan::vec2>(m_texture_coordinates.begin() + offset, m_texture_coordinates.begin() + offset + m_text[i].size() * 6);
//}
//
//std::vector<f_t> fan_2d::gui::text_renderer::get_font_size(uint_t i) {
//	uint_t offset = this->get_character_offset(i, true);
//	return std::vector<f_t>(m_font_size.begin() + offset, m_font_size.begin() + offset + m_text[i].size());
//}

void fan_2d::gui::text_renderer::load_characters(uint_t i, fan::vec2 position, const std::string& text, bool edit, bool insert) {
	const f_t converted_font_size(convert_font_size(m_font_size[i][0]));

	int advance = 0;

	uint_t iletter = 0;
	for (const auto& letter : text) {
		if (letter == '\n') {
			advance = 0;
			position.y += fan_2d::gui::font_properties::new_line * converted_font_size;
			continue;
		}
		if (insert) {
			this->insert_letter_data(i, letter, position, advance, converted_font_size);
		}
		else if (edit) {
			this->edit_letter_data(i, iletter, letter, position, advance, converted_font_size);
			iletter += 6;
		}
		else {
			this->write_letter_data(i, letter, position, advance, converted_font_size);
		}
	}
}

void fan_2d::gui::text_renderer::edit_letter_data(uint_t i, uint_t j, const char letter, const fan::vec2& position, int& advance, f_t converted_font_size) {
	const fan::vec2 letter_position = m_font[letter].m_position;
	const fan::vec2 letter_size = m_font[letter].m_size;
	const fan::vec2 letter_offset = m_font[letter].m_offset;
	
	const fan::vec2 texture_offset = letter_position / this->m_original_image_size;
	const fan::vec2 texture_width = (letter_position + letter_size) / this->m_original_image_size;

	m_vertices[i][j + 0] = position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size;
	m_vertices[i][j + 1] = position + fan::vec2(0, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size;
	m_vertices[i][j + 2] = position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size;												   
	m_vertices[i][j + 3] = position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size;
	m_vertices[i][j + 4] = position + fan::vec2(1, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size;
	m_vertices[i][j + 5] = position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size;
					 
	m_texture_coordinates[i][j + 0] = fan::vec2(texture_width.x , texture_offset.y);
	m_texture_coordinates[i][j + 1] = fan::vec2(texture_offset.x, texture_offset.y);
	m_texture_coordinates[i][j + 2] = fan::vec2(texture_offset.x, texture_width.y);
	m_texture_coordinates[i][j + 3] = fan::vec2(texture_offset.x, texture_width.y);
	m_texture_coordinates[i][j + 4] = fan::vec2(texture_width.x , texture_width.y);
	m_texture_coordinates[i][j + 5] = fan::vec2(texture_width.x , texture_offset.y);

	advance += m_font[letter].m_advance;
}

void fan_2d::gui::text_renderer::insert_letter_data(uint_t i, const char letter, const fan::vec2& position, int& advance, f_t converted_font_size)
{
	const fan::vec2 letter_position = m_font[letter].m_position;
	const fan::vec2 letter_size = m_font[letter].m_size;
	const fan::vec2 letter_offset = m_font[letter].m_offset;
	
	const fan::vec2 texture_offset = letter_position / this->m_original_image_size;
	const fan::vec2 texture_width = (letter_position + letter_size) / this->m_original_image_size;

	m_vertices[i].insert(m_vertices[i].begin() + i    , position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].insert(m_vertices[i].begin() + i + 1, position + fan::vec2(0, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].insert(m_vertices[i].begin() + i + 2, position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);																						   
	m_vertices[i].insert(m_vertices[i].begin() + i + 3, position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].insert(m_vertices[i].begin() + i + 4, position + fan::vec2(1, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].insert(m_vertices[i].begin() + i + 5, position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);

	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i    , fan::vec2(texture_width.x , texture_offset.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 1, fan::vec2(texture_offset.x, texture_offset.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 2, fan::vec2(texture_offset.x, texture_width.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 3, fan::vec2(texture_offset.x, texture_width.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 4, fan::vec2(texture_width.x , texture_width.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 5, fan::vec2(texture_width.x , texture_offset.y));

	advance += m_font[letter].m_advance;
}

void fan_2d::gui::text_renderer::write_letter_data(uint_t i, const char letter, const fan::vec2& position, int& advance, f_t converted_font_size) {
	const fan::vec2 letter_position = m_font[letter].m_position;
	const fan::vec2 letter_size = m_font[letter].m_size;
	const fan::vec2 letter_offset = m_font[letter].m_offset;
	
	const fan::vec2 texture_offset = letter_position / this->m_original_image_size;
	const fan::vec2 texture_width = (letter_position + letter_size) / this->m_original_image_size;

	m_vertices[i].emplace_back(position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].emplace_back(position + fan::vec2(0, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].emplace_back(position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);																						   
	m_vertices[i].emplace_back(position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].emplace_back(position + fan::vec2(1, 1) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].emplace_back(position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2i(advance, 0) + letter_offset) * converted_font_size);

	m_texture_coordinates[i].emplace_back(fan::vec2(texture_width.x , texture_offset.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_offset.x, texture_offset.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_offset.x, texture_width.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_offset.x, texture_width.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_width.x , texture_width.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_width.x , texture_offset.y));

	advance += m_font[letter].m_advance;
}

void fan_2d::gui::text_renderer::write_vertices()
{
	std::vector<fan::vec2> vertices;

	for (uint_t i = 0; i < m_vertices.size(); i++) {
		vertices.insert(vertices.end(), m_vertices[i].begin(), m_vertices[i].end());
	}

	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fan::vec2) * fan::vector_size(vertices), vertices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void fan_2d::gui::text_renderer::write_texture_coordinates()
{
	std::vector<fan::vec2> texture_coordinates;

	for (uint_t i = 0; i < m_texture_coordinates.size(); i++) {
		texture_coordinates.insert(texture_coordinates.end(), m_texture_coordinates[i].begin(), m_texture_coordinates[i].end());
	}

	glBindBuffer(GL_ARRAY_BUFFER, m_texture_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fan::vec2) * fan::vector_size(texture_coordinates), texture_coordinates.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void fan_2d::gui::text_renderer::write_font_sizes()
{
	std::vector<f32_t> font_sizes;

	for (uint_t i = 0; i < m_font_size.size(); i++) {
		font_sizes.insert(font_sizes.end(), m_font_size[i].begin(), m_font_size[i].end());
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_letter_ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(font_sizes[0]) * fan::vector_size(font_sizes), font_sizes.data(), GL_STATIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_letter_ssbo);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void fan_2d::gui::text_renderer::write_data() {

	this->write_vertices();
	this->write_texture_coordinates();
	this->write_font_sizes();

	text_color_t::write_data();
	outline_color_t::write_data();
}

fan_2d::gui::text_box::text_box(fan::camera& camera, const std::string& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color)
	: m_tr(camera, text, position + border_size / 2, text_color, font_size), m_rv(camera, position, m_tr.get_text_size(m_tr.get_text(0), m_tr.get_font_size(0)) + border_size, box_color), m_border_size(border_size) { }

fan::vec2 fan_2d::gui::text_box::get_position(uint_t i) const
{
	return m_rv.get_position(i);
}

void fan_2d::gui::text_box::set_position(uint_t i, const fan::vec2& position, bool queue)
{
	m_rv.set_position(i, position, queue);
	m_tr.set_position(i, position + m_border_size / 2, queue);
}

void fan_2d::gui::text_box::set_text(uint_t i, const std::string& text, bool queue)
{
	m_tr.set_text(i, text, queue);
	m_rv.set_size(i, m_tr.get_text_size(m_tr.get_text(i), m_tr.get_font_size(i)) + m_border_size, queue);
}

fan::color fan_2d::gui::text_box::get_box_color(uint_t i) const
{
	return fan::color();
}

void fan_2d::gui::text_box::set_box_color(uint_t i, const fan::color& color, bool queue)
{
	m_rv.set_color(i, color, queue);
}

fan::color fan_2d::gui::text_box::get_text_color(uint_t i) const
{
	return m_tr.get_color(i);
}

void fan_2d::gui::text_box::set_text_color(uint_t i, const fan::color& color, bool queue)
{
	m_tr.set_text_color(i, color, queue);
}

void fan_2d::gui::text_box::draw() const
{
	m_rv.draw();
	m_tr.draw();
}

bool fan_2d::gui::text_box::inside(uint_t i) const
{
	return m_rv.inside(i);
}

void fan_2d::gui::text_box::on_touch(std::function<void()> function)
{
	m_on_touch = function;

	m_rv.m_window.add_mouse_move_callback([&] {
		for (uint_t i = 0; i < m_rv.size(); i++) {
			if (m_rv.inside(i)) {
				m_on_touch();
			}
		}
	});
}

void fan_2d::gui::text_box::on_touch(uint_t i, const std::function<void()>& function)
{
	m_on_touch = function;

	m_rv.m_window.add_mouse_move_callback([&] {
		if (m_rv.inside(i)) {
			m_on_touch();
		}
	});
}

void fan_2d::gui::text_box::on_click(std::function<void()> function, uint16_t key)
{
	m_on_click = function;

	m_rv.m_window.add_key_callback(key, [&] {
		for (int i = 0; i < m_rv.size(); i++) {
			if (m_rv.inside(i)) {
				m_on_click();
			}
		}
	});
}

void fan_2d::gui::text_box::on_click(uint_t i, const std::function<void()>& function, uint16_t key)
{
	m_on_click = function;

	m_rv.m_window.add_key_callback(key, [&] {
		if (m_rv.inside(i)) {
			m_on_click();
		}
	});
}

void fan_2d::gui::text_box::on_release(std::function<void()> function, uint16_t key)
{
	m_on_release = function;

	m_rv.m_window.add_key_callback(key, [&] {
		for (int i = 0; i < m_rv.size(); i++) {
			if (inside(i)) {
				m_on_release();
			}
		}
	}, true);
}

void fan_2d::gui::text_box::on_release(uint_t i, const std::function<void()>& function, uint16_t key)
{
	m_on_release = function;

	m_rv.m_window.add_key_callback(key, [&] {
		if (inside(i)) {
			m_on_release();
		}
	}, true);
}

void fan_2d::gui::text_box::on_exit(std::function<void()> function)
{
	m_on_exit = function;

	m_rv.m_window.add_mouse_move_callback([&] {
		for (int i = 0; i < m_rv.size(); i++) {
			if (!inside(i)) {
				m_on_exit();
			}
		}
	});
}

void fan_2d::gui::text_box::on_exit(uint_t i, const std::function<void()>& function)
{
	m_on_exit = function;

	m_rv.m_window.add_mouse_move_callback([&] {
		if (!inside(i)) {
			m_on_exit();
		}
	});
}

void fan_3d::add_camera_rotation_callback(fan::camera& camera) {
	camera.m_window.add_mouse_move_callback(std::function<void()>(std::bind(&fan::camera::rotate_camera, std::ref(camera), 0)));
}

fan_3d::line_vector::line_vector(fan::camera& camera)
	: basic_shape(camera, fan::shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs)), basic_shape_color_vector()
{
	glBindVertexArray(m_vao);

	line_vector::basic_shape_color_vector::initialize_buffers(true);
	line_vector::basic_shape_position::initialize_buffers(true);
	line_vector::basic_shape_size::initialize_buffers(true);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

fan_3d::line_vector::line_vector(fan::camera& camera, const fan::mat2x3& begin_end, const fan::color& color)
	: basic_shape(camera, fan::shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs),
		begin_end[0], begin_end[1]), basic_shape_color_vector(color)
{
	glBindVertexArray(m_vao);

	line_vector::basic_shape_color_vector::initialize_buffers(true);
	line_vector::basic_shape_position::initialize_buffers(true);
	line_vector::basic_shape_size::initialize_buffers(true);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void fan_3d::line_vector::push_back(const fan::mat2x3& begin_end, const fan::color& color, bool queue)
{
	basic_shape_color_vector::basic_push_back(color, queue);
	basic_shape::basic_push_back(begin_end[0], begin_end[1], queue);
}

void fan_3d::line_vector::draw() {

	fan::mat4 projection(1);
	projection = fan::perspective<fan::mat4>(fan::radians(90.f), (f_t)m_window.get_size().x / (f_t)m_window.get_size().y, 0.1, 1000.0);

	fan::mat4 view(m_camera.get_view_matrix());


	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::LINE));

	basic_shape::basic_draw(GL_LINE_STRIP, 2, size());
}

void fan_3d::line_vector::set_position(uint_t i, const fan::mat2x3 begin_end, bool queue)
{
	basic_shape::set_position(i, begin_end[0], queue);
	basic_shape::set_size(i, begin_end[1], queue);
}

void fan_3d::line_vector::release_queue(bool position, bool color)
{
	if (position) {
		basic_shape::write_data(true, true);
	}
	if (color) {
		basic_shape_color_vector::write_data();
	}
}

fan_3d::terrain_generator::terrain_generator(fan::camera& camera, const std::string& path, const f32_t texture_scale, const fan::vec3& position, const fan::vec2ui& map_size, f_t triangle_size, const fan::vec2& mesh_size)
	: m_shader(fan_3d::shader_paths::triangle_vector_vs, fan_3d::shader_paths::triangle_vector_fs), m_triangle_size(triangle_size), m_window(camera.m_window), m_camera(camera)
{
	std::vector<fan::vec2> texture_coordinates;
	this->m_triangle_vertices.reserve((uint64_t)map_size.x * (uint64_t)map_size.y);

	FastNoiseLite noise;
	noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);

	std::vector<fan::vec3> normals;
	uint_t index = 0;
	fan::vec2 mesh_height;
	for (uint_t y = 0; y < map_size.y; y++) {
		mesh_height.x = 0;
		for (uint_t x = 0; x < map_size.x; x++) {
			this->m_triangle_vertices.emplace_back(fan::vec3(x * triangle_size, y * triangle_size, noise.GetNoise(mesh_height.x, mesh_height.y) * 100) + position);
			index++;
			texture_coordinates.emplace_back(fan::vec2(x * triangle_size, y * triangle_size) / texture_scale);
			mesh_height.x += mesh_size.x;
		}
		mesh_height.y += mesh_size.y;
	}

	for (uint_t y = 0; y < map_size.y - 1; y++) {
		for (uint_t x = 0; x < map_size.x; x++) {
			for (uint_t i = 0; i < 2; i++) {
				this->m_indices.emplace_back(((y + i) * map_size.x) + x);
			}
		}
		for (uint_t x = map_size.x - 1; x != (uint_t)-1; x--) {
			this->m_indices.emplace_back(((y + 1) * map_size.x) + x);
			if (x == map_size.x - 1 || !x) {
				continue;
			}
			this->m_indices.emplace_back(((y + 1) * map_size.x) + x);
		}
	}


	const uint_t first_corners[] = { 1, 0, map_size.x };

	for (uint_t i = 0; i + first_corners[2] < map_size.x * map_size.y; i++) {
		auto v = fan_3d::normalize(fan::cross(
			m_triangle_vertices[first_corners[2] + i] - m_triangle_vertices[first_corners[0] + i],
			m_triangle_vertices[first_corners[1] + i] - m_triangle_vertices[first_corners[0] + i]
		));
		if (v.z < 0) {
			v = -v;
		}
		normals.emplace_back(
			v
		);
	}

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_normals_vbo);
	glGenBuffers(1, &m_vertices_vbo);
	glGenBuffers(1, &m_texture_vbo);
	glGenBuffers(1, &m_ebo);
	glBindVertexArray(m_vao);

	basic_shape_color_vector::initialize_buffers(false);
	m_texture = load_texture(path, std::string(), false);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertices_vbo);
	glBufferData(GL_ARRAY_BUFFER, m_vertice_size * m_triangle_vertices.size(), m_triangle_vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, fan::vec3::size(), fan::GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, m_texture_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texture_coordinates[0]) * texture_coordinates.size(), texture_coordinates.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(2, fan::vec2::size(), fan::GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(2);
	
	glBindBuffer(GL_ARRAY_BUFFER, m_normals_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(normals[0]) * normals.size(), normals.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(3, fan::vec3::size(), fan::GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	fan::print("amount of veritices:", m_triangle_vertices.size());

}

fan_3d::terrain_generator::~terrain_generator()
{
	fan_validate_buffer(m_texture, glDeleteTextures(1, &m_texture));
	fan_validate_buffer(m_vao, glDeleteVertexArrays(1, &m_vao));
	fan_validate_buffer(m_texture_vbo, glDeleteBuffers(1, &m_texture_vbo));
	fan_validate_buffer(m_vertices_vbo, glDeleteBuffers(1, &m_vertices_vbo));
	fan_validate_buffer(m_ebo, glDeleteBuffers(1, &m_ebo));
	fan_validate_buffer(m_normals_vbo, glDeleteBuffers(1, &m_normals_vbo));
	fan_validate_buffer(m_shader.m_id, glDeleteProgram(m_shader.m_id));
}

void fan_3d::terrain_generator::insert(const std::vector<triangle_vertices_t>& vertices, const std::vector<fan::color>& color, bool queue)
{
	fan_3d::terrain_generator::m_triangle_vertices.insert(fan_3d::terrain_generator::m_triangle_vertices.end(), vertices.begin(), vertices.end());

	basic_shape_color_vector::m_color.insert(basic_shape_color_vector::m_color.end(), color.begin(), color.end());

	if (!queue) {
		basic_shape_color_vector::write_data();
		fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
	}
}

void fan_3d::terrain_generator::push_back(const triangle_vertices_t& vertices, const fan::color& color, bool queue) {
	fan_3d::terrain_generator::m_triangle_vertices.emplace_back(vertices);

	std::fill_n(std::back_inserter(basic_shape_color_vector::m_color), 4, color);

	if (!queue) {
		basic_shape_color_vector::write_data();
		fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
	}
}

void fan_3d::terrain_generator::edit_data(uint_t i, const triangle_vertices_t& vertices, const fan::color& color)
{
	basic_shape_color_vector::m_color[i] = color;
	fan_3d::terrain_generator::m_triangle_vertices[i] = vertices;

	glBindBuffer(GL_ARRAY_BUFFER, fan_3d::terrain_generator::m_vertices_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, fan_3d::terrain_generator::m_vertice_size * i, fan_3d::terrain_generator::m_vertice_size, vertices.data());
	
	basic_shape_color_vector::edit_data(i); // could be broken
}

void fan_3d::terrain_generator::release_queue()
{
	basic_shape_color_vector::write_data();
	fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
}

void fan_3d::terrain_generator::draw() {
	fan::mat4 projection(1);
	projection = fan::perspective<fan::mat4>(fan::radians(90.f), (f32_t)m_window.get_size().x / (f32_t)m_window.get_size().y, 0.1f, 10000000.0f);

	fan::mat4 view(1);
		
	view = m_camera.get_view_matrix();

	fan_3d::terrain_generator::m_shader.use();
	fan_3d::terrain_generator::m_shader.set_mat4("projection", projection);
	fan_3d::terrain_generator::m_shader.set_mat4("view", view);
	fan_3d::terrain_generator::m_shader.set_int("triangle_size", m_triangle_size);
	fan_3d::terrain_generator::m_shader.set_vec3("light_position", m_camera.get_position());
	fan_3d::terrain_generator::m_shader.set_vec3("view_position",  m_camera.get_position());

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBindVertexArray(m_vao);
	glDrawElements(GL_TRIANGLE_STRIP, fan_3d::terrain_generator::size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	//lv.draw();

}

void fan_3d::terrain_generator::erase_all()
{
	fan_3d::terrain_generator::m_triangle_vertices.clear();
	basic_shape_color_vector::m_color.clear();
	basic_shape_color_vector::write_data();
	fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
}

uint_t fan_3d::terrain_generator::size() {
	return fan_3d::terrain_generator::m_indices.size();
}

fan_3d::rectangle_vector::rectangle_vector(fan::camera& camera, const std::string& path, uint_t block_size)
	: basic_shape(camera, fan::shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs)),
	  block_size(block_size)
{
	glBindVertexArray(m_vao);

	rectangle_vector::basic_shape_position::initialize_buffers(true);
	rectangle_vector::basic_shape_size::initialize_buffers(true);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	generate_textures(path, block_size);
}

//fan_3d::rectangle_vector::rectangle_vector(fan::camera& camera, const fan::color& color, uint_t block_size)
//	: basic_shape(camera, fan::shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs)),
//	block_size(block_size)
//{
//	glBindVertexArray(m_vao);
//
//	rectangle_vector::basic_shape_color_vector::initialize_buffers(true);
//	rectangle_vector::basic_shape_position::initialize_buffers(true);
//	rectangle_vector::basic_shape_size::initialize_buffers(true);
//
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//	glBindVertexArray(0);
//	//TODO
//	//generate_textures(path, block_size);
//}

fan_3d::rectangle_vector::~rectangle_vector()
{
	fan_validate_buffer(m_texture_ssbo, glDeleteBuffers(1, &m_texture_ssbo));
	fan_validate_buffer(m_texture_id_ssbo, glDeleteBuffers(1, &m_texture_id_ssbo));
}

void fan_3d::rectangle_vector::push_back(const fan::vec3& src, const fan::vec3& dst, const fan::vec2& texture_id, bool queue)
{
	basic_shape::basic_push_back(src, dst, queue);

	this->m_textures.emplace_back(((m_amount_of_textures.y - 1) - texture_id.y) * m_amount_of_textures.x + texture_id.x);

	if (!queue) {
		this->write_textures();
	}
}

fan::vec3 fan_3d::rectangle_vector::get_src(uint_t i) const
{
	return this->m_position[i];
}

fan::vec3 fan_3d::rectangle_vector::get_dst(uint_t i) const
{
	return this->m_size[i];
}

fan::vec3 fan_3d::rectangle_vector::get_size(uint_t i) const
{
	return this->get_dst(i) - this->get_src(i);
}

void fan_3d::rectangle_vector::set_position(uint_t i, const fan::vec3& src, const fan::vec3& dst, bool queue)
{
	this->m_position[i] = src;
	this->m_size[i] = dst;

	if (!queue) {
		rectangle_vector::basic_shape::write_data(true, true);
	}
}

void fan_3d::rectangle_vector::set_size(uint_t i, const fan::vec3& size, bool queue)
{
	rectangle_vector::basic_shape::set_size(i, this->get_src(i) + size, queue);
}

// make sure glEnable(GL_DEPTH_TEST) and glDepthFunc(GL_ALWAYS) is set
void fan_3d::rectangle_vector::draw() {

	fan::mat4 projection(1);
	projection = fan::perspective<fan::mat4>(fan::radians(90.f), (f32_t)m_window.get_size().x / (f32_t)m_window.get_size().y, 0.1f, 10000.0f);

	fan::mat4 view(m_camera.get_view_matrix());

	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::SQUARE));
	this->m_shader.set_int("texture_sampler", 0);

	this->m_shader.set_vec3("player_position", m_camera.get_position());

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->m_texture);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_texture_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	basic_shape::basic_draw(GL_TRIANGLES, 36, size());
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void fan_3d::rectangle_vector::set_texture(uint_t i, const fan::vec2& texture_id, bool queue)
{
	this->m_textures[i] = (f_t)block_size.x / 6 * texture_id.y + texture_id.x;

	if (!queue) {
		write_textures();
	}
}

void fan_3d::rectangle_vector::generate_textures(const std::string& path, const fan::vec2& block_size)
{
	glGenBuffers(1, &m_texture_ssbo);
	glGenBuffers(1, &m_texture_id_ssbo);

	fan::vec2i image_size = fan_2d::load_image(m_texture, path, true);

	glBindTexture(GL_TEXTURE_2D, m_texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	const fan::vec2 texturepack_size = fan::vec2(image_size.x / block_size.x, image_size.y / block_size.y);
	m_amount_of_textures = fan::vec2(texturepack_size.x / 6, texturepack_size.y);
	// 0 = up, 1 = down, 2 = front, 3 = right, 4 = back, 5 = left

	constexpr int side_order[] = { 0, 1, 2, 3, 4, 5 };
	std::vector<fan::vec2> textures;
	for (fan::vec2i m_texture; m_texture.y < m_amount_of_textures.y; m_texture.y++) {
		const fan::vec2 begin(1.f / texturepack_size.x, 1.f / texturepack_size.y);
		const float up = 1 - begin.y * m_texture.y;
		const float down = 1 - begin.y * (m_texture.y + 1);
		for (m_texture.x = 0; m_texture.x < m_amount_of_textures.x; m_texture.x++) {
			for (uint_t side = 0; side < std::size(side_order); side++) {
				const float left = begin.x * side_order[side] + ((begin.x * (m_texture.x)) * 6);
				const float right = begin.x * (side_order[side] + 1) + ((begin.x * (m_texture.x)) * 6);
				const fan::vec2 texture_coordinates[] = {
					fan::vec2(left,  up),
					fan::vec2(left,  down),
					fan::vec2(right, down),
					fan::vec2(right, down),
					fan::vec2(right, up),
					fan::vec2(left,  up)
				};
				for (auto coordinate : texture_coordinates) {
					textures.emplace_back(coordinate);
				}
			}
		}
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_texture_ssbo);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		sizeof(textures[0]) * textures.size(),
		textures.data(),
		GL_STATIC_DRAW
	);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_texture_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void fan_3d::rectangle_vector::write_textures()
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_texture_id_ssbo);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		sizeof(m_textures[0]) * m_textures.size(),
		m_textures.data(),
		GL_STATIC_DRAW
	);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void fan_3d::rectangle_vector::release_queue(bool position, bool size, bool textures)
{
	if (position) {
		basic_shape::write_data(true, false);
	}
	if (size) {
		basic_shape::write_data(false, true);
	}
	if (textures) {
		this->write_textures();
	}
}

fan_3d::square_corners fan_3d::rectangle_vector::get_corners(uint_t i) const
{

	const fan::vec3 position = fan::da_t<f32_t, 2, 3>{ this->get_position(i), this->get_size(i) }.avg();
	const fan::vec3 size = this->get_size(i);
	const fan::vec3 half_size = size * 0.5;

	return square_corners{ 
		{ 
			{ position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
			{ position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
			{ position[0] + half_size[0], position[1] + half_size[1], position[2] - half_size[2] }, 
			{ position[0] - half_size[0], position[1] + half_size[1], position[2] - half_size[2] } 
		}, // left
		{ 
			{ position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
			{ position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
			{ position[0] - half_size[0], position[1] - half_size[1], position[2] - half_size[2] }, 
			{ position[0] + half_size[0], position[1] - half_size[1], position[2] - half_size[2] } 
		}, // right
		{ 
			{ position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
			{ position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
			{ position[0] - half_size[0], position[1] + half_size[1], position[2] - half_size[2] }, 
			{ position[0] - half_size[0], position[1] - half_size[1], position[2] - half_size[2] } 
		}, // front
		{ 
			{ position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
			{ position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
			{ position[0] + half_size[0], position[1] - half_size[1], position[2] - half_size[2] }, 
			{ position[0] + half_size[0], position[1] + half_size[1], position[2] - half_size[2] } 
		}, // back
		{ 
			{ position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
			{ position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
			{ position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
			{ position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2] } 
		}, // top
		{ 
			{ position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
			{ position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
			{ position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
			{ position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2] } 
		}  // bottom
	};
}

uint_t fan_3d::rectangle_vector::size() const
{
	return this->m_position.size();
}

fan_3d::skybox::skybox(
	fan::camera& camera,
	const std::string& left,
	const std::string& right,
	const std::string& front,
	const std::string back,
	const std::string bottom,
	const std::string& top
) : m_shader(fan_3d::shader_paths::skybox_vs, fan_3d::shader_paths::skybox_fs), m_camera(camera) {

	std::array<std::string, 6> images{ right, left, top, bottom, back, front };

	for (int i = 0; i < images.size(); i++) {
		if (!fan::io::file::exists(images[i])) {
			fan::print("path does not exist:", images[i]);
			exit(1);
		}
	}

	glGenTextures(1, &m_texture_id);

	glBindTexture(GL_TEXTURE_CUBE_MAP, m_texture_id);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_NEAREST);

	for (uint_t i = 0; i < images.size(); i++) {
		fan::vec2i image_size;
		unsigned char* image = SOIL_load_image(images[i].c_str(), image_size.data(), image_size.data() + 1, 0, SOIL_LOAD_RGB);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, image_size.x, image_size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
		SOIL_free_image_data(image);
	}

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

	glGenVertexArrays(1, &m_skybox_vao);
	glGenBuffers(1, &m_skybox_vbo);
	glBindVertexArray(m_skybox_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_skybox_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skybox_vertices), &skybox_vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, fan::GL_FLOAT_T, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glBindVertexArray(0);
}

fan_3d::skybox::~skybox() {
	fan_validate_buffer(m_skybox_vao, glDeleteVertexArrays(1, &m_skybox_vao));
	fan_validate_buffer(m_skybox_vbo, glDeleteBuffers(1, &m_skybox_vbo));
	fan_validate_buffer(m_texture_id, glDeleteTextures(1, &m_texture_id));
}

void fan_3d::skybox::draw() {

	fan::mat4 view(1);
	view = m_camera.get_view_matrix();

	fan::mat4 projection = fan::perspective<fan::mat4>(fan::radians(90.f), (f32_t)m_camera.m_window.get_size().x / (f32_t)m_camera.m_window.get_size().y, 0.1, 100.0);


	m_shader.use();

	view[3][0] = 0;
	view[3][1] = 0;
	view[3][2] = 0;

	m_shader.set_mat4("view", view);
	m_shader.set_mat4("projection", projection);
//	m_shader.set_vec3("fog_color", fan::vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));

	glDepthFunc(GL_LEQUAL);
	glBindVertexArray(m_skybox_vao);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, m_texture_id);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glDepthFunc(GL_LESS);
}

fan_3d::model_mesh::model_mesh(
	const std::vector<mesh_vertex>& vertices,
	const std::vector<unsigned int>& indices,
	const std::vector<mesh_texture>& textures
) : vertices(vertices), indices(indices), textures(textures) {
	initialize_mesh();
}

void fan_3d::model_mesh::initialize_mesh() {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(mesh_vertex), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, fan::GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), 0);

	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, fan::GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), reinterpret_cast<void*>(offsetof(mesh_vertex, normal)));

	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 2, fan::GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), reinterpret_cast<void*>(offsetof(mesh_vertex, texture_coordinates)));
	glBindVertexArray(0);
}

fan_3d::model_loader::model_loader(const std::string& path, const fan::vec3& size) {
	load_model(path, size);
}

void fan_3d::model_loader::load_model(const std::string& path, const fan::vec3& size) {
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

	if (scene == nullptr || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr) {
		std::cout << "assimp error: " << importer.GetErrorString() << '\n';
		return;
	}

	directory = path.substr(0, path.find_last_of('/'));

	process_node(scene->mRootNode, scene, size);
}

void fan_3d::model_loader::process_node(aiNode* node, const aiScene* scene, const fan::vec3& size) {
	for (GLuint i = 0; i < node->mNumMeshes; i++) {
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

		meshes.emplace_back(process_mesh(mesh, scene, size));
	}

	for (GLuint i = 0; i < node->mNumChildren; i++) {
		process_node(node->mChildren[i], scene, size);
	}
}

fan_3d::model_mesh fan_3d::model_loader::process_mesh(aiMesh* mesh, const aiScene* scene, const fan::vec3& size) {
	std::vector<mesh_vertex> vertices;
	std::vector<GLuint> indices;
	std::vector<mesh_texture> textures;

	for (GLuint i = 0; i < mesh->mNumVertices; i++)
	{
		mesh_vertex vertex;
		fan::vec3 vector;

		vector.x = mesh->mVertices[i].x / 2 * size.x;
		vector.y = mesh->mVertices[i].y / 2 * size.y;
		vector.z = mesh->mVertices[i].z / 2 * size.z;
		vertex.position = vector;
		if (mesh->mNormals != nullptr) {
			vector.x = mesh->mNormals[i].x;
			vector.y = mesh->mNormals[i].y;
			vector.z = mesh->mNormals[i].z;
			vertex.normal = vector;
		}
		else {
			vertex.normal = fan::vec3();
		}

		if (mesh->mTextureCoords[0]) {
			fan::vec2 vec;
			vec.x = mesh->mTextureCoords[0][i].x;
			vec.y = mesh->mTextureCoords[0][i].y;
			vertex.texture_coordinates = vec;
		}

		vertices.emplace_back(vertex);
	}

	for (GLuint i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (uint_t j = 0; j < face.mNumIndices; j++) {
			indices.emplace_back(face.mIndices[j]);
		}
	}

	aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

	std::vector<mesh_texture> diffuseMaps = this->load_material_textures(material, aiTextureType_DIFFUSE, "texture_diffuse");
	textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

	std::vector<mesh_texture> specularMaps = this->load_material_textures(material, aiTextureType_SPECULAR, "texture_specular");
	textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

	if (textures.empty()) {
		mesh_texture m_texture;
		unsigned int texture_id;
		glGenTextures(1, &texture_id);

		aiColor4D color(0.f, 0.f, 0.f, 0.f);
		aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &color);
		std::vector<unsigned char> pixels;
		pixels.emplace_back(color.r * 255.f);
		pixels.emplace_back(color.g * 255.f);
		pixels.emplace_back(color.b * 255.f);
		pixels.emplace_back(color.a * 255.f);

		glBindTexture(GL_TEXTURE_2D, texture_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);

		m_texture.id = texture_id;
		textures.emplace_back(m_texture);
		textures_loaded.emplace_back(m_texture);
	}
	return model_mesh(vertices, indices, textures);
}

std::vector<fan_3d::mesh_texture> fan_3d::model_loader::load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name) {
	std::vector<mesh_texture> textures;

	for (uint_t i = 0; i < mat->GetTextureCount(type); i++) {
		aiString a_str;
		mat->GetTexture(type, i, &a_str);
		bool skip = false;
		for (const auto& j : textures_loaded) {
			if (j.path == a_str) {
				textures.emplace_back(j);
				skip = true;
				break;
			}
		}

		if (!skip) {
			mesh_texture m_texture;
			m_texture.id = load_texture(a_str.C_Str(), directory, false);
			m_texture.type = type_name;
			m_texture.path = a_str;
			textures.emplace_back(m_texture);
			textures_loaded.emplace_back(m_texture);
		}
	}
	return textures;
}

fan_3d::model::model(fan::camera& camera) : model_loader("", fan::vec3()), m_shader(fan_3d::shader_paths::model_vs, fan_3d::shader_paths::model_fs), m_window(camera.m_window), m_camera(camera.m_window){}

fan_3d::model::model(fan::camera& camera, const std::string& path, const fan::vec3& position, const fan::vec3& size)
	: model_loader(path, size / 2.f), m_shader(fan_3d::shader_paths::model_vs, fan_3d::shader_paths::model_fs),
	m_position(position), m_size(size), m_window(camera.m_window), m_camera(camera.m_window)
{
	for (uint_t i = 0; i < this->meshes.size(); i++) {
		glBindVertexArray(this->meshes[i].vao);
	}
	glBindVertexArray(0);

#ifndef RAM_SAVER
	constexpr auto gpu_prealloc = sizeof(float) * 100000000;
	glBindBuffer(GL_ARRAY_BUFFER, _Shape_Matrix_VBO);
	glBufferData(
		GL_ARRAY_BUFFER,
		gpu_prealloc,
		nullptr,
		GL_STATIC_DRAW
	);
#endif
}

void fan_3d::model::draw() {

	fan::mat4 model(1);
	model = translate(model, get_position());
	model = scale(model, get_size());

	this->m_shader.use();

	fan::mat4 projection(1);
	projection = fan::perspective<fan::mat4>(fan::radians(90.f), (f32_t)m_window.get_size().x / (f32_t)m_window.get_size().y, 0.1f, 1000.0f);

	fan::mat4 view(m_camera.get_view_matrix());

	this->m_shader.set_int("texture_sampler", 0);
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_vec3("light_position", m_camera.get_position());
	this->m_shader.set_vec3("view_position",m_camera.get_position());
	this->m_shader.set_vec3("light_color", fan::vec3(1, 1, 1));
	this->m_shader.set_int("texture_diffuse", 0);
	this->m_shader.set_mat4("model", model);

	//_Shader.set_vec3("sky_color", fan::vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->textures_loaded[0].id);

	glDepthFunc(GL_LEQUAL);
	for (uint_t i = 0; i < this->meshes.size(); i++) {
		glBindVertexArray(this->meshes[i].vao);
		glDrawElementsInstanced(GL_TRIANGLES, (GLsizei)this->meshes[i].indices.size(), GL_UNSIGNED_INT, 0, 1);
	}
	glDepthFunc(GL_LESS);

	glBindVertexArray(0);
}

fan::vec3 fan_3d::model::get_position()
{
	return this->m_position;
}

void fan_3d::model::set_position(const fan::vec3& position)
{
	this->m_position = position;
}

fan::vec3 fan_3d::model::get_size()
{
	return this->m_size;
}

void fan_3d::model::set_size(const fan::vec3& size)
{
	this->m_size = size;
}

fan::vec3 line_triangle_intersection(const fan::vec3& ray_begin, const fan::vec3& ray_end, const fan::vec3& p0, const fan::vec3& p1, const fan::vec3& p2) {

	const auto lab = (ray_begin + ray_end) - ray_begin;

	const auto p01 = p1 - p0;
	const auto p02 = p2 - p0;

	const auto normal = fan::cross(p01, p02);

	const auto t = fan_3d::dot(normal, ray_begin - p0) / fan_3d::dot(-lab, normal);
	const auto u = fan_3d::dot(fan::cross(p02, -lab), ray_begin - p0) / fan_3d::dot(-lab, normal);
	const auto v = fan_3d::dot(fan::cross(-lab, p01), ray_begin - p0) / fan_3d::dot(-lab, normal);

	if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
		return ray_begin + lab * t;
	}

	return INFINITY;

}

fan::vec3 fan_3d::line_triangle_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 3, 3>& triangle) {

	const auto lab = (line[0] + line[1]) - line[0];

	const auto p01 = triangle[1] - triangle[0];
	const auto p02 = triangle[2] - triangle[0];

	const auto normal = fan::cross(p01, p02);

	const auto t = fan_3d::dot(normal, line[0] - triangle[0]) / fan_3d::dot(-lab, normal);
	const auto u = fan_3d::dot(fan::cross(p02, -lab), line[0] - triangle[0]) / fan_3d::dot(-lab, normal);
	const auto v = fan_3d::dot(fan::cross(-lab, p01), line[0] - triangle[0]) / fan_3d::dot(-lab, normal);

	if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
		return line[0] + lab * t;
	}

	return INFINITY;
}

fan::vec3 fan_3d::line_plane_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 4, 3>& square) {
	const fan::da_t<f32_t, 3> plane_normal = fan::normalize_no_sqrt(cross(square[3] - square[2], square[0] - square[2]));
	const f32_t nl_dot(dot(plane_normal, line[1]));

	if (!nl_dot) {
		return fan::vec3(INFINITY);
	}

	const f32_t ray_length = dot(square[2] - line[0], plane_normal) / nl_dot;
	if (ray_length <= 0) {
		return fan::vec3(INFINITY);
	}
	if (fan::custom_pythagorean_no_sqrt(fan::vec3(line[0]), fan::vec3(line[0] + line[1])) < ray_length) {
		return fan::vec3(INFINITY);
	}
	const fan::vec3 intersection(line[0] + line[1] * ray_length);

	auto result = fan_3d::dot((square[2] - line[0]), plane_normal);
	fan::print(result);
	if (!result) {
		fan::print("on plane");
	}

	if (intersection[1] >= square[3][1] && intersection[1] <= square[0][1] &&
		intersection[2] >= square[3][2] && intersection[2] <= square[0][2])
	{
		return intersection;
	}
	return fan::vec3(INFINITY);
}