#include <fan/graphics.hpp>

#include <functional>
#include <numeric>

#include <fan/fast_noise.hpp>

fan::camera::camera(fan::window& window) : m_window(window), yaw(0), pitch(0) {
	this->update_view();
}

void fan::camera::move(f32_t movement_speed, bool noclip, f32_t friction)
{
	if (!noclip) {
		//if (fan::is_colliding) {
			this->velocity.x /= friction * m_window.get_delta_time() + 1;
			this->velocity.y /= friction * m_window.get_delta_time() + 1;
		//}
	}
	else {
		this->velocity /= friction * m_window.get_delta_time() + 1;
	}
	static constexpr auto minimum_velocity = 0.001;
	if (this->velocity.x < minimum_velocity && this->velocity.x > -minimum_velocity) {
		this->velocity.x = 0;
	}
	if (this->velocity.y < minimum_velocity && this->velocity.y > -minimum_velocity) {
		this->velocity.y = 0;
	}
	if (this->velocity.z < minimum_velocity && this->velocity.z > -minimum_velocity) {
		this->velocity.z = 0;
	}
	if (m_window.key_press(fan::input::key_w)) {
		const fan::vec2 direction(fan::direction_vector(fan::radians(this->yaw)));
		this->velocity.x += direction.x * (movement_speed * m_window.get_delta_time());
		this->velocity.y += direction.y * (movement_speed * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_s)) {
		const fan::vec2 direction(fan::direction_vector(fan::radians(this->yaw)));
		this->velocity.x -= direction.x * (movement_speed * m_window.get_delta_time());
		this->velocity.y -= direction.y * (movement_speed * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_a)) {
		this->velocity -= this->right * (movement_speed * m_window.get_delta_time());
	}
	if (m_window.key_press(fan::input::key_d)) {
		this->velocity += this->right * (movement_speed * m_window.get_delta_time());
	}
	if (!noclip) {
		// is COLLIDING
		if (m_window.key_press(fan::input::key_space/*, true*/)) { // FIX THISSSSSS
			this->velocity.z += jump_force;
			jumping = true;
		}
		else {
			jumping = false;
		}
		this->velocity.z += -gravity * m_window.get_delta_time();
	}
	else {
		if (m_window.key_press(fan::input::key_space)) {
			this->velocity.z += movement_speed * m_window.get_delta_time();
		}
		// IS COLLIDING
		if (m_window.key_press(fan::input::key_left_shift)) {
			this->velocity.z -= movement_speed * m_window.get_delta_time();
		}
	}
	this->position += this->velocity * m_window.get_delta_time();
	this->update_view();
}

void fan::camera::rotate_camera(bool when) // this->updateCameraVectors(); move function updates
{
	if (when) {
		return;
	}

	static f32_t lastX, lastY;

	f32_t xpos = m_window.get_cursor_position().x;
	f32_t ypos = m_window.get_cursor_position().y;

	if (first_movement)
	{
		lastX = xpos;
		lastY = ypos;
		first_movement = false;
	}

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
	return look_at_left(this->position, this->position + this->front, this->up);
}

fan::mat4 fan::camera::get_view_matrix(fan::mat4 m) const {
	//																	 to prevent extra trash in camera class
	return m * look_at_right(this->position, this->position + fan::vec3(this->front[0], this->front[2], this->front[1]), vec3(0, 1, 0));
}

fan::vec3 fan::camera::get_position() const {
	return this->position;
}

void fan::camera::set_position(const fan::vec3& position) {
	this->position = position;
}

fan::vec3 fan::camera::get_velocity() const
{
	return fan::camera::velocity;
}

void fan::camera::set_velocity(const fan::vec3& velocity)
{
	fan::camera::velocity = velocity;
}

f32_t fan::camera::get_yaw() const
{
	return this->yaw;
}

f32_t fan::camera::get_pitch() const
{
	return this->pitch;
}

void fan::camera::set_yaw(f32_t angle)
{
	this->yaw = angle;
	if (yaw > fan::camera::max_yaw) {
		yaw = -fan::camera::max_yaw;
	}
	if (yaw < -fan::camera::max_yaw) {
		yaw = fan::camera::max_yaw;
	}
}

void fan::camera::set_pitch(f32_t angle)
{
	this->pitch = angle;
	if (this->pitch > fan::camera::max_pitch) {
		this->pitch = fan::camera::max_pitch;
	}
	if (this->pitch < -fan::camera::max_pitch) {
		this->pitch = -fan::camera::max_pitch;
	} 
}

void fan::camera::update_view() {
	this->front = normalize(fan::direction_vector(this->yaw, this->pitch));
	this->right = normalize(cross(this->world_up, this->front)); 
	this->up = normalize(cross(this->front, this->right));
}

uint32_t load_texture(const std::string_view path, const std::string& directory, bool flip_image) {

	std::string file_name = std::string(directory + (directory.empty() ? "" : "/") + path.data());
	auto texture_info = fan_2d::sprite::load_image(file_name, flip_image);

	glBindTexture(GL_TEXTURE_2D, texture_info.texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	return texture_info.texture_id;
}

fan::mat4 fan_2d::get_projection(const fan::vec2i& window_size, const fan::mat4& projection) {

	return fan::ortho((f_t)window_size.x / 2, window_size.x + (f_t)window_size.x * 0.5f, window_size.y + (f_t)window_size.y * 0.5f, (f_t)window_size.y / 2.f, 0.1f, 1000.0f);
}

fan::mat4 fan_2d::get_view_translation(const fan::vec2i& window_size, const fan::mat4& view)
{
	return fan::translate(view, fan::vec3((f_t)window_size.x / 2, (f_t)window_size.y / 2, -700.0f));
}

void fan_2d::move_object(fan::window& window, fan::vec2& position, fan::vec2& velocity, f32_t speed, f32_t gravity, f32_t jump_force, f32_t friction) {
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
	if (velocity.x < minimum_velocity && velocity.x > -minimum_velocity) {
		velocity.x = 0;
	}
	if (velocity.y < minimum_velocity && velocity.y > -minimum_velocity) {
		velocity.y = 0;
	}

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
	if constexpr (std::is_same<decltype(velocity.x), f32_t>::value) {
		if (velocity.x >= FLT_MAX) {
			velocity.x = FLT_MAX;
		}
		if (velocity.y >= FLT_MAX) {
			velocity.y = FLT_MAX;
		}
	}
	else {
		if (velocity.x >= DBL_MAX) {
			velocity.x = DBL_MAX;
		}
		if (velocity.y >= DBL_MAX) {
			velocity.y = DBL_MAX;
		}
	}
	position += velocity * delta_time;
}

fan_2d::basic_single_shape::basic_single_shape(fan::camera& camera) : m_window(camera.m_window), m_camera(camera)
{
	glGenVertexArrays(1, &vao);
}

fan_2d::basic_single_shape::basic_single_shape(fan::camera& camera, const fan::shader& shader, const fan::vec2& position, const fan::vec2& size)
	: position(position), size(size), shader(shader), m_window(camera.m_window), m_camera(camera)
{
	glGenVertexArrays(1, &vao);
}

fan_2d::basic_single_shape::~basic_single_shape()
{
	glDeleteVertexArrays(1, &this->vao);
	glValidateProgram(this->shader.id);
    int status = 0;
    glGetProgramiv(this->shader.id, GL_VALIDATE_STATUS, &status);
    if (status) {
        glDeleteProgram(this->shader.id);
    }
}

fan::vec2 fan_2d::basic_single_shape::get_position() const
{
	return position;
}

fan::vec2 fan_2d::basic_single_shape::get_size() const
{
	return this->size;
}

fan::vec2 fan_2d::basic_single_shape::get_velocity() const
{
	return fan_2d::basic_single_shape::velocity;
}

fan::vec2 fan_2d::basic_single_shape::get_center() const
{
	return this->position + size / 2;
}

void fan_2d::basic_single_shape::set_size(const fan::vec2& size)
{
	this->size = size;
}

void fan_2d::basic_single_shape::set_position(const fan::vec2& position)
{
	this->position = position;
}

void fan_2d::basic_single_shape::set_velocity(const fan::vec2& velocity)
{
	fan_2d::basic_single_shape::velocity = velocity;
}

void fan_2d::basic_single_shape::basic_draw(GLenum mode, GLsizei count) const
{
	glDisable(GL_DEPTH_TEST);
	glBindVertexArray(vao);
	glDrawArrays(mode, 0, count);
	glBindVertexArray(0);
	glEnable(GL_DEPTH_TEST);
}

void fan_2d::basic_single_shape::move(f32_t speed, f32_t gravity, f32_t jump_force, f32_t friction)
{
	fan_2d::move_object(m_window, this->position, this->velocity, speed, gravity, jump_force, friction);
}

bool fan_2d::basic_single_shape::inside() const
{
	const fan::vec2i cursor_position = m_window.get_cursor_position();
	if (cursor_position.x >= position.x && cursor_position.x <= position.x + size.x &&
		cursor_position.y >= position.y && cursor_position.y <= position.y + size.y)
	{
		return true;
	}
	return false;
}

fan_2d::basic_single_color::basic_single_color() {}

fan_2d::basic_single_color::basic_single_color(const fan::color& color) : color(color) {}

fan::color fan_2d::basic_single_color::get_color() const
{
	return this->color;
}

void fan_2d::basic_single_color::set_color(const fan::color& color)
{
	this->color = color;
}

fan_2d::line::line(fan::camera& camera) : basic_single_shape(camera, fan::shader(shader_paths::single_shapes_vs, shader_paths::single_shapes_fs), fan::vec2(), fan::vec2()), fan_2d::basic_single_color() {}

fan_2d::line::line(fan::camera& camera, const fan::mat2& begin_end, const fan::color& color)
	: basic_single_shape(camera, fan::shader(shader_paths::single_shapes_vs, shader_paths::single_shapes_fs), begin_end[0], begin_end[1]),
	fan_2d::basic_single_color(color) {}

void fan_2d::line::draw()
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size(), projection);

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->shader.use();

	this->shader.set_mat4("projection", projection);
	this->shader.set_mat4("view", view);
	this->shader.set_vec4("shape_color", get_color());
	this->shader.set_int("shape_type", fan::eti(fan::e_shapes::LINE));
	this->shader.set_vec2("begin", basic_single_shape::get_position());
	this->shader.set_vec2("end", get_size());

	fan_2d::basic_single_shape::basic_draw(GL_LINES, 2);
}

fan::mat2 fan_2d::line::get_position() const
{
	return fan::mat2(position, position + size);
}

void fan_2d::line::set_position(const fan::mat2& begin_end)
{
	fan_2d::line::set_position(fan::vec2(begin_end[0]));
	set_size(begin_end[1]);
}

fan_2d::square::square(fan::camera& camera)
	: basic_single_shape(
		camera,
		fan::shader(shader_paths::single_shapes_vs, shader_paths::single_shapes_fs),
		fan::vec2(),
		fan::vec2()
	), fan_2d::basic_single_color() {
}

fan_2d::square::square(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color)
	: basic_single_shape(
		camera,
		fan::shader(shader_paths::single_shapes_vs, shader_paths::single_shapes_fs),
		position, size
	), fan_2d::basic_single_color(color) {}

fan_2d::square_corners_t fan_2d::square::get_corners() const
{
	return fan_2d::square_corners_t::get_corners(this->position, this->size);
}

fan::vec2 fan_2d::square::center() const
{
	return fan_2d::square::position + fan_2d::square::size / 2;
}

void fan_2d::square::draw() const
{

	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size(), projection);

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	fan::mat4 model(1);
	model = translate(model, get_position());
	model = scale(model, get_size());

	this->shader.use();
	this->shader.set_mat4("projection", projection);
	this->shader.set_mat4("view", view);
	this->shader.set_mat4("model", model);
	this->shader.set_vec4("shape_color", get_color());
	this->shader.set_int("shape_type", fan::eti(fan::e_shapes::SQUARE));

	fan_2d::basic_single_shape::basic_draw(GL_TRIANGLES, 6);
}
//
//fan_2d::bloom_square::bloom_square()
//	: fan_2d::basic_single_shape(fan::shader(shader_paths::single_shapes_bloom_vs, shader_paths::single_shapes_bloom_fs), fan::vec2(), fan::vec2())
//{
//	glGenFramebuffers(1, &m_hdr_fbo);
//	glBindFramebuffer(GL_FRAMEBUFFER, m_hdr_fbo);
//	glGenTextures(2, m_color_buffers);
//
//	for (uint_t i = 0; i < 2; i++)
//	{
//		glBindTexture(GL_TEXTURE_2D, m_color_buffers[i]);
//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, fan::window_size.x, fan::window_size.y, 0, GL_RGBA, GL_FLOAT, NULL);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // we clamp to the edge as the blur filter would otherwise sample repeated texture values!
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//		// attach texture to framebuffer
//		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, m_color_buffers[i], 0);
//	}
//
//	glGenRenderbuffers(1, &m_rbo);
//	glBindRenderbuffer(GL_RENDERBUFFER, m_rbo);
//	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, fan::window_size.x, fan::window_size.y);
//	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_rbo);
//	// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
//	unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
//	glDrawBuffers(2, attachments);
//	// finally check if framebuffer is complete
//	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
//		fan::print("Framebuffer not complete!");
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//	glGenFramebuffers(2, m_pong_fbo);
//	glGenTextures(2, m_pong_color_buffer);
//	for (uint_t i = 0; i < 2; i++)
//	{
//		glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[i]);
//		glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[i]);
//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, fan::window_size.x, fan::window_size.y, 0, GL_RGBA, GL_FLOAT, NULL);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // we clamp to the edge as the blur filter would otherwise sample repeated texture values!
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_pong_color_buffer[i], 0);
//		// also check if framebuffers are complete (no need for depth buffer)
//		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
//			std::cout << "Framebuffer not complete!" << std::endl;
//	}
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//	m_shader_blur.use();
//	m_shader_blur.set_int("image", 0);
//	m_shader_bloom.use();
//	m_shader_bloom.set_int("scene", 0);
//	m_shader_bloom.set_int("bloomBlur", 1);
//}
//
//fan_2d::bloom_square::bloom_square(const fan::vec2& position, const fan::vec2& size, const fan::color& color)
//	: fan_2d::bloom_square::bloom_square()
//{
//	this->set_position(position);
//	this->set_size(size);
//	this->set_color(color);
//
//}
//
//void fan_2d::bloom_square::bind_fbo() const
//{
//	glBindFramebuffer(GL_FRAMEBUFFER, m_hdr_fbo);
//}
//
//unsigned int bbb = 0;
//unsigned int ccc;
//static void renderQuad()
//{
//	if (bbb == 0)
//	{
//
//		/*
//		fan::vec2(0, 0),
//		fan::vec2(0, 1),
//		fan::vec2(1, 1),
//		fan::vec2(1, 1),
//		fan::vec2(1, 0),
//		fan::vec2(0, 0)
//
//		*/
//		float quadVertices[] = {
//			// positions        // texture Coords
//			-1.0f,  1.0f, 0.0f, 1.0f,
//			-1.0f, -1.0f, 0.0f, 0.0f,
//			 1.0f,  1.0f, 1.0f, 1.0f,
//			 1.0f, -1.0f, 1.0f, 0.0f,
//		};
//		// setup plane VAO
//		glGenVertexArrays(1, &bbb);
//		glGenBuffers(1, &ccc);
//		glBindVertexArray(bbb);
//		glBindBuffer(GL_ARRAY_BUFFER, ccc);
//		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
//		glEnableVertexAttribArray(0);
//		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
//		glEnableVertexAttribArray(1);
//		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
//	}
//	glBindVertexArray(bbb);
//	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
//	glBindVertexArray(0);
//}
//
//
//void fan_2d::bloom_square::draw()
//{
//	this->shader.use();
//
//	fan::mat4 model(1);
//	model = translate(model, get_position());
//	model = scale(model, get_size());
//
//	this->shader.set_mat4("projection", fan_2d::frame_projection);
//	this->shader.set_mat4("view", fan_2d::frame_view);
//	this->shader.set_mat4("model", model);
//	this->shader.set_vec4("shape_color", get_color());
//	this->shader.set_int("shape_type", fan::eti(fan::e_shapes::SQUARE));
//
//	fan_2d::basic_single_shape::basic_draw(GL_TRIANGLES, 6);
//
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//	// 2. blur bright fragments with two-pass Gaussian Blur 
//	// --------------------------------------------------
//	bool horizontal = true;
//	m_shader_blur.use();
//
//	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[1]);
//	m_shader_blur.set_int("horizontal", 1);
//
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, m_color_buffers[1]);  // bind texture of other framebuffer (or scene if first iteration)
//
//	renderQuad();
//
//	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[0]);
//	m_shader_blur.set_int("horizontal", 0);
//
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[1]);  // bind texture of other framebuffer (or scene if first iteration)
//
//	renderQuad();
//
//
//	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[1]);
//	m_shader_blur.set_int("horizontal", 1);
//
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[0]);  // bind texture of other framebuffer (or scene if first iteration)
//	renderQuad();
//
//	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[0]);
//	m_shader_blur.set_int("horizontal", 0);
//
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[1]);
//
//	renderQuad();
//
//	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[1]);
//	m_shader_blur.set_int("horizontal", 1);
//
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[0]);
//	renderQuad();
//
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//
//	// 3. now render floating point color buffer to 2D quad and tonemap HDR colors to default framebuffer's (clamped) color range
//	// --------------------------------------------------------------------------------------------------------------------------
//
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	m_shader_bloom.use();
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, m_color_buffers[0]);
//	glActiveTexture(GL_TEXTURE1);
//	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[!horizontal]);
//	m_shader_bloom.set_float("exposure", 1.0f);
//	renderQuad();
//
//}

fan_2d::sprite::sprite(fan::camera& camera) :
	basic_single_shape(camera, fan::shader(shader_paths::single_sprite_vs, shader_paths::single_sprite_fs), fan::vec2(), fan::vec2()) {}

fan_2d::sprite::sprite(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size, f_t transparency)
	: basic_single_shape(camera, fan::shader(shader_paths::single_sprite_vs, shader_paths::single_sprite_fs), position, size), m_transparency(transparency) {
	auto texture_info = load_image(path);
	this->m_texture = texture_info.texture_id;
	fan::vec2 image_size = texture_info.image_size;
	if (size != 0) {
		image_size = size;
	}
	set_size(image_size);
}

fan_2d::sprite::sprite(fan::camera& camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size, f_t transparency)
	: basic_single_shape(camera, fan::shader(shader_paths::single_sprite_vs, shader_paths::single_sprite_fs), position, size), m_transparency(transparency)
{
	auto texture_info = load_image(pixels, size);
	this->m_texture = texture_info.texture_id;
	fan::vec2 image_size = texture_info.image_size;
	if (size != 0) {
		image_size = size;
	}
	set_size(image_size);
}

void fan_2d::sprite::reload_image(unsigned char* pixels, const fan::vec2i& size)
{
	glBindTexture(GL_TEXTURE_2D, this->m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, pixels);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void fan_2d::sprite::reload_image(const std::string& path, const fan::vec2i& size, bool flip_image)
{
	auto texture_info = fan_2d::sprite::load_image(path, flip_image);

	this->m_texture = texture_info.texture_id;

	if (size == 0) {
		this->set_size(texture_info.image_size);
	}
	else {
		this->set_size(size);
	}
}

void fan_2d::sprite::draw()
{

	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size(), projection);

	fan::mat4 view(1);
	view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	fan::mat4 model(1);
	model = translate(model, get_position());
	model = scale(model, get_size());
	//model = Rotate(model, fan::radians(get_rotation()), fan::vec3(0, 0, 1));

	shader.use();
	shader.set_mat4("projection", projection);
	shader.set_mat4("view", view);
	shader.set_mat4("model", model);
	shader.set_int("texture_sampler", 0);
	shader.set_float("transparency", m_transparency);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	fan_2d::basic_single_shape::basic_draw(GL_TRIANGLES, 6);
}

f32_t fan_2d::sprite::get_rotation()
{
	return this->m_rotation;
}

void fan_2d::sprite::set_rotation(f32_t degrees)
{
	this->m_rotation = degrees;
}

fan_2d::image_info fan_2d::sprite::load_image(const std::string& path, bool flip_image)
{
	std::ifstream file(path);
	if (!file.good()) {
		fan::print("sprite loading error: File path does not exist for", path.c_str());
		exit(1);
	}

	uint32_t texture_id = SOIL_load_OGL_texture(path.c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, (flip_image ? SOIL_FLAG_INVERT_Y : 0) | SOIL_FLAG_MIPMAPS | SOIL_FLAG_NTSC_SAFE_RGB);
	fan::vec2i image_size;

	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, image_size.begin());
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, image_size.begin() + 1);

	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	return { image_size, texture_id };
}

fan_2d::image_info fan_2d::sprite::load_image(unsigned char* pixels, const fan::vec2i& size)
{
	unsigned int texture_id = 0;

	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	if (pixels != nullptr) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA, size.x, size.y, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, pixels);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	return { fan::vec2i(1920, 1080), texture_id };
}

//fan_2d::animation::animation(const fan::vec2& position, const fan::vec2& size) : basic_single_shape(fan::shader(shader_paths::single_sprite_vs, shader_paths::single_sprite_fs), position, size) {}
//
//void fan_2d::animation::add(const std::string& path)
//{
//	auto texture_info = fan_2d::sprite::load_image(path);
//	this->m_textures.push_back(texture_info.texture_id);
//	fan::vec2 image_size = texture_info.image_size;
//	if (size != 0) {
//		image_size = size;
//	}
//	this->set_size(image_size);
//}
//
//void fan_2d::animation::draw(std::uint64_t m_texture)
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

void fan::write_glbuffer(unsigned int buffer, void* data, std::uint64_t size, uint_t target, uint_t location)
{
	glBindBuffer(target, buffer);
	glBufferData(target, size, data, GL_DYNAMIC_DRAW);
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


template<typename _Vector>
fan::basic_shape_position_vector<_Vector>::basic_shape_position_vector() : m_position_vbo(-1) {
	glGenBuffers(1, &m_position_vbo);
}

template<typename _Vector>
fan::basic_shape_position_vector<_Vector>::basic_shape_position_vector(const _Vector& position) : fan::basic_shape_position_vector<_Vector>()
{
	this->basic_push_back(position);
}

template<typename _Vector>
fan::basic_shape_position_vector<_Vector>::~basic_shape_position_vector()
{
	glDeleteBuffers(1, &m_position_vbo);
}


template<typename _Vector>
void fan::basic_shape_position_vector<_Vector>::set_positions(const std::vector<_Vector>& positions)
{
	this->m_position.clear();
	this->m_position.insert(this->m_position.begin(), positions.begin(), positions.end());
}

template <typename _Vector>
_Vector fan::basic_shape_position_vector<_Vector>::get_position(std::uint64_t i) const
{
	return this->m_position[i];
}

template<typename _Vector>
void fan::basic_shape_position_vector<_Vector>::set_position(std::uint64_t i, const _Vector& position, bool queue)
{
	this->m_position[i] = position;
	if (!queue) {
		this->edit_data(i);
	}
}

template<typename _Vector>
void fan::basic_shape_position_vector<_Vector>::erase(std::uint64_t i, bool queue)
{
	this->m_position.erase(this->m_position.begin() + i);
	if (!queue) {
		this->write_data();
	}
}

template<typename _Vector>
void fan::basic_shape_position_vector<_Vector>::basic_push_back(const _Vector& position, bool queue)
{
	this->m_position.push_back(position);
	if (!queue) {
		this->write_data();
	}
}

template<typename _Vector>
void fan::basic_shape_position_vector<_Vector>::edit_data(uint_t i)
{
	fan::edit_glbuffer(this->m_position_vbo, m_position.data() + i, sizeof(_Vector) * i, sizeof(_Vector));
}

template<typename _Vector>
void fan::basic_shape_position_vector<_Vector>::edit_data(void* data, uint_t offset, uint_t size)
{
	fan::edit_glbuffer(this->m_position_vbo, data, offset, size);
}

template<typename _Vector>
void fan::basic_shape_position_vector<_Vector>::write_data()
{
	fan::write_glbuffer(this->m_position_vbo, m_position.data(), sizeof(_Vector) * m_position.size());
}

template<typename _Vector>
void fan::basic_shape_position_vector<_Vector>::initialize_buffers(bool divisor)
{
	glBindBuffer(GL_ARRAY_BUFFER, m_position_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vector) * m_position.size(), m_position.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, _Vector::size(), fan::GL_FLOAT_T, GL_FALSE, sizeof(_Vector), 0);
	if (divisor) {
		glVertexAttribDivisor(1, 1);
	}
}

template<typename _Vector>
fan::basic_shape_size_vector<_Vector>::basic_shape_size_vector() : m_size_vbo(-1) {
	glGenBuffers(1, &m_size_vbo);
}

template<typename _Vector>
fan::basic_shape_size_vector<_Vector>::basic_shape_size_vector(const _Vector& size) : fan::basic_shape_size_vector<_Vector>() {
	this->basic_push_back(size);
}

template<typename _Vector>
fan::basic_shape_size_vector<_Vector>::~basic_shape_size_vector()
{
	glDeleteBuffers(1, &m_size_vbo);
}

template<typename _Vector>
_Vector fan::basic_shape_size_vector<_Vector>::get_size(std::uint64_t i) const
{
	return this->m_size[i];
}

template<typename _Vector>
void fan::basic_shape_size_vector<_Vector>::set_size(std::uint64_t i, const _Vector& size, bool queue)
{
	this->m_size[i] = size;
	if (!queue) {
		this->edit_data(i);
	}
}

template<typename _Vector>
void fan::basic_shape_size_vector<_Vector>::erase(std::uint64_t i, bool queue)
{
	this->m_size.erase(this->m_size.begin() + i);
	if (!queue) {
		this->write_data();
	}
}

template<typename _Vector>
void fan::basic_shape_size_vector<_Vector>::basic_push_back(const _Vector& size, bool queue)
{
	this->m_size.push_back(size);
	if (!queue) {
		this->write_data();
	}
}

template<typename _Vector>
void fan::basic_shape_size_vector<_Vector>::edit_data(uint_t i)
{
	fan::edit_glbuffer(this->m_size_vbo, m_size.data() + i, sizeof(_Vector) * i, sizeof(_Vector));
}

template<typename _Vector>
void fan::basic_shape_size_vector<_Vector>::edit_data(void* data, uint_t offset, uint_t size)
{
	fan::edit_glbuffer(this->m_size_vbo, data, offset, size);
}

template<typename _Vector>
void fan::basic_shape_size_vector<_Vector>::write_data()
{
	fan::write_glbuffer(m_size_vbo, m_size.data(), sizeof(_Vector) * m_size.size());
}

template<typename _Vector>
void fan::basic_shape_size_vector<_Vector>::initialize_buffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, m_size_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vector) * m_size.size(), m_size.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, _Vector::size(), fan::GL_FLOAT_T, GL_FALSE, sizeof(_Vector), 0);
	glVertexAttribDivisor(2, 1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template <typename _Vector>
fan::basic_shape_velocity_vector<_Vector>::basic_shape_velocity_vector() : m_velocity(1) {}

template<typename _Vector>
fan::basic_shape_velocity_vector<_Vector>::basic_shape_velocity_vector(const _Vector& velocity)
{
	this->m_velocity.push_back(velocity);
}

template <uint_t layout_location, uint_t gl_buffer>
fan::color fan::basic_shape_color_vector<layout_location, gl_buffer>::get_color(std::uint64_t i)
{
	return this->m_color[i];
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector<layout_location, gl_buffer>::set_color(std::uint64_t i, const fan::color& color, bool queue)
{
	this->m_color[i] = color;
	if (!queue) {
		write_data();
	}
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector<layout_location, gl_buffer>::erase(uint_t i, bool queue)
{
	this->m_color.erase(this->m_color.begin() + i);
	if (!queue) {
		this->write_data();
	}
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector<layout_location, gl_buffer>::basic_push_back(const fan::color& color, bool queue)
{
	this->m_color.push_back(color);
	if (!queue) {
		write_data();
	}
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector<layout_location, gl_buffer>::edit_data(uint_t i)
{
	fan::edit_glbuffer(this->m_color_vbo, this->m_color.data() + i, sizeof(fan::color) * i, sizeof(fan::color), gl_buffer, layout_location);
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector<layout_location, gl_buffer>::edit_data(void* data, uint_t offset, uint_t size)
{
	fan::edit_glbuffer(this->m_color_vbo, data, offset, size, gl_buffer, layout_location);
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector<layout_location, gl_buffer>::initialize_buffers(bool divisor)
{
	glBindBuffer(gl_buffer, m_color_vbo);
	glBufferData(gl_buffer, sizeof(fan::color) * m_color.size(), m_color.data(), GL_DYNAMIC_DRAW);
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

template <typename _Vector>
fan::basic_shape_vector<_Vector>::basic_shape_vector(fan::camera& camera, const fan::shader& shader)
	: m_shader(shader), m_camera(camera), m_window(camera.m_window)
{
	glGenVertexArrays(1, &m_vao);
}

template <typename _Vector>
fan::basic_shape_vector<_Vector>::basic_shape_vector(fan::camera& camera, const fan::shader& shader, const _Vector& position, const _Vector& size)
	: fan::basic_shape_position_vector<_Vector>(position), fan::basic_shape_size_vector<_Vector>(size), m_shader(shader), m_camera(camera), m_window(camera.m_window)
{
	glGenVertexArrays(1, &m_vao);
}

template <typename _Vector>
fan::basic_shape_vector<_Vector>::~basic_shape_vector()
{
	glDeleteVertexArrays(1, &m_vao);
	glValidateProgram(this->m_shader.id);
    int status = 0;
    glGetProgramiv(this->m_shader.id, GL_VALIDATE_STATUS, &status);
    if (status) {
        glDeleteProgram(this->m_shader.id);
    }
}

template <typename _Vector>
std::uint64_t fan::basic_shape_vector<_Vector>::size() const
{
	return this->m_position.size();
}

template<typename _Vector>
void fan::basic_shape_vector<_Vector>::basic_push_back(const _Vector& position, const _Vector& size, bool queue)
{
	fan::basic_shape_position_vector<_Vector>::basic_push_back(position, queue);
	fan::basic_shape_size_vector<_Vector>::basic_push_back(size, queue);
}

template<typename _Vector>
void fan::basic_shape_vector<_Vector>::erase(uint_t i, bool queue)
{
	fan::basic_shape_position_vector<_Vector>::erase(i, queue);
	fan::basic_shape_size_vector<_Vector>::erase(i, queue);
}

template<typename _Vector>
void fan::basic_shape_vector<_Vector>::edit_data(uint_t i, bool position, bool size)
{
	if (position) {
		fan::basic_shape_position_vector<_Vector>::edit_data(i);
	}
	if (size) {
		fan::basic_shape_size_vector<_Vector>::edit_data(i);
	}
}

template<typename _Vector>
void fan::basic_shape_vector<_Vector>::write_data(bool position, bool size)
{
	if (position) {
		fan::basic_shape_position_vector<_Vector>::write_data();
	}
	if (size) {
		fan::basic_shape_size_vector<_Vector>::write_data();
	}
}

template <typename _Vector>
void fan::basic_shape_vector<_Vector>::basic_draw(unsigned int mode, std::uint64_t count, std::uint64_t primcount, std::uint64_t i)
{
	glBindVertexArray(m_vao);
	if (i != (std::uint64_t)-1) {
		glDrawArraysInstancedBaseInstance(mode, 0, count, 1, i);
	}
	else {
		glDrawArraysInstanced(mode, 0, count, primcount);
	}

	glBindVertexArray(0);
}

template <typename _Vector>
fan::basic_vertice_vector<_Vector>::basic_vertice_vector(fan::camera& camera, const fan::shader& shader)
	: m_shader(shader), m_window(camera.m_window), m_camera(camera)
{
	glGenVertexArrays(1, &m_vao);
}

template<typename _Vector>
fan::basic_vertice_vector<_Vector>::basic_vertice_vector(fan::camera& camera, const fan::shader& shader, const fan::vec2& position, const fan::color& color)
	:  fan::basic_shape_position_vector<_Vector>(position), fan::basic_shape_color_vector<>(color), m_shader(shader), m_window(camera.m_window), m_camera(camera)
{
	glGenVertexArrays(1, &m_vao);
}

template<typename _Vector>
fan::basic_vertice_vector<_Vector>::~basic_vertice_vector()
{
	glDeleteVertexArrays(1, &m_vao);
	glValidateProgram(this->m_shader.id);
    int status = 0;
    glGetProgramiv(this->m_shader.id, GL_VALIDATE_STATUS, &status);
    if (status) {
        glDeleteProgram(this->m_shader.id);
    }
}

template<typename _Vector>
std::uint64_t fan::basic_vertice_vector<_Vector>::size() const
{
	return fan::basic_shape_position_vector<_Vector>::m_position.size();
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::basic_push_back(const _Vector& position, const fan::color& color, bool queue)
{
	fan::basic_shape_position_vector<_Vector>::basic_push_back(position, queue);
	fan::basic_shape_color_vector<>::basic_push_back(color, queue);
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::erase(uint_t i, bool queue)
{
	fan::basic_shape_position_vector<_Vector>::erase(i, queue);
	fan::basic_shape_color_vector<>::erase(i, queue);
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::edit_data(uint_t i, bool position, bool color)
{
	if (position) {
		fan::basic_shape_position_vector<_Vector>::edit_data(i);
	}
	if (color) {
		fan::basic_shape_color_vector<>::edit_data(i);
	}
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::write_data(bool position, bool color)
{
	if (position) {
		fan::basic_shape_position_vector<_Vector>::write_data();
	}
	if (color) {
		fan::basic_shape_color_vector<>::write_data();
	}
}

template<typename _Vector>
void fan::basic_vertice_vector<_Vector>::basic_draw(unsigned int mode, std::uint64_t count)
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
	this->m_color.push_back(color);

	this->write_data();
}

template <uint_t layout_location, uint_t gl_buffer>
fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::~basic_shape_color_vector_vector()
{
	glDeleteBuffers(1, &m_color_vbo);
}

template <uint_t layout_location, uint_t gl_buffer>
std::vector<fan::color> fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::get_color(std::uint64_t i)
{
	return this->m_color[i];
}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::set_color(std::uint64_t i, const std::vector<fan::color>& color, bool queue)
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
	this->m_color.push_back(color);
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

//template <uint_t layout_location, uint_t gl_buffer>
//void fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::edit_data(void* data, uint_t offset, uint_t size)
//{
//	fan::edit_glbuffer(this->m_color_vbo, data, offset, size, gl_buffer, layout_location);
//}

template <uint_t layout_location, uint_t gl_buffer>
void fan::basic_shape_color_vector_vector<layout_location, gl_buffer>::initialize_buffers(bool divisor)
{
	glBindBuffer(gl_buffer, m_color_vbo);
	glBufferData(gl_buffer, 0, nullptr, GL_DYNAMIC_DRAW);
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
	: basic_vertice_vector(camera, fan::shader(fan_2d::shader_paths::shape_vector_vs, fan_2d::shader_paths::shape_vector_fs)), m_index_restart(index_restart)
{ 
	glGenBuffers(1, &m_ebo);
	fan::bind_vao(this->m_vao, [&] {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_position_vector<fan::vec2>::initialize_buffers(false);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::initialize_buffers(false);
	});
}

fan_2d::vertice_vector::vertice_vector(fan::camera& camera, const fan::vec2& position, const fan::color& color, uint_t index_restart)
	: basic_vertice_vector(camera, fan::shader(fan_2d::shader_paths::shape_vector_vs, fan_2d::shader_paths::shape_vector_fs), position, color), m_index_restart(index_restart)
{
	glGenBuffers(1, &m_ebo);
	fan::bind_vao(this->m_vao, [&] {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_position_vector<fan::vec2>::initialize_buffers(false);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::initialize_buffers(false);
	});
}

void fan_2d::vertice_vector::release_queue(bool position, bool color, bool indices)
{
	if (position) {
		fan::basic_vertice_vector<fan::vec2>::basic_shape_position_vector<fan::vec2>::write_data();
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
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position_vector<fan::vec2>::basic_push_back(position, queue);
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::basic_push_back(color, queue);

	static uint32_t offset = 0;

	m_indices.push_back(offset);
	offset++;

	if (!(offset % this->m_index_restart) && !m_indices.empty()) {
		m_indices.push_back(UINT32_MAX);
	}
	if (!queue) {
		this->write_data();
	}
}

void fan_2d::vertice_vector::draw(uint32_t mode)
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size(), projection);

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

void fan_2d::vertice_vector::write_data()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

fan_2d::line_vector::line_vector(fan::camera& camera)
	: basic_shape_vector(camera, fan::shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs))
{
	fan::bind_vao(this->m_vao, [&] {
		fan::basic_shape_position_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_size_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_color_vector<>::initialize_buffers();
	});
}

fan_2d::line_vector::line_vector(fan::camera& camera, const fan::mat2& begin_end, const fan::color& color)
	: basic_shape_vector(camera, fan::shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), begin_end[0], begin_end[1]), basic_shape_color_vector(color)
{
	fan::bind_vao(this->m_vao, [&] {
		fan::basic_shape_position_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_size_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_color_vector<>::initialize_buffers();
	});
}

void fan_2d::line_vector::push_back(const fan::mat2& begin_end, const fan::color& color, bool queue)
{
	basic_shape_vector::basic_push_back(begin_end[0], begin_end[1], queue);
	m_color.push_back(color);

	if (!queue) {
		release_queue(false, true);
	}
}

void fan_2d::line_vector::draw()
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size(), projection);

	fan::mat4 view(1);
view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->m_shader.use();

	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::LINE));

	basic_shape_vector::basic_draw(GL_LINES, 2, size());
}

void fan_2d::line_vector::set_position(std::uint64_t i, const fan::mat2& begin_end, bool queue)
{
	basic_shape_vector::set_position(i, begin_end[0], true);
	basic_shape_vector::set_size(i, begin_end[1], true);

	if (!queue) {
		release_queue(true, false);
	}
}

void fan_2d::line_vector::release_queue(bool position, bool color)
{
	if (position) {
		basic_shape_vector::write_data(true, true);
	}
	if (color) {
		basic_shape_color_vector::write_data();
	}
}


//fan_2d::triangle_vector::triangle_vector(const fan::mat3x2& corners, const fan::color& color)
//{
//
//}
//
//void fan_2d::triangle_vector::set_position(std::uint64_t i, const fan::mat3x2& corners)
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

fan_2d::square_vector::square_vector(fan::camera& camera)
	: basic_shape_vector(camera, fan::shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs)) {
	fan::bind_vao(this->m_vao, [&] {
		fan::basic_shape_position_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_size_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_color_vector<>::initialize_buffers();
	});
}

fan_2d::square_vector::square_vector(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color)
	: basic_shape_vector(camera, fan::shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), position, size), basic_shape_color_vector(color)
{
	fan::bind_vao(this->m_vao, [&] {
		fan::basic_shape_color_vector<>::initialize_buffers();
		fan::basic_shape_position_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_size_vector<fan::vec2>::initialize_buffers();
	});
	
	this->m_velocity.resize(1);
}

fan_2d::square fan_2d::square_vector::construct(uint_t i)
{
	return fan_2d::square(
		m_camera,
		fan_2d::square_vector::m_position[i], 
		fan_2d::square_vector::m_size[i], 
		fan_2d::square_vector::m_color[i]
	);
}

void fan_2d::square_vector::release_queue(bool position, bool size, bool color)
{
	if (position) {
		basic_shape_vector::write_data(true, false);
	}
	if (size) {
		basic_shape_vector::write_data(false, true);
	}
	if (color) {
		basic_shape_color_vector::write_data();
	}
}

void fan_2d::square_vector::push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, bool queue)
{
	basic_shape_vector::basic_push_back(position, size, queue);
	m_color.push_back(color);
	this->m_velocity.push_back(fan::vec2());

	if (!queue) {
		release_queue(false, false, true);
	}
}

void fan_2d::square_vector::erase(uint_t i)
{
	basic_shape_vector::erase(i);
}

void fan_2d::square_vector::draw(std::uint64_t i)
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size(), projection);

	fan::mat4 view(1);
view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->m_shader.use();

	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::SQUARE));
	basic_shape_vector::basic_draw(GL_TRIANGLES, 6, size(), i);
}

fan::vec2 fan_2d::square_vector::center(uint_t i) const
{
	return fan_2d::square_vector::m_position[i] - fan_2d::square_vector::m_size[i] / 2;
}

fan_2d::square_corners_t fan_2d::square_vector::get_corners(uint_t i) const
{
	return fan_2d::square_corners_t::get_corners(this->get_position(i), this->get_size(i));
}

void fan_2d::square_vector::move(uint_t i, f32_t speed, f32_t gravity, f32_t jump_force, f32_t friction)
{
	move_object(m_window, this->m_position[i], this->m_velocity[i], speed, gravity, jump_force, friction);
	glBindBuffer(GL_ARRAY_BUFFER, m_position_vbo);
	fan::vec2 data;
	glGetBufferSubData(GL_ARRAY_BUFFER, sizeof(fan::vec2) * i, sizeof(fan::vec2), data.data());
	if (data != this->m_position[i]) {
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(fan::vec2) * i, sizeof(fan::vec2), this->m_position[i].data());
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

fan_2d::sprite_vector::sprite_vector(fan::camera& camera)
	: basic_shape_vector(camera, fan::shader(shader_paths::sprite_vector_vs, shader_paths::sprite_vector_fs)), m_texture(0)
{
	fan::bind_vao(this->m_vao, [&] {
		fan::basic_shape_position_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_size_vector<fan::vec2>::initialize_buffers();
	});
}

fan_2d::sprite_vector::sprite_vector(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size)
	: basic_shape_vector(camera, fan::shader(shader_paths::sprite_vector_vs, shader_paths::sprite_vector_fs), position, size)
{
	fan::bind_vao(this->m_vao, [&] {
		fan::basic_shape_position_vector<fan::vec2>::initialize_buffers();
		fan::basic_shape_size_vector<fan::vec2>::initialize_buffers();
	});

	image_info info = sprite::load_image(path);
	this->m_texture = info.texture_id;
	m_original_image_size = info.image_size;
	if (size == 0) {
		this->m_size[this->m_size.size() - 1] = info.image_size;
		basic_shape_vector::write_data(false, true);
	}
}

fan_2d::sprite_vector::~sprite_vector()
{
	glDeleteTextures(1, &m_texture);
}

void fan_2d::sprite_vector::push_back(const fan::vec2& position, const fan::vec2& size, bool queue)
{
	this->m_position.push_back(position);
	if (size == 0) {
		this->m_size.push_back(m_original_image_size);
	}
	else {
		this->m_size.push_back(size);
	}
	if (!queue) {
		release_queue(true, true);
	}
}

void fan_2d::sprite_vector::draw()
{
	fan::mat4 projection(1);
	projection = fan_2d::get_projection(m_window.get_size(), projection);

	fan::mat4 view(1);
view = m_camera.get_view_matrix(fan_2d::get_view_translation(m_window.get_size(), view));

	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("texture_sampler", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->m_texture);

	basic_shape_vector::basic_draw(GL_TRIANGLES, 6, size());
}

void fan_2d::sprite_vector::release_queue(bool position, bool size)
{
	if (position) {
		basic_shape_vector::write_data(true, false);
	}
	if (size) {
		basic_shape_vector::write_data(false, true);
	}
}

fan_2d::particles::particles(fan::camera& camera)
	: square_vector(camera) { }

void fan_2d::particles::add(const fan::vec2& position, const fan::vec2& size, const fan::vec2& velocity, const fan::color& color, std::uint64_t time)
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

fan_2d::gui::square::square(fan::camera& camera) : fan_2d::square(camera) {
	add_resize_callback(m_window, this->position);
}

fan_2d::gui::square::square(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color) 
	: fan_2d::square(camera, position, size, color)
{
	add_resize_callback(m_window, this->position);
}

fan_2d::gui::square_vector::square_vector(fan::camera& camera) : fan_2d::square_vector(camera) {
	m_window.add_resize_callback([&] {
		for (auto& i : this->m_position) {
			i += fan_2d::gui::get_resize_movement_offset(m_window);
		}
		fan_2d::gui::square_vector::basic_shape_vector::write_data(true, false);
	});
}

fan_2d::gui::square_vector::square_vector(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color) 
	: fan_2d::square_vector(camera, position, size, color)
{
	m_window.add_resize_callback([&] {
		for (auto& i : this->m_position) {
			i += fan_2d::gui::get_resize_movement_offset(m_window);
		}
		fan_2d::gui::square_vector::basic_shape_vector::write_data(true, false);
	});
}


fan_2d::gui::sprite::sprite(fan::camera& camera) : fan_2d::sprite(camera) {
	add_resize_callback(m_window, this->position);
}

fan_2d::gui::sprite::sprite(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size, f_t transparency)
	: fan_2d::sprite(camera, path, position, size, transparency) 
{
	add_resize_callback(m_window, this->position);
}

fan_2d::gui::sprite::sprite(fan::camera& camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size, f_t transparency) 
	: fan_2d::sprite(camera, pixels, position, size, transparency) 
{
	add_resize_callback(m_window, this->position);
}

fan_2d::gui::sprite_vector::sprite_vector(fan::camera& camera) : fan_2d::sprite_vector(camera) {
	m_window.add_resize_callback([&] {
		for (auto& i : this->m_position) {
			i += fan_2d::gui::get_resize_movement_offset(m_window);
		}
		fan_2d::gui::sprite_vector::write_data(true, false);
	});
}

fan_2d::gui::sprite_vector::sprite_vector(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size) 
	: fan_2d::sprite_vector(camera, path, position, size) 
{
	m_window.add_resize_callback([&] {
		for (auto& i : this->m_position) {
			i += fan_2d::gui::get_resize_movement_offset(m_window);
		}
		fan_2d::gui::sprite_vector::write_data(true, false);
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
			fan_2d::vertice_vector::push_back(fan::vec2(cos(t), sin(t)) * size.y / 2 + position, color, true);
		}
		else if (i == (int)(segments - 1)) {
			offset.x += (size.x - size.y);
			fan_2d::vertice_vector::push_back(fan::vec2(cos(t), sin(t)) * size.y / 2 + position, color, true);
		}

		fan_2d::vertice_vector::push_back(fan::vec2(cos(t), sin(t)) * size.y / 2 + position + offset, color, true);
	}
	if (!queue) {
		this->release_queue(true, true, true);
	}
	if (fan_2d::vertice_vector::m_index_restart == UINT32_MAX) {
		fan_2d::vertice_vector::m_index_restart = this->size();
	}
	fan_2d::gui::rounded_rectangle::m_position.push_back(position);
	fan_2d::gui::rounded_rectangle::m_size.push_back(fan::vec2(size.x, size.y));
	data_offset.push_back(this->size());
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
	
	basic_shape_position_vector::edit_data(fan_2d::vertice_vector::m_position.data() + previous_offset, sizeof(fan::vec2) * previous_offset, sizeof(fan::vec2) * (offset - previous_offset));
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

	fan_2d::image_info image = fan_2d::sprite::load_image("fonts/arial.png");

	this->m_original_image_size = image.image_size;
	this->m_texture = image.texture_id;
	
	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_texture_vbo);
	glGenBuffers(1, &m_vertex_vbo);
	glGenBuffers(1, &m_letter_ssbo);

	auto font_info = fan::io::file::parse_font("fonts/arial.fnt");
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
	glDeleteVertexArrays(1, &m_vao);
	glDeleteBuffers(1, &m_texture_vbo);
	glDeleteBuffers(1, &m_vertex_vbo);
	glDeleteBuffers(1, &m_letter_ssbo);
	glDeleteTextures(1, &m_texture);
	glValidateProgram(this->m_shader.id);
    int status = 0;
    glGetProgramiv(this->m_shader.id, GL_VALIDATE_STATUS, &status);
    if (status) {
        glDeleteProgram(this->m_shader.id);
    }
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
	if (!queue) {
		this->write_vertices();
	}
}

void fan_2d::gui::text_renderer::set_font_size(uint_t i, f_t font_size, bool queue) {
	if (this->get_font_size(i)[0] == font_size) {
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

	auto position = this->get_position(i);
	auto text_color = text_color_t::get_color(i)[0];
	auto outline_color = outline_color_t::get_color(i)[0];
	auto font_size = this->m_font_size[i][0];
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

void fan_2d::gui::text_renderer::free_queue() {
	this->write_data();
}

void fan_2d::gui::text_renderer::push_back(const std::string& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color, bool queue) {
	if (text.empty()) {
		fan::print("text renderer string is empty");
		exit(1);
	}
	m_text.push_back(text);
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
	fan::mat4 projection = fan::ortho(0, window_size.x, window_size.y, 0);

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

std::vector<f32_t> fan_2d::gui::text_renderer::get_font_size(uint_t i)
{
	return this->m_font_size[i];
}

void fan_2d::gui::text_renderer::load_characters(uint_t i, fan::vec2 position, const std::string& text, bool edit, bool insert) {
	const f_t converted_font_size = 1.0 / m_original_font_size * m_font_size[i][0];

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

	advance += fan_2d::gui::font_properties::get_gap_size(m_font[letter].m_advance);
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

	advance += fan_2d::gui::font_properties::get_gap_size(m_font[letter].m_advance);
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

	advance += fan_2d::gui::font_properties::get_gap_size(m_font[letter].m_advance);
}

void fan_2d::gui::text_renderer::write_vertices()
{
	std::vector<fan::vec2> vertices;

	for (uint_t i = 0; i < m_vertices.size(); i++) {
		vertices.insert(vertices.end(), m_vertices[i].begin(), m_vertices[i].end());
	}

	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fan::vec2) * fan::vector_size(vertices), vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void fan_2d::gui::text_renderer::write_texture_coordinates()
{
	std::vector<fan::vec2> texture_coordinates;

	for (uint_t i = 0; i < m_texture_coordinates.size(); i++) {
		texture_coordinates.insert(texture_coordinates.end(), m_texture_coordinates[i].begin(), m_texture_coordinates[i].end());
	}

	glBindBuffer(GL_ARRAY_BUFFER, m_texture_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fan::vec2) * fan::vector_size(texture_coordinates), texture_coordinates.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void fan_2d::gui::text_renderer::write_font_sizes()
{
	std::vector<f32_t> font_sizes;

	for (uint_t i = 0; i < m_font_size.size(); i++) {
		font_sizes.insert(font_sizes.end(), m_font_size[i].begin(), m_font_size[i].end());
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_letter_ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(font_sizes[0]) * fan::vector_size(font_sizes), font_sizes.data(), GL_DYNAMIC_DRAW);
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

void fan_3d::add_camera_rotation_callback(fan::camera& camera) {
	camera.m_window.add_mouse_move_callback(std::bind(&fan::camera::rotate_camera, camera, 0));
}

fan_3d::line_vector::line_vector(fan::camera& camera)
	: basic_shape_vector(camera, fan::shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs)), basic_shape_color_vector()
{
	glBindVertexArray(m_vao);

	fan::basic_shape_color_vector<>::initialize_buffers();
	fan::basic_shape_position_vector<fan::vec3>::initialize_buffers();
	fan::basic_shape_size_vector<fan::vec3>::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

fan_3d::line_vector::line_vector(fan::camera& camera, const fan::mat2x3& begin_end, const fan::color& color)
	: basic_shape_vector(camera, fan::shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs),
		begin_end[0], begin_end[1]), basic_shape_color_vector(color)
{
	glBindVertexArray(m_vao);

	fan::basic_shape_color_vector<>::initialize_buffers();
	fan::basic_shape_position_vector<fan::vec3>::initialize_buffers();
	fan::basic_shape_size_vector<fan::vec3>::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void fan_3d::line_vector::push_back(const fan::mat2x3& begin_end, const fan::color& color, bool queue)
{
	basic_shape_color_vector::basic_push_back(color, queue);
	basic_shape_vector::basic_push_back(begin_end[0], begin_end[1], queue);
}

void fan_3d::line_vector::draw() {

	fan::mat4 projection(1);
	projection = fan::perspective(fan::radians(90.f), (f32_t)m_window.get_size().x / (f32_t)m_window.get_size().y, 0.1f, 1000.0f);

	fan::mat4 view(m_camera.get_view_matrix());


	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::LINE));

	basic_shape_vector::basic_draw(GL_LINE_STRIP, 2, size());
}

void fan_3d::line_vector::set_position(std::uint64_t i, const fan::mat2x3 begin_end, bool queue)
{
	basic_shape_vector::set_position(i, begin_end[0], queue);
	basic_shape_vector::set_size(i, begin_end[1], queue);
}

void fan_3d::line_vector::release_queue(bool position, bool color)
{
	if (position) {
		basic_shape_vector::write_data(true, true);
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
			this->m_triangle_vertices.push_back(fan::vec3(x * triangle_size, y * triangle_size, noise.GetNoise(mesh_height.x, mesh_height.y) * 100));
			index++;
			texture_coordinates.push_back(fan::vec2(x * triangle_size, y * triangle_size) / texture_scale);
			mesh_height.x += mesh_size.x;
		}
		mesh_height.y += mesh_size.y;
	}

	for (uint_t y = 0; y < map_size.y - 1; y++) {
		for (uint_t x = 0; x < map_size.x; x++) {
			for (uint_t i = 0; i < 2; i++) {
				this->m_indices.push_back(((y + i) * map_size.x) + x);
			}
		}
		for (uint_t x = map_size.x - 1; x != (uint_t)-1; x--) {
			this->m_indices.push_back(((y + 1) * map_size.x) + x);
			if (x == map_size.x - 1 || !x) {
				continue;
			}
			this->m_indices.push_back(((y + 1) * map_size.x) + x);
		}
	}


	unsigned int first_corners[] = { 1, 0, map_size.x };

	for (uint_t i = 0; i + first_corners[2] < map_size.x * map_size.y; i++) {
		auto v = fan::normalize(fan::cross(
			m_triangle_vertices[first_corners[2] + i] - m_triangle_vertices[first_corners[0] + i],
			m_triangle_vertices[first_corners[1] + i] - m_triangle_vertices[first_corners[0] + i]
		));
		if (v.z < 0) {
			v = -v;
		}
		normals.push_back(
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
	glBufferData(GL_ARRAY_BUFFER, m_vertice_size * m_triangle_vertices.size(), m_triangle_vertices.data(), GL_DYNAMIC_DRAW);
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
	glDeleteTextures(1, &m_texture);
	glDeleteVertexArrays(1, &m_vao);
	glDeleteBuffers(1, &m_texture_vbo);
	glDeleteBuffers(1, &m_vertices_vbo);
	glDeleteBuffers(1, &m_ebo);
	glDeleteBuffers(1, &m_normals_vbo);
	glDeleteProgram(m_shader.id);
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
	fan_3d::terrain_generator::m_triangle_vertices.push_back(vertices);

	std::fill_n(std::back_inserter(basic_shape_color_vector::m_color), 4, color);

	if (!queue) {
		basic_shape_color_vector::write_data();
		fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
	}
}

void fan_3d::terrain_generator::edit_data(std::uint64_t i, const triangle_vertices_t& vertices, const fan::color& color)
{
	basic_shape_color_vector::m_color[i] = color;
	fan_3d::terrain_generator::m_triangle_vertices[i] = vertices;

	glBindBuffer(GL_ARRAY_BUFFER, fan_3d::terrain_generator::m_vertices_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, fan_3d::terrain_generator::m_vertice_size * i, fan_3d::terrain_generator::m_vertice_size, vertices.data());
	glBindBuffer(GL_ARRAY_BUFFER, basic_shape_color_vector::m_color_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(fan::color) * i, sizeof(fan::color), color.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void fan_3d::terrain_generator::release_queue()
{
	basic_shape_color_vector::write_data();
	fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
}

void fan_3d::terrain_generator::draw() {
	fan::mat4 projection(1);
	projection = fan::perspective(fan::radians(90.f), (f32_t)m_window.get_size().x / (f32_t)m_window.get_size().y, 0.1f, 1000.0f);

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

fan_3d::square_vector::square_vector(fan::camera& camera, const std::string& path, std::uint64_t block_size)
	: basic_shape_vector(camera, fan::shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs)),
	block_size(block_size)
{
	glBindVertexArray(m_vao);

	fan::basic_shape_color_vector<>::initialize_buffers();
	fan::basic_shape_position_vector<fan::vec3>::initialize_buffers();
	fan::basic_shape_size_vector<fan::vec3>::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	generate_textures(path, block_size);
}

fan_3d::square_vector::square_vector(fan::camera& camera, const fan::color& color, std::uint64_t block_size)
	: basic_shape_vector(camera, fan::shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs)),
	block_size(block_size)
{
	glBindVertexArray(m_vao);

	fan::basic_shape_color_vector<>::initialize_buffers();
	fan::basic_shape_position_vector<fan::vec3>::initialize_buffers();
	fan::basic_shape_size_vector<fan::vec3>::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	//TODO
	//generate_textures(path, block_size);
}

fan_3d::square_vector::~square_vector()
{
	glDeleteTextures(1, &m_texture);
	glDeleteBuffers(1, &m_texture_ssbo);
	glDeleteBuffers(1, &m_texture_id_ssbo);
}

void fan_3d::square_vector::push_back(const fan::vec3& src, const fan::vec3& dst, const fan::vec2& texture_id, bool queue)
{
	basic_shape_vector::basic_push_back(src, dst, queue);

	this->m_textures.push_back(texture_id.y * m_amount_of_textures.x + texture_id.x);

	if (!queue) {
		this->write_textures();
	}
}

fan::vec3 fan_3d::square_vector::get_src(uint_t i) const
{
	return this->m_position[i];
}

fan::vec3 fan_3d::square_vector::get_dst(uint_t i) const
{
	return this->m_size[i];
}

fan::vec3 fan_3d::square_vector::get_size(uint_t i) const
{
	return this->get_dst(i) - this->get_src(i);
}

void fan_3d::square_vector::set_position(uint_t i, const fan::vec3& src, const fan::vec3& dst, bool queue)
{
	this->m_position[i] = src;
	this->m_size[i] = dst;

	if (!queue) {
		fan::basic_shape_vector<fan::vec3>::write_data(true, true);
	}
}

void fan_3d::square_vector::set_size(uint_t i, const fan::vec3& size, bool queue)
{
	fan::basic_shape_vector<fan::vec3>::set_size(i, this->get_src(i) + size, queue);
}

void fan_3d::square_vector::draw() {

	fan::mat4 projection(1);
	projection = fan::perspective(fan::radians(90.f), (f32_t)m_window.get_size().x / (f32_t)m_window.get_size().y, 0.1f, 1000.0f);

	fan::mat4 view(m_camera.get_view_matrix());

	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	this->m_shader.set_int("shape_type", fan::eti(fan::e_shapes::SQUARE));
	this->m_shader.set_int("texture_sampler", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->m_texture);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_texture_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	basic_shape_vector::basic_draw(GL_TRIANGLES, 36, size());
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void fan_3d::square_vector::set_texture(std::uint64_t i, const fan::vec2& texture_id, bool queue)
{
	this->m_textures[i] = (f_t)block_size.x / 6 * texture_id.y + texture_id.x;

	if (!queue) {
		write_textures();
	}
}

void fan_3d::square_vector::generate_textures(const std::string& path, const fan::vec2& block_size)
{
	glGenBuffers(1, &m_texture_ssbo);
	glGenBuffers(1, &m_texture_id_ssbo);

	auto texture_info = fan_2d::sprite::load_image(path, true);

	m_texture = texture_info.texture_id;
	fan::vec2i image_size = texture_info.image_size;

	glBindTexture(GL_TEXTURE_2D, m_texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	const fan::vec2 texturepack_size = fan::vec2(image_size.x / block_size.x, image_size.y / block_size.y);
	m_amount_of_textures = fan::vec2(texturepack_size.x / 6, texturepack_size.y);
	// side_order order = { bottom, top, front, back, right, left }
	// top, bottom, 
	constexpr int side_order[] = { 1, 0, 2, 3, 4, 5 };
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
					textures.push_back(coordinate);
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

void fan_3d::square_vector::write_textures()
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

void fan_3d::square_vector::release_queue(bool position, bool size, bool textures)
{
	if (position) {
		basic_shape_vector::write_data(true, false);
	}
	if (size) {
		basic_shape_vector::write_data(false, true);
	}
	if (textures) {
		this->write_textures();
	}
}

fan_3d::square_corners fan_3d::square_vector::get_corners(std::uint64_t i) const
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

fan_3d::skybox::skybox(
	fan::window& window,
	const std::string& left,
	const std::string& right,
	const std::string& front,
	const std::string back,
	const std::string bottom,
	const std::string& top
) : shader(fan_3d::shader_paths::skybox_vs, fan_3d::shader_paths::skybox_fs), camera(window) {
	std::array<std::string, 6> images{ right, left, top, bottom, back, front };
	glGenTextures(1, &texture_id);

	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id);
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

	glGenVertexArrays(1, &skybox_vao);
	glGenBuffers(1, &skybox_vbo);
	glBindVertexArray(skybox_vao);
	glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, fan::GL_FLOAT_T, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glBindVertexArray(0);
}

fan_3d::skybox::~skybox() {
	glDeleteVertexArrays(1, &skybox_vao);
	glDeleteBuffers(1, &skybox_vbo);
	glDeleteTextures(1, &texture_id);
}

//void fan_3d::skybox::draw() {
//	shader.use();
//
//	fan::mat4 view(1);
//	fan::mat4 projection(1);
//
//	view = fan::mat4(mat3(camera->get_view_matrix()));
//	projection = perspective(fan::radians(90.f), (f32_t)fan::window_size.x / (f32_t)fan::window_size.y, 0.1f, 1000.0f);
//
//	shader.set_mat4("view", view);
//	shader.set_mat4("projection", projection);
//	shader.set_vec3("fog_color", fan::vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));
//
//	glDepthFunc(GL_LEQUAL);
//	glBindVertexArray(skybox_vao);
//	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id);
//	glDrawArrays(GL_TRIANGLES, 0, 36);
//	glBindVertexArray(0);
//	glDepthFunc(GL_LESS);
//}

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
		GL_DYNAMIC_DRAW
	);
#endif
}

void fan_3d::model::draw() {

	fan::mat4 model(1);
	model = translate(model, get_position());
	model = scale(model, get_size());

	this->m_shader.use();

	fan::mat4 projection(1);
	projection = fan::perspective(fan::radians(90.f), (f32_t)m_window.get_size().x / (f32_t)m_window.get_size().y, 0.1f, 1000.0f);

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
		glDrawElementsInstanced(GL_TRIANGLES, this->meshes[i].indices.size(), GL_UNSIGNED_INT, 0, 1);
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

	const auto t = fan::dot(normal, ray_begin - p0) / fan::dot(-lab, normal);
	const auto u = fan::dot(fan::cross(p02, -lab), ray_begin - p0) / fan::dot(-lab, normal);
	const auto v = fan::dot(fan::cross(-lab, p01), ray_begin - p0) / fan::dot(-lab, normal);

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

	const auto t = fan::dot(normal, line[0] - triangle[0]) / fan::dot(-lab, normal);
	const auto u = fan::dot(fan::cross(p02, -lab), line[0] - triangle[0]) / fan::dot(-lab, normal);
	const auto v = fan::dot(fan::cross(-lab, p01), line[0] - triangle[0]) / fan::dot(-lab, normal);

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

	auto result = fan::dot((square[2] - line[0]), plane_normal);
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