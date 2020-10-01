#include <FAN/Graphics.hpp>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <functional>
#include <numeric>

mat4 fan_2d::frame_projection;
mat4 fan_2d::frame_view;
mat4 fan_3d::frame_projection;
mat4 fan_3d::frame_view;

Texture::Texture() : texture(0), width(0), height(0), VBO(0), VAO(0) { }

vec2i window_position() {
	vec2i position;
	glfwGetWindowPos(window, &position.x, &position.y);
	return position;
}

bmp LoadBMP(const char* path, Texture& texture) {
	FILE* file = fopen(path, "rb");
	if (!file) {
		printf("wrong path %s", path);
		exit(1);
	}
	fseek(file, 0, SEEK_END);
	uint64_t size = ftell(file);
	fseek(file, 0, SEEK_SET);
	bmp data;
	data.data = (unsigned char*)malloc(size);
	if (data.data) {
		static_cast<void>(fread(data.data, 1, size, file) + 1);
	}
	fclose(file);

	uint32_t pixelOffset = *(uint32_t*)(data.data + BMP_Offsets::PIXELDATA);
	texture.width = *(uint32_t*)(data.data + BMP_Offsets::WIDTH);
	texture.height = *(uint32_t*)(data.data + BMP_Offsets::HEIGHT);
	data.image = (data.data + pixelOffset);
	return data;
}

uint64_t _2d_1d(vec2 position) {
	return (int(position.x / block_size)) +
		int(position.y / block_size) * (window_size.y / block_size);
}

Camera::Camera(vec3 position, vec3 up, float yaw, float pitch) : front(vec3(0.0f, 0.0f, -1.0f)) {
	this->position = position;
	this->worldUp = up;
	this->yaw = yaw;
	this->pitch = pitch;
	this->update_vectors();
}

void Camera::move(bool noclip, f_t movement_speed)
{
	constexpr double accel = -40;

	constexpr double jump_force = 100;
	if (!noclip) {
		this->velocity.x /= this->friction * delta_time + 1;
		this->velocity.z /= this->friction * delta_time + 1;
	}
	else {
		this->velocity /= this->friction * delta_time + 1;
	}
	static constexpr auto magic_number = 0.001;
	if (this->velocity.x < magic_number && this->velocity.x > -magic_number) {
		this->velocity.x = 0;
	}
	if (this->velocity.y < magic_number && this->velocity.y > -magic_number) {
		this->velocity.y = 0;
	}
	if (this->velocity.z < magic_number && this->velocity.z > -magic_number) {
		this->velocity.z = 0;
	}
	if (glfwGetKey(window, GLFW_KEY_W)) {
		const vec2 direction(DirectionVector(radians(this->yaw)));
		this->velocity.x += direction.x * (movement_speed * delta_time);
		this->velocity.z += direction.y * (movement_speed * delta_time);
	}
	if (glfwGetKey(window, GLFW_KEY_S)) {
		const vec2 direction(DirectionVector(radians(this->yaw)));
		this->velocity.x -= direction.x * (movement_speed * delta_time);
		this->velocity.z -= direction.y * (movement_speed * delta_time);
	}
	if (glfwGetKey(window, GLFW_KEY_A)) {
		this->velocity -= this->right * (movement_speed * delta_time);
	}
	if (glfwGetKey(window, GLFW_KEY_D)) {
		this->velocity += this->right * (movement_speed * delta_time);
	}
	if (!noclip) {
		if (glfwGetKey(window, GLFW_KEY_SPACE)) {
			this->velocity.y += jump_force * delta_time;
		}
		this->velocity.y += accel * delta_time;
	}
	else {
		if (glfwGetKey(window, GLFW_KEY_SPACE)) {
			this->velocity.y += movement_speed * delta_time;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) {
			this->velocity.y -= movement_speed * delta_time;
		}
	}
	this->position += this->velocity * delta_time;
	this->update_vectors();
}

void Camera::rotate_camera(bool when) // this->updateCameraVectors(); move function updates
{
	if (when) {
		return;
	}

	static float lastX, lastY;
	float xpos = cursor_position.x;
	float ypos = cursor_position.y;

	float& yaw = fan_3d::camera.yaw;
	float& pitch = fan_3d::camera.pitch;

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.05f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	if (yaw > 180) {
		yaw = -180;
	}
	if (yaw < -180) {
		yaw = 180;
	}

	if (pitch > 89)
		pitch = 89;
	if (pitch < -89)
		pitch = -89;
}

mat4 Camera::get_view_matrix() {
	return look_at(this->position, (this->position + (this->front)), (this->up));
}

mat4 Camera::get_view_matrix(mat4 m) {
	return m * look_at(this->position, (this->position + (this->front)).rounded(), (this->up).rounded());
}

vec3 Camera::get_position() const {
	return this->position;
}

void Camera::set_position(const vec3& position) {
	this->position = position;
}

void Camera::update_vectors() {
	vec3 front;
	front.x = cos(radians(this->yaw)) * cos(radians(this->pitch));
	front.y = sin(radians(this->pitch));
	front.z = sin(radians(this->yaw)) * cos(radians(this->pitch));
	this->front = normalize(front);
	// Also re-calculate the Right and Up vector
	this->right = normalize(cross(this->front, this->worldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
	this->up = normalize(cross(this->right, this->front));
}

Camera fan_2d::camera(vec3(), vec3(0, 1, 0), -90, 0);
Camera fan_3d::camera;

int load_texture(const std::string_view path, const std::string& directory, bool flip_image, bool alpha) {
	std::string file_name = std::string(directory + (directory.empty() ? "" : "/") + path.data());
	GLuint texture_id;
	glGenTextures(1, &texture_id);

	int width, height;

	stbi_set_flip_vertically_on_load(flip_image);
	unsigned char* image = SOIL_load_image(file_name.c_str(), &width, &height, 0, alpha ? SOIL_LOAD_RGBA : SOIL_LOAD_RGB);

	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, alpha ? GL_RGBA : GL_RGB, width, height, 0, alpha ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, image);
	glGenerateMipmap(GL_TEXTURE_2D);
	SOIL_free_image_data(image);
	glBindTexture(GL_TEXTURE_2D, 0);

	return texture_id;
}

fan_2d::basic_single_shape::basic_single_shape()
{
	glGenVertexArrays(1, &vao);
}

fan_2d::basic_single_shape::basic_single_shape(const Shader& shader, const vec2& position, const vec2& size)
	: shader(shader), position(position), size(size)
{
	glGenVertexArrays(1, &vao);
}

fan_2d::basic_single_shape::~basic_single_shape()
{
	glDeleteVertexArrays(1, &this->vao);
}

vec2 fan_2d::basic_single_shape::get_position() const
{
	return position;
}

vec2 fan_2d::basic_single_shape::get_size() const
{
	return this->size;
}

vec2 fan_2d::basic_single_shape::get_velocity() const
{
	return fan_2d::basic_single_shape::velocity;
}

void fan_2d::basic_single_shape::set_size(const vec2& size)
{
	this->size = size;
}

void fan_2d::basic_single_shape::set_position(const vec2& position)
{
	this->position = position;
}

void fan_2d::basic_single_shape::set_velocity(const vec2& velocity)
{
	fan_2d::basic_single_shape::velocity = velocity;
}

void fan_2d::basic_single_shape::basic_draw(GLenum mode, GLsizei count)
{
	glBindVertexArray(vao);
	glDrawArrays(mode, 0, count);
	glBindVertexArray(0);
}

void fan_2d::basic_single_shape::move(f_t speed, f_t gravity, f_t jump_force, f_t friction)
{
	if (gravity != 0) {
		if (glfwGetKey(window, GLFW_KEY_SPACE) && is_colliding) { // AND COLLIDING
			this->velocity.y = jump_force;
		}
		else {
			this->velocity.y += gravity * delta_time;
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

	if (key_press(GLFW_KEY_W)) {
		velocity.y -= speed * delta_time;
	}
	if (key_press(GLFW_KEY_S)) {
		velocity.y += speed * delta_time;
	}
	if (key_press(GLFW_KEY_A)) {
		velocity.x -= speed * delta_time;
	}
	if (key_press(GLFW_KEY_D)) {
		velocity.x += speed * delta_time;
	}
	if constexpr(std::is_same<decltype(velocity.x), f_t>::value) {
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

bool fan_2d::basic_single_shape::inside() const
{
	if (cursor_position.x >= position.x && cursor_position.x <= position.x + size.x &&
		cursor_position.y >= position.y && cursor_position.y <= position.y + size.y)
	{
		return true;
	}
	return false;
}

fan_2d::basic_single_color::basic_single_color() {}

fan_2d::basic_single_color::basic_single_color(const Color& color) : color(color) {}

Color fan_2d::basic_single_color::get_color()
{
	return this->color;
}

void fan_2d::basic_single_color::set_color(const Color& color)
{
	this->color = color;
}

fan_2d::line::line() : basic_single_shape(Shader(shader_paths::single_shapes_path_vs, shader_paths::single_shapes_path_fs), vec2(), vec2()), fan_2d::basic_single_color() {}

fan_2d::line::line(const mat2x2& begin_end, const Color& color)
	: basic_single_shape(Shader(shader_paths::single_shapes_path_vs, shader_paths::single_shapes_path_fs), begin_end[0], begin_end[1]),
	  fan_2d::basic_single_color(color) {}

void fan_2d::line::draw()
{
	this->shader.use();

	this->shader.set_mat4("projection", fan_2d::frame_projection);
	this->shader.set_mat4("view", fan_2d::frame_view);
	this->shader.set_vec4("shape_color", get_color());
	this->shader.set_int("shape_type", eti(shape_types::LINE));
	this->shader.set_vec2("begin", get_position());
	this->shader.set_vec2("end", get_size());

	fan_2d::basic_single_shape::basic_draw(GL_LINES, 2);
}

void fan_2d::line::set_position(const mat2x2& begin_end)
{
	fan_2d::line::set_position(vec2(begin_end[0]));
	set_size(begin_end[1]);
}

fan_2d::square::square()
	: basic_single_shape(
		Shader(shader_paths::single_shapes_path_vs, shader_paths::single_shapes_path_fs),
		vec2(), 
		vec2()
	), fan_2d::basic_single_color() {
}

fan_2d::square::square(const vec2& position, const vec2& size, const Color& color)
	: basic_single_shape(
		Shader(shader_paths::single_shapes_path_vs, shader_paths::single_shapes_path_fs),
		position, size
	), fan_2d::basic_single_color(color) {}

void fan_2d::square::draw()
{
	this->shader.use();

	mat4 model(1);
	model = translate(model, get_position());
	model = scale(model, get_size());

	this->shader.set_mat4("projection", fan_2d::frame_projection);
	this->shader.set_mat4("view", fan_2d::frame_view);
	this->shader.set_mat4("model", model);
	this->shader.set_vec4("shape_color", get_color());
	this->shader.set_int("shape_type", eti(shape_types::SQUARE));

	fan_2d::basic_single_shape::basic_draw(GL_TRIANGLES, 6);
}

fan_2d::bloom_square::bloom_square()
	: fan_2d::basic_single_shape(Shader(shader_paths::single_bloom_shapes_path_vs, shader_paths::single_bloom_shapes_path_fs), vec2(), vec2())
{
	glGenFramebuffers(1, &m_hdr_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, m_hdr_fbo);
	glGenTextures(2, m_color_buffers);

	for (unsigned int i = 0; i < 2; i++)
	{
		glBindTexture(GL_TEXTURE_2D, m_color_buffers[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, window_size.x, window_size.y, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // we clamp to the edge as the blur filter would otherwise sample repeated texture values!
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		// attach texture to framebuffer
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, m_color_buffers[i], 0);
	}

	glGenRenderbuffers(1, &m_rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, m_rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, window_size.x, window_size.y);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_rbo);
	// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, attachments);
	// finally check if framebuffer is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		LOG("Framebuffer not complete!");
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glGenFramebuffers(2, m_pong_fbo);
	glGenTextures(2, m_pong_color_buffer);
	for (unsigned int i = 0; i < 2; i++)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[i]);
		glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, window_size.x, window_size.y, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // we clamp to the edge as the blur filter would otherwise sample repeated texture values!
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_pong_color_buffer[i], 0);
		// also check if framebuffers are complete (no need for depth buffer)
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete!" << std::endl;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_shader_blur.use();
	m_shader_blur.set_int("image", 0);
	m_shader_bloom.use();
	m_shader_bloom.set_int("scene", 0);
	m_shader_bloom.set_int("bloomBlur", 1);
}

fan_2d::bloom_square::bloom_square(const vec2& position, const vec2& size, const Color& color)
	: fan_2d::bloom_square::bloom_square() 
{
	this->set_position(position);
	this->set_size(size);
	this->set_color(color);

}

void fan_2d::bloom_square::bind_fbo() const
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_hdr_fbo);
}

unsigned int bbb = 0;
unsigned int ccc;
static void renderQuad()
{
	if (bbb == 0)
	{

		/*
		vec2(0, 0),
		vec2(0, 1),
		vec2(1, 1),
		vec2(1, 1),
		vec2(1, 0),
		vec2(0, 0)

		*/
		float quadVertices[] = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &bbb);
		glGenBuffers(1, &ccc);
		glBindVertexArray(bbb);
		glBindBuffer(GL_ARRAY_BUFFER, ccc);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	}
	glBindVertexArray(bbb);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}


void fan_2d::bloom_square::draw()
{
	this->shader.use();

	mat4 model(1);
	model = translate(model, get_position());
	model = scale(model, get_size());

	this->shader.set_mat4("projection", fan_2d::frame_projection);
	this->shader.set_mat4("view", fan_2d::frame_view);
	this->shader.set_mat4("model", model);
	this->shader.set_vec4("shape_color", get_color());
	this->shader.set_int("shape_type", eti(shape_types::SQUARE));

	fan_2d::basic_single_shape::basic_draw(GL_TRIANGLES, 6);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// 2. blur bright fragments with two-pass Gaussian Blur 
	// --------------------------------------------------
	bool horizontal = true, first_iteration = true;
	unsigned int amount = 10;
	m_shader_blur.use();

	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[1]);
	m_shader_blur.set_int("horizontal", 1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_color_buffers[1]);  // bind texture of other framebuffer (or scene if first iteration)

	renderQuad();

	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[0]);
	m_shader_blur.set_int("horizontal", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[1]);  // bind texture of other framebuffer (or scene if first iteration)

	renderQuad();


	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[1]);
	m_shader_blur.set_int("horizontal", 1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[0]);  // bind texture of other framebuffer (or scene if first iteration)
	renderQuad();

	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[0]);
	m_shader_blur.set_int("horizontal", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[1]);

	renderQuad();

	glBindFramebuffer(GL_FRAMEBUFFER, m_pong_fbo[1]);
	m_shader_blur.set_int("horizontal", 1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[0]);
	renderQuad();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	// 3. now render floating point color buffer to 2D quad and tonemap HDR colors to default framebuffer's (clamped) color range
	// --------------------------------------------------------------------------------------------------------------------------

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m_shader_bloom.use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_color_buffers[0]);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, m_pong_color_buffer[!horizontal]);
	m_shader_bloom.set_float("exposure", 1.0f);
	renderQuad();

}

fan_2d::sprite::sprite() :
	basic_single_shape(Shader(shader_paths::single_sprite_path_vs, shader_paths::single_sprite_path_fs), vec2(), vec2()) {}

fan_2d::sprite::sprite(const std::string& path, const vec2& position, const vec2& size)
	: basic_single_shape(Shader(shader_paths::single_sprite_path_vs, shader_paths::single_sprite_path_fs), position, size) {
	auto texture_info = load_image(path);
	this->texture = texture_info.texture_id;
	vec2 image_size = texture_info.image_size;
	if (size != 0) {
		image_size = size;
	}
	set_size(image_size);
}

fan_2d::sprite::sprite(unsigned char* pixels, const vec2& position, const vec2i& size)
	: basic_single_shape(Shader(shader_paths::single_sprite_path_vs, shader_paths::single_sprite_path_fs), position, size) 
{
	auto texture_info = load_image(pixels, size);
	this->texture = texture_info.texture_id;
	vec2 image_size = texture_info.image_size;
	if (size != 0) {
		image_size = size;
	}
	set_size(image_size);
}

void fan_2d::sprite::reload_image(unsigned char* pixels, const vec2i& size)
{
	glBindTexture(GL_TEXTURE_2D, this->texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, pixels);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void fan_2d::sprite::reload_image(const std::string& path, const vec2i& size)
{
	std::ifstream file(path);
	if (!file.good()) {
		LOG("sprite loading error: File path does not exist for ", path.c_str());
		return;
	}
	glBindTexture(GL_TEXTURE_2D, this->texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	vec2i image_size;

	stbi_set_flip_vertically_on_load(true);

	unsigned char* data = SOIL_load_image(
		path.c_str(),
		&image_size.x,
		&image_size.y,
		0,
		SOIL_LOAD_RGBA
	);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_size.x, image_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	SOIL_free_image_data(data);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void fan_2d::sprite::draw()
{
	shader.use();

	mat4 model(1);
	model = translate(model, get_position());
	model = scale(model, get_size());
	//model = Rotate(model, radians(get_rotation()), vec3(0, 0, 1));

	shader.set_mat4("projection", fan_2d::frame_projection);
	shader.set_mat4("view", fan_2d::frame_view);
	shader.set_mat4("model", model);
	shader.set_int("texture_sampler", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	fan_2d::basic_single_shape::basic_draw(GL_TRIANGLES, 6);
}

f_t fan_2d::sprite::get_rotation()
{
	return this->m_rotation;
}

void fan_2d::sprite::set_rotation(f_t degrees)
{
	this->m_rotation = degrees;
}

fan_2d::image_info fan_2d::sprite::load_image(const std::string& path, bool flip_image)
{
	std::ifstream file(path);
	if (!file.good()) {
		LOG("sprite loading error: File path does not exist for ", path.c_str());
		return image_info{};
	}

	unsigned int texture_id = 0;

	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	vec2i image_size;

	if (flip_image) {
		stbi_set_flip_vertically_on_load(true);
	}
	int components = 0;
	
	unsigned char* data = stbi_load(
		path.c_str(),
		&image_size.x,
		&image_size.y,
		&components,
		SOIL_LOAD_RGBA
	);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	GLenum format;
	if (components == 1)
		format = GL_RED;
	else if (components == 3)
		format = GL_RGB;
	else if (components == 4)
		format = GL_RGBA;
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_size.x, image_size.y, 0, GL_ABGR_EXT, GL_UNSIGNED_INT_8_8_8_8, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	SOIL_free_image_data(data);
	glBindTexture(GL_TEXTURE_2D, 0);
	return { image_size, texture_id };
}

fan_2d::image_info fan_2d::sprite::load_image(unsigned char* pixels, const vec2i& size)
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
	return { vec2i(1920, 1080), texture_id };
}

fan_2d::animation::animation(const vec2& position, const vec2& size) : basic_single_shape(Shader(shader_paths::single_sprite_path_vs, shader_paths::single_sprite_path_fs), position, size) {}

void fan_2d::animation::add(const std::string& path)
{
	auto texture_info = fan_2d::sprite::load_image(path);
	this->m_textures.push_back(texture_info.texture_id);
	vec2 image_size = texture_info.image_size;
	if (size != 0) {
		image_size = size;
	}
	this->set_size(image_size);
}

void fan_2d::animation::draw(std::uint64_t texture)
{
	shader.use();

	mat4 model(1);
	model = translate(model, get_position());

	model = scale(model, get_size());

	shader.set_mat4("projection", fan_2d::frame_projection);
	shader.set_mat4("view", fan_2d::frame_view);
	shader.set_mat4("model", model);
	shader.set_int("texture_sampler", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_textures[texture]);

	fan_2d::basic_single_shape::basic_draw(GL_TRIANGLES, 6);
}

void write_vbo(unsigned int buffer, void* data, std::uint64_t size)
{
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template class basic_shape_vector<vec2>;
template class basic_shape_vector<vec3>;
template class basic_shape_vector<vec4>;

template <typename _Vector>
basic_shape_vector<_Vector>::basic_shape_vector(const Shader& shader)
	: m_shader(shader)
{
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &position_vbo);
	glGenBuffers(1, &size_vbo);
}

template <typename _Vector>
basic_shape_vector<_Vector>::basic_shape_vector(const Shader& shader, const _Vector& position, const _Vector& size)
	: basic_shape_vector::basic_shape_vector(shader)
{
	this->m_position.push_back(position);
	this->m_size.push_back(size);
}

template <typename _Vector>
basic_shape_vector<_Vector>::~basic_shape_vector()
{
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &position_vbo);
	glDeleteBuffers(1, &size_vbo);
}

template <typename _Vector>
_Vector basic_shape_vector<_Vector>::get_size(std::uint64_t i) const
{
	return this->m_size[i];
}

template <typename _Vector>
void basic_shape_vector<_Vector>::set_size(std::uint64_t i, const _Vector& size, bool queue)
{
	this->m_size[i] = size;
	if (!queue) {
		write_vbo(size_vbo, m_size.data(), sizeof(_Vector) * m_size.size());
	}
}

template<typename _Vector>
std::vector<_Vector> basic_shape_vector<_Vector>::get_positions() const
{
	return this->m_position;
}

template <typename _Vector>
_Vector basic_shape_vector<_Vector>::get_position(std::uint64_t i) const
{
	return this->m_position[i];
}

template<typename _Vector>
void basic_shape_vector<_Vector>::set_position(std::uint64_t i, const _Vector& position, bool queue)
{
	this->m_position[i] = position;
	if (!queue) {
		write_vbo(position_vbo, m_position.data(), sizeof(vec2) * m_position.size());
	}
}

template<typename _Vector>
void basic_shape_vector<_Vector>::basic_push_back(const _Vector& position, const _Vector& size, bool queue)
{
	this->m_position.push_back(position);
	this->m_size.push_back(size);

	if (!queue) {
		write_vbo(position_vbo, m_position.data(), sizeof(m_position[0]) * m_position.size());
		write_vbo(size_vbo, m_size.data(), sizeof(m_size[0]) * m_size.size());
	}
}

template<typename _Vector>
void basic_shape_vector<_Vector>::erase(std::uint64_t i)
{
	m_position.erase(m_position.begin() + i);
	m_size.erase(m_size.begin() + i);
	
	write_vbo(position_vbo, m_position.data(), sizeof(m_position[0]) * m_position.size());
	write_vbo(size_vbo, m_size.data(), sizeof(m_size[0]) * m_size.size());
}

template <typename _Vector>
std::uint64_t basic_shape_vector<_Vector>::size() const
{
	return this->m_position.size();
}

template<typename _Vector>
bool basic_shape_vector<_Vector>::empty() const
{
	return !basic_shape_vector<_Vector>::size();
}

template<typename _Vector>
void basic_shape_vector<_Vector>::write_data(bool position, bool size)
{
	if (position) {
		write_vbo(position_vbo, m_position.data(), sizeof(_Vector) * m_position.size());
	}
	if (size) {
		write_vbo(size_vbo, m_size.data(), sizeof(_Vector) * m_size.size());
	}
}

template <typename _Vector>
void basic_shape_vector<_Vector>::initialize_buffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vector) * m_position.size(), m_position.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, _Vector::size(), GL_FLOAT_T, GL_FALSE, sizeof(_Vector), 0);
	glVertexAttribDivisor(1, 1);

	glBindBuffer(GL_ARRAY_BUFFER, size_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vector) * m_size.size(), m_size.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, _Vector::size(), GL_FLOAT_T, GL_FALSE, sizeof(_Vector), 0);
	glVertexAttribDivisor(2, 1);
}

template <typename _Vector>
void basic_shape_vector<_Vector>::basic_draw(unsigned int mode, std::uint64_t count, std::uint64_t primcount)
{
	glDepthFunc(GL_LEQUAL);
	glBindVertexArray(vao);
	glDrawArraysInstanced(mode, 0, count, primcount);
	glBindVertexArray(0);
	glDepthFunc(GL_FALSE);
}

basic_shape_color_vector::basic_shape_color_vector()
{
	glGenBuffers(1, &color_vbo);
}

basic_shape_color_vector::basic_shape_color_vector(const Color& color)
	: basic_shape_color_vector()
{
	this->m_color.push_back(color);

	write_vbo(color_vbo, m_color.data(), sizeof(Color) * m_color.size());
}

basic_shape_color_vector::~basic_shape_color_vector()
{
	glDeleteBuffers(1, &color_vbo);
}

Color basic_shape_color_vector::get_color(std::uint64_t i)
{
	return this->m_color[i];
}

void basic_shape_color_vector::set_color(std::uint64_t i, const Color& color, bool queue)
{
	this->m_color[i] = color;
	if (!queue) {
		write_data();
	}
}

void basic_shape_color_vector::basic_push_back(const Color& color, bool queue)
{
	this->m_color.push_back(color);
	write_data();
}

void basic_shape_color_vector::write_data()
{
	write_vbo(color_vbo, m_color.data(), sizeof(Color) * m_color.size());
}

void basic_shape_color_vector::initialize_buffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Color) * m_color.size(), m_color.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT_T, GL_FALSE, sizeof(Color), 0);
	glVertexAttribDivisor(0, 1);
}

fan_2d::line_vector::line_vector() 
	: basic_shape_vector(Shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs)),
	  basic_shape_color_vector() {
	glBindVertexArray(vao);

	basic_shape_color_vector::initialize_buffers();
	basic_shape_vector::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

fan_2d::line_vector::line_vector(const mat2& begin_end, const Color& color)
	: basic_shape_vector(Shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), begin_end[0], begin_end[1]),
	  basic_shape_color_vector(color) 
{
	glBindVertexArray(vao);

	basic_shape_color_vector::initialize_buffers();
	basic_shape_vector::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void fan_2d::line_vector::push_back(const mat2& begin_end, const Color& color, bool queue)
{
	basic_shape_vector::basic_push_back(begin_end[0], begin_end[1], queue);
	m_color.push_back(color);

	if (!queue) {
		release_queue(false, true);
	}
}

void fan_2d::line_vector::draw()
{
	this->m_shader.use();

	this->m_shader.set_mat4("projection", fan_2d::frame_projection);
	this->m_shader.set_mat4("view", fan_2d::frame_view);
	this->m_shader.set_int("shape_type", eti(shape_types::LINE));

	basic_shape_vector::basic_draw(GL_LINES, 2, size());
}

void fan_2d::line_vector::set_position(std::uint64_t i, const mat2& begin_end, bool queue)
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


fan_2d::triangle_vector::triangle_vector(const mat3x2& corners, const Color& color)
	: basic_shape_vector(Shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs)),
	basic_shape_color_vector(color) 
{

	glGenBuffers(1, &l_vbo);
	glGenBuffers(1, &m_vbo);
	glGenBuffers(1, &r_vbo);

	glBindVertexArray(vao);

	basic_shape_color_vector::initialize_buffers();

	vec2 l = corners[0];
	vec2 m = corners[1];
	vec2 r = corners[2];

	glBindBuffer(GL_ARRAY_BUFFER, l_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(l), l.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 2, GL_FLOAT_T, GL_FALSE, sizeof(l), 0);
	glVertexAttribDivisor(3, 1);

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m), m.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 2, GL_FLOAT_T, GL_FALSE, sizeof(m), 0);
	glVertexAttribDivisor(4, 1);

	glBindBuffer(GL_ARRAY_BUFFER, r_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(r), r.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 2, GL_FLOAT_T, GL_FALSE, sizeof(r), 0);
	glVertexAttribDivisor(5, 1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	m_lcorners.push_back(corners[0]);
	m_mcorners.push_back(corners[1]);
	m_rcorners.push_back(corners[2]);
}

void fan_2d::triangle_vector::set_position(std::uint64_t i, const mat3x2& corners)
{
	this->m_lcorners[i] = corners[0];
	this->m_mcorners[i] = corners[1];
	this->m_rcorners[i] = corners[2];

	write_vbo(l_vbo, m_lcorners.data(), sizeof(m_lcorners[0]) * m_lcorners.size());
	write_vbo(m_vbo, m_mcorners.data(), sizeof(m_mcorners[0]) * m_mcorners.size());
	write_vbo(r_vbo, m_rcorners.data(), sizeof(m_rcorners[0]) * m_rcorners.size());
}

void fan_2d::triangle_vector::push_back(const mat3x2& corners, const Color& color)
{
	m_lcorners.push_back(corners[0]);
	m_mcorners.push_back(corners[1]);
	m_rcorners.push_back(corners[2]);

	write_vbo(l_vbo, m_lcorners.data(), sizeof(m_lcorners[0]) * m_lcorners.size());
	write_vbo(m_vbo, m_mcorners.data(), sizeof(m_mcorners[0]) * m_mcorners.size());
	write_vbo(r_vbo, m_rcorners.data(), sizeof(m_rcorners[0]) * m_rcorners.size());

	m_color.push_back(color);

	basic_shape_color_vector::write_data();
}

void fan_2d::triangle_vector::draw()
{
	this->m_shader.use();

	this->m_shader.set_mat4("projection", fan_2d::frame_projection);
	this->m_shader.set_mat4("view", fan_2d::frame_view);
	this->m_shader.set_int("shape_type", eti(shape_types::TRIANGLE));

	basic_shape_vector::basic_draw(GL_TRIANGLES, 3, m_lcorners.size());
}

fan_2d::square_vector::square_vector()
	: basic_shape_vector(Shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs)) {
	glBindVertexArray(vao);

	basic_shape_color_vector::initialize_buffers();
	basic_shape_vector::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

fan_2d::square_vector::square_vector(const vec2& position, const vec2& size, const Color& color)
	: basic_shape_vector(Shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), position, size),
	  basic_shape_color_vector(color)
{
	glBindVertexArray(vao);

	basic_shape_color_vector::initialize_buffers();
	basic_shape_vector::initialize_buffers();

	m_icorners.push_back(mat2x2(position, position + size));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
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

void fan_2d::square_vector::push_back(const vec2& position, const vec2& size, const Color& color, bool queue)
{
	basic_shape_vector::basic_push_back(position, size, queue);
	m_color.push_back(color);

	m_icorners.push_back(mat2x2(position, position + size));

	if (!queue) {
		release_queue(false, false, true);
	}
}

void fan_2d::square_vector::erase(uint_t i)
{
	basic_shape_vector::erase(i);
	this->m_icorners.erase(this->m_icorners.begin() + i);
}

void fan_2d::square_vector::draw()
{
	this->m_shader.use();

	this->m_shader.set_mat4("projection", fan_2d::frame_projection);
	this->m_shader.set_mat4("view", fan_2d::frame_view);
	this->m_shader.set_int("shape_type", eti(shape_types::SQUARE));

	basic_shape_vector::basic_draw(GL_TRIANGLES, 6, size());
}

std::vector<mat2x2> fan_2d::square_vector::get_icorners() const
{

	return fan_2d::square_vector::m_icorners;
}

fan_2d::sprite_vector::sprite_vector()
	: basic_shape_vector(Shader(shader_paths::sprite_vector_vs, shader_paths::sprite_vector_fs))
{
	glBindVertexArray(vao);

	basic_shape_vector::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

fan_2d::sprite_vector::sprite_vector(const std::string& path, const vec2& position, const vec2& size)
	: basic_shape_vector(Shader(shader_paths::sprite_vector_vs, shader_paths::sprite_vector_fs), position, size)
{
	glBindVertexArray(vao);

	basic_shape_vector::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	image_info info = sprite::load_image(path, true);
	this->texture = info.texture_id;
	original_image_size = info.image_size;
	if (size == 0) {
		this->m_size[this->m_size.size() - 1] = info.image_size;
		write_vbo(size_vbo, m_size.data(), sizeof(vec2) * m_size.size());
	}
}

fan_2d::sprite_vector::~sprite_vector()
{
	glDeleteTextures(1, &texture);
}

void fan_2d::sprite_vector::push_back(const vec2& position, const vec2& size, bool queue)
{
	this->m_position.push_back(position);
	if (size == 0) {
		this->m_size.push_back(original_image_size);
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
	this->m_shader.use();
	this->m_shader.set_mat4("projection", fan_2d::frame_projection);
	this->m_shader.set_mat4("view", fan_2d::frame_view);
	this->m_shader.set_int("texture_sampler", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->texture);

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

void fan_2d::particles::add(const vec2& position, const vec2& size, const vec2& velocity, const Color& color, std::uint64_t time)
{
	this->push_back(position, size, color);
	this->m_particles.push_back({ velocity, Timer(Timer::start(), time) });
}

void fan_2d::particles::update()
{
	for (int i = 0; i < this->size(); i++) {
		if (this->m_particles[i].m_timer.finished()) {
			this->erase(i);
			this->m_particles.erase(this->m_particles.begin() + i);
			continue;
		}
		this->set_position(i, this->get_position(i) + this->m_particles[i].m_velocity * delta_time, true);
	}
	this->release_queue(true, false, false);
}

void fan_3d::add_camera_movement_callback() {
	callback::cursor_move.add(std::bind(&Camera::rotate_camera, fan_3d::camera, 0));
}

fan_3d::line_vector::line_vector()
	: basic_shape_vector(Shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs)),
	basic_shape_color_vector() 
{
	glBindVertexArray(vao);

	basic_shape_color_vector::initialize_buffers();
	basic_shape_vector::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

fan_3d::line_vector::line_vector(const mat2x3& begin_end, const Color& color)
	: basic_shape_vector(Shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs),
		begin_end[0], begin_end[1]),
	basic_shape_color_vector(color)
{
	glBindVertexArray(vao);

	basic_shape_color_vector::initialize_buffers();
	basic_shape_vector::initialize_buffers();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void fan_3d::line_vector::push_back(const mat2x3& begin_end, const Color& color, bool queue)
{
	basic_shape_color_vector::basic_push_back(color, queue);
	basic_shape_vector::basic_push_back(begin_end[0], begin_end[1], queue);
}

void fan_3d::line_vector::draw() {

	this->m_shader.use();
	this->m_shader.set_mat4("projection", fan_3d::frame_projection);
	this->m_shader.set_mat4("view", fan_3d::frame_view);
	this->m_shader.set_int("shape_type", eti(shape_types::LINE));

	basic_shape_vector::basic_draw(GL_LINE_STRIP, 2, size());
}

void fan_3d::line_vector::set_position(std::uint64_t i, const mat2x3 begin_end, bool queue)
{
	basic_shape_vector::set_position(i, begin_end[0], queue);
	basic_shape_vector::set_size(i, begin_end[1], queue);
}

void fan_3d::line_vector::release_queue(bool position, bool color)
{
	if (color) {	
		basic_shape_color_vector::write_data();
	}
}
#include 	<io.h>
fan_3d::triangle_vector::triangle_vector()
	: m_shader(fan_3d::shader_paths::triangle_vector_vs, fan_3d::shader_paths::triangle_vector_fs)
{
	/*for (int j = 0; j < 3; j++)
	for (auto i : m_base_indices) {
		m_indices.push_back(i + 4 * j);
	}*/
	m_indices.push_back(0);
	m_indices.push_back(2);
	m_indices.push_back(1);

	m_indices.push_back(2);
	m_indices.push_back(1);
	m_indices.push_back(3);

	m_indices.push_back(0 + 4);
	m_indices.push_back(2 + 4);
	m_indices.push_back(1 +4);

	m_indices.push_back(2 + 4);
	m_indices.push_back(1 + 4);
	m_indices.push_back(3 + 4);

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vertices_vbo);
	glGenBuffers(1, &m_ebo);
	glBindVertexArray(m_vao);
	basic_shape_color_vector::initialize_buffers();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertices_vbo);
	glBufferData(GL_ARRAY_BUFFER, m_vertice_size * m_triangle_vertices.size(), m_triangle_vertices.data(), GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, vec3::size(), GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void fan_3d::triangle_vector::push_back(const triangle_vertices_t& vertices, const Color& color, bool queue) {

	/*for (auto i : m_base_indices) {
		LOG(i + 4 * fan_3d::triangle_vector::size());
		m_indices.push_back(i + 4 * fan_3d::triangle_vector::size());
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);*/

	fan_3d::triangle_vector::m_triangle_vertices.push_back(vertices);

	std::fill_n(std::back_inserter(basic_shape_color_vector::m_color), 4, color);

	if (!queue) {
		basic_shape_color_vector::write_data();
		write_vbo(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
	}
}

fan_3d::triangle_vertices_t fan_3d::triangle_vector::get_vertices(std::uint64_t i)
{
	return fan_3d::triangle_vector::m_triangle_vertices[i];
}

void fan_3d::triangle_vector::release_queue()
{
	basic_shape_color_vector::write_data();
	write_vbo(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
}

void fan_3d::triangle_vector::draw() {
	fan_3d::triangle_vector::m_shader.use();
	fan_3d::triangle_vector::m_shader.set_mat4("projection", fan_3d::frame_projection);
	fan_3d::triangle_vector::m_shader.set_mat4("view", fan_3d::frame_view);
	glBindVertexArray(m_vao);
	glDrawElements(GL_TRIANGLE_STRIP, 6 * fan_3d::triangle_vector::size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

uint_t fan_3d::triangle_vector::size() {
	return fan_3d::triangle_vector::m_triangle_vertices.size();
}

fan_3d::square_vector::square_vector(const std::string& path, std::uint64_t block_size)
	: basic_shape_vector(Shader(fan_3d::shader_paths::shape_vector_vs, fan_3d::shader_paths::shape_vector_fs)),
	block_size(block_size)
{
	glBindVertexArray(vao);

	basic_shape_vector::initialize_buffers();

	glBindVertexArray(0);

	generate_textures(path, block_size);
}

void fan_3d::square_vector::push_back(const vec3& position, const vec3& size, const vec2& texture_id, bool queue)
{
	basic_shape_vector::basic_push_back(position, size, queue);

	this->m_textures.push_back(block_size.x / 6 * texture_id.y + texture_id.x);

	if (!queue) {
		this->write_textures();
	}
}

void fan_3d::square_vector::draw() {

	this->m_shader.use();
	this->m_shader.set_mat4("projection", fan_3d::frame_projection);
	this->m_shader.set_mat4("view", fan_3d::frame_view);
	this->m_shader.set_int("shape_type", eti(shape_types::SQUARE));
	this->m_shader.set_int("texture_sampler", 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->m_texture);
	
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_texture_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);

	basic_shape_vector::basic_draw(GL_TRIANGLES, 36, size());
}

void fan_3d::square_vector::set_texture(std::uint64_t i, const vec2& texture_id, bool queue)
{
	this->m_textures[i] = block_size.x / 6 * texture_id.y + texture_id.x;

	if (!queue) {
		write_textures();
	}
}

void fan_3d::square_vector::generate_textures(const std::string& path, const vec2& block_size)
{
	glGenBuffers(1, &m_texture_ssbo);
	glGenBuffers(1, &m_texture_id_ssbo);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	vec2i image_size;

	stbi_set_flip_vertically_on_load(true);
	unsigned char* data = SOIL_load_image(path.data(), &image_size.x, &image_size.y, 0, SOIL_LOAD_RGBA);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_size.x, image_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	SOIL_free_image_data(data);
	glBindTexture(GL_TEXTURE_2D, 0);

	const vec2 texturepack_size = vec2(image_size.x / block_size.x, image_size.y / block_size.y);
	vec2 amount_of_textures = vec2(texturepack_size.x / 6, texturepack_size.y);
	constexpr int side_order[] = { 0, 1, 4, 5, 3, 2 };
	std::vector<vec2> textures;
	for (vec2i texture; texture.y < amount_of_textures.y; texture.y++) {
		const vec2 begin(1.f / texturepack_size.x, 1.f / texturepack_size.y);
		const float up = 1 - begin.y * texture.y;
		const float down = 1 - begin.y * (texture.y + 1);
		for (texture.x = 0; texture.x < amount_of_textures.x; texture.x++) {
			for (int side = 0; side < ArrLen(side_order); side++) {
				const float left = begin.x * side_order[side] + ((begin.x * (texture.x)) * 6);
				const float right = begin.x * (side_order[side] + 1) + ((begin.x * (texture.x)) * 6);
				const vec2 texture_coordinates[] = {
					vec2(left,  up),
					vec2(left,  down),
					vec2(right, down),
					vec2(right, down),
					vec2(right, up),
					vec2(left,  up)
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

std::vector<float> g_distances(6);

int numX = 512,
numY = 512,
numOctaves = 7;

double persistence = 0.5;

#define maxPrimeIndex 10
int primeIndex = 0;

int primes[maxPrimeIndex][3] = {
  { 995615039, 600173719, 701464987 },
  { 831731269, 162318869, 136250887 },
  { 174329291, 946737083, 245679977 },
  { 362489573, 795918041, 350777237 },
  { 457025711, 880830799, 909678923 },
  { 787070341, 177340217, 593320781 },
  { 405493717, 291031019, 391950901 },
  { 458904767, 676625681, 424452397 },
  { 531736441, 939683957, 810651871 },
  { 997169939, 842027887, 423882827 }
};

double Noise(int i, int x, int y) {
	int n = x + y * 57;
	n = (n << 13) ^ n;
	int a = primes[i][0], b = primes[i][1], c = primes[i][2];
	int t = (n * (n * n * a + b) + c) & 0x7fffffff;
	return 1.0 - (double)(t) / 1073741824.0;
}

double SmoothedNoise(int i, int x, int y) {
	double corners = (Noise(i, x - 1, y - 1) + Noise(i, x + 1, y - 1) +
		Noise(i, x - 1, y + 1) + Noise(i, x + 1, y + 1)) / 16,
		sides = (Noise(i, x - 1, y) + Noise(i, x + 1, y) + Noise(i, x, y - 1) +
			Noise(i, x, y + 1)) / 8,
		center = Noise(i, x, y) / 4;
	return corners + sides + center;
}

double Interpolate(double a, double b, double x) {
	double ft = x * 3.1415927,
		f = (1 - cos(ft)) * 0.5;
	return  a * (1 - f) + b * f;
}

double InterpolatedNoise(int i, double x, double y) {
	int integer_X = x;
	double fractional_X = x - integer_X;
	int integer_Y = y;
	double fractional_Y = y - integer_Y;

	double v1 = SmoothedNoise(i, integer_X, integer_Y),
		v2 = SmoothedNoise(i, integer_X + 1, integer_Y),
		v3 = SmoothedNoise(i, integer_X, integer_Y + 1),
		v4 = SmoothedNoise(i, integer_X + 1, integer_Y + 1),
		i1 = Interpolate(v1, v2, fractional_X),
		i2 = Interpolate(v3, v4, fractional_X);
	return Interpolate(i1, i2, fractional_Y);
}

vec3 line_plane_intersection3d(const da_t<f_t, 2, 3>& line, const da_t<f_t, 4, 3>& square) {
	const da_t<f_t, 3> n = normalize(cross(square[0] - square[2], square[3] - square[2]));
	const f_t nl_dot(dot(n, line[1]));

	if (!nl_dot) {
		return vec3(INFINITY);
	}

	const f_t d = dot(square[2] - line[0], n) / nl_dot;
	if (d <= 0) {
		return vec3(INFINITY);
	}
	if (distance(vec3(line[0]), vec3(line[0] + line[1])) < d) {
		return vec3(INFINITY);
	}
	const vec3 intersect = line[0] + line[1] * d;
	if (intersect.y > square[3][1] && intersect.y < square[0][1] &&
		intersect.z > square[0][2] && intersect.z < square[3][2]) {
		return intersect;
	}

	return vec3(INFINITY);
}

//vec3 line_plane_intersection3d(const vec3 plane_position, const vec3& plane_size, const vec3& position, const vec3& direction) {
//	const vec3 p0 = plane_position - plane_size / 2.f;
//	const vec3 a = (p0 + vec3(0, plane_size.y, 0));
//	const vec3 b = (p0 + vec3(0, 0, plane_size.x));
//	const vec3 n = normalize(cross(a - p0, b - p0));
//	const f_t nl_dot(dot(n, direction));
//
//	if (!nl_dot) {
//		return vec3(-1);
//	}
//
//	const f_t d = dot(p0 - position, n) / nl_dot;
//	if (d <= 0) {
//		return vec3(INFINITY);
//	}
//	const vec3 intersect = position + direction * d;
//	if (intersect.y > b.y && intersect.y < a.y &&
//		intersect.z > a.z && intersect.z < b.z) {
//		return intersect;
//	}
//
//	return vec3(INFINITY);
//}

vec3 intersection_point3d(const vec3& plane_position, const vec3& plane_size, const vec3& position, e_cube side)
{
	vec3 p0;
	vec3 a;
	vec3 b;
	const vec3 l0 = position;

	switch (side) {
	case e_cube::left: {
		p0 = plane_position - vec3(plane_size.x / 2, plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(0, plane_size.y, 0);
		b = p0 + vec3(plane_size.x, 0, 0);
		break;
	}
	case e_cube::right: {
		p0 = plane_position - vec3(plane_size.x / 2, plane_size.y / 2, -plane_size.z / 2);
		a = p0 + vec3(0, plane_size.y, 0);
		b = p0 + vec3(plane_size.x, 0, 0);
		break;
	}
	case e_cube::front: {
		p0 = plane_position - vec3(plane_size.x / 2, plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(0, plane_size.y, 0);
		b = p0 + vec3(0, 0, plane_size.z);
		break;
	}
	case e_cube::back: {
		p0 = plane_position - vec3(-plane_size.x / 2, plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(0, plane_size.y, 0);
		b = p0 + vec3(0, 0, plane_size.z);
		break;
	}
	case e_cube::up: {
		p0 = plane_position - vec3(plane_size.x / 2, -plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(plane_size.x, 0, 0);
		b = p0 + vec3(0, 0, plane_size.z);
		break;
	}
	case e_cube::down: {
		p0 = plane_position - vec3(plane_size.x / 2, plane_size.y / 2, plane_size.z / 2);
		a = p0 + vec3(plane_size.x, 0, 0);
		b = p0 + vec3(0, 0, plane_size.z);
		break;
	}
	}

	const vec3 n = normalize(cross((a - p0), (b - p0)));

	const vec3 l = DirectionVector(fan_3d::camera.yaw, fan_3d::camera.pitch);

	const float nl_dot(dot(n, l));

	if (!nl_dot) {
		return vec3(-1);
	}

	const float d = dot(p0 - l0, n) / nl_dot;
	if (d <= 0) {
		return vec3(-1);
	}

	g_distances[eti(side)] = d;

	const vec3 intersect(l0 + l * d);
	switch (side) {
	case e_cube::right:
	case e_cube::left: {
		if (intersect.y > b.y && intersect.y < a.y &&
			intersect.x > a.x && intersect.x < b.x) {
			return intersect;
		}
		break;
	}
	case e_cube::back:
	case e_cube::front: {
		if (intersect.y > b.y && intersect.y < a.y &&
			intersect.z > a.z && intersect.z < b.z) {
			return intersect;
		}
		break;
	}
	case e_cube::up:
	case e_cube::down: {
		if (intersect.x > b.x && intersect.x < a.x &&
			intersect.z > a.z && intersect.z < b.z) {
			return intersect;
		}
		break;
	}
	}

	return vec3(-1);
}

double ValueNoise_2D(double x, double y) {
	double total = 0,
		frequency = pow(2, numOctaves),
		amplitude = 1;
	for (int i = 0; i < numOctaves; ++i) {
		frequency /= 2;
		amplitude *= persistence;
		total += InterpolatedNoise((primeIndex + i) % maxPrimeIndex,
			x / frequency, y / frequency) * amplitude;
	}
	return total / frequency;
}

fan_3d::skybox::skybox(
	const std::string& left,
	const std::string& right,
	const std::string& front,
	const std::string back,
	const std::string bottom,
	const std::string& top
) : shader(fan_3d::shader_paths::skybox_vs, fan_3d::shader_paths::skybox_fs), camera(&fan_3d::camera) {
	std::array<std::string, 6> images{ right, left, top, bottom, back, front };
	glGenTextures(1, &texture_id);

	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_NEAREST);

	for (int i = 0; i < images.size(); i++) {
		vec2i image_size;
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
	glVertexAttribPointer(0, 3, GL_FLOAT_T, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
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
//	mat4 view(1);
//	mat4 projection(1);
//
//	view = mat4(mat3(camera->get_view_matrix()));
//	projection = perspective(radians(90.f), (f_t)window_size.x / (f_t)window_size.y, 0.1f, 1000.0f);
//
//	shader.set_mat4("view", view);
//	shader.set_mat4("projection", projection);
//	shader.set_vec3("fog_color", vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));
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
	glVertexAttribPointer(3, 3, GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), 0);

	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), reinterpret_cast<void*>(offsetof(mesh_vertex, normal)));

	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 2, GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), reinterpret_cast<void*>(offsetof(mesh_vertex, texture_coordinates)));
	glBindVertexArray(0);
}

fan_3d::model_loader::model_loader(const std::string& path, const vec3& size) {
	load_model(path, size);
}

void fan_3d::model_loader::load_model(const std::string& path, const vec3& size) {
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

	if (scene == nullptr || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr) {
		std::cout << "assimp error: " << importer.GetErrorString() << '\n';
		return;
	}

	directory = path.substr(0, path.find_last_of('/'));

	process_node(scene->mRootNode, scene, size);
}

void fan_3d::model_loader::process_node(aiNode* node, const aiScene* scene, const vec3& size) {
	for (GLuint i = 0; i < node->mNumMeshes; i++) {
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

		meshes.emplace_back(process_mesh(mesh, scene, size));
	}

	for (GLuint i = 0; i < node->mNumChildren; i++) {
		process_node(node->mChildren[i], scene, size);
	}
}

fan_3d::model_mesh fan_3d::model_loader::process_mesh(aiMesh* mesh, const aiScene* scene, const vec3& size) {
	std::vector<mesh_vertex> vertices;
	std::vector<GLuint> indices;
	std::vector<mesh_texture> textures;

	for (GLuint i = 0; i < mesh->mNumVertices; i++)
	{
		mesh_vertex vertex;
		vec3 vector;

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
			vertex.normal = vec3();
		}

		if (mesh->mTextureCoords[0]) {
			vec2 vec;
			vec.x = mesh->mTextureCoords[0][i].x;
			vec.y = mesh->mTextureCoords[0][i].y;
			vertex.texture_coordinates = vec;
		}

		vertices.emplace_back(vertex);
	}

	for (GLuint i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (GLuint j = 0; j < face.mNumIndices; j++) {
			indices.emplace_back(face.mIndices[j]);
		}
	}

	aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

	std::vector<mesh_texture> diffuseMaps = this->load_material_textures(material, aiTextureType_DIFFUSE, "texture_diffuse");
	textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

	std::vector<mesh_texture> specularMaps = this->load_material_textures(material, aiTextureType_SPECULAR, "texture_specular");
	textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

	if (textures.empty()) {
		mesh_texture texture;
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

		texture.id = texture_id;
		textures.emplace_back(texture);
		textures_loaded.emplace_back(texture);
	}
	return model_mesh(vertices, indices, textures);
}

std::vector<fan_3d::mesh_texture> fan_3d::model_loader::load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name) {
	std::vector<mesh_texture> textures;

	for (int i = 0; i < mat->GetTextureCount(type); i++) {
		aiString a_str;
		mat->GetTexture(type, i, &a_str);
		bool skip = false;
		for (auto j : textures_loaded) {
			if (j.path == a_str) {
				textures.emplace_back(j);
				skip = true;
				break;
			}
		}

		if (!skip) {
			mesh_texture texture;
			texture.id = load_texture(a_str.C_Str(), directory);
			texture.type = type_name;
			texture.path = a_str;
			textures.emplace_back(texture);
			textures_loaded.emplace_back(texture);
		}
	}
	return textures;
}

fan_3d::model::model(const std::string& path, const vec3& position, const vec3& size) 
	: model_loader(path, size / 2.f), m_shader(fan_3d::shader_paths::model_vs, fan_3d::shader_paths::model_fs), 
	m_position(position), m_size(size)
{
	for (int i = 0; i < this->meshes.size(); i++) {
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

	mat4 model(1);
	model = translate(model, get_position());
	model = scale(model, get_size());

	this->m_shader.use();

	this->m_shader.set_int("texture_sampler", 0);
	this->m_shader.set_mat4("projection", fan_3d::frame_projection);
	this->m_shader.set_mat4("view", fan_3d::frame_view);
	this->m_shader.set_vec3("light_position", fan_3d::camera.get_position());
	this->m_shader.set_vec3("view_position", fan_3d::camera.get_position());
	this->m_shader.set_vec3("light_color", vec3(1, 1, 1));
	this->m_shader.set_int("texture_diffuse", 0);
	this->m_shader.set_mat4("model", model);

	//_Shader.set_vec3("sky_color", vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->textures_loaded[0].id);

	glDepthFunc(GL_LEQUAL);
	for (int i = 0; i < this->meshes.size(); i++) {
		glBindVertexArray(this->meshes[i].vao);
		glDrawElementsInstanced(GL_TRIANGLES, this->meshes[i].indices.size(), GL_UNSIGNED_INT, 0, 1);
	}
	glDepthFunc(GL_LESS);

	glBindVertexArray(0);
}

vec3 fan_3d::model::get_position()
{
	return this->m_position;
}

void fan_3d::model::set_position(const vec3& position)
{
	this->m_position = position;
}

vec3 fan_3d::model::get_size()
{
	return this->m_size;
}

void fan_3d::model::set_size(const vec3& size)
{
	this->m_size = size;
}

void GetFps(bool title, bool print) {
	static int fps = 0;
	static Timer timer(Timer::start(), 1000);
	static _Timer<microseconds> frame_time(Timer::start());
	static int old_fps = 0;
	float current_frame = glfwGetTime();
	static float last_frame = 0;
	delta_time = current_frame - last_frame;
	last_frame = current_frame;

	fan_2d::frame_view = mat4(1);
	fan_2d::frame_view = fan_2d::camera.get_view_matrix(translate(fan_2d::frame_view, vec3(window_size.x / 2, window_size.y / 2, -700.0f)));

	fan_2d::frame_projection = mat4(1);
	fan_2d::frame_projection = ortho(window_size.x / 2, window_size.x + window_size.x * 0.5f, window_size.y + window_size.y * 0.5f, window_size.y / 2.f, 0.1f, 1000.0f);

	fan_3d::frame_view = mat4(1);
	fan_3d::frame_view = fan_3d::camera.get_view_matrix();

	fan_3d::frame_projection = mat4(1);
	fan_3d::frame_projection = perspective(radians(90.f), (f_t)window_size.x / (f_t)window_size.y, 0.1f, 1000000.0f);

	if (timer.finished()) {
		old_fps = fps - 1;
		fps = 0;
		if (title) {
			glfwSetWindowTitle(window, (
				std::string("FPS: ") +
				std::to_string(old_fps) +
				std::string(" frame time: ") +
				std::to_string(static_cast<f_t>(frame_time.elapsed()) / static_cast<f_t>(1000)) +
				std::string(" ms")
				).c_str());
		}
		if (print) {
			std::cout << (std::string("FPS: ") +
				std::to_string(old_fps) +
				std::string(" frame time: ") +
				std::to_string(static_cast<f_t>(frame_time.elapsed()) / static_cast<f_t>(1000)) +
				std::string(" ms")
				) << '\n';
		}
		timer.restart();
	}
	frame_time.restart();
	fps++;
}

// -------------------------------GUI--------------------------------

suckless_font_t suckless_font_fr(uint_t datasize, uint_t fontsize) {
	suckless_font_t font;
	font.offset = { 0, 0 };
	font.datasize = datasize;
	font.fontsize = fontsize;
	font.data.resize(font.datasize * font.datasize);
	return font;
}

letter_info_t suckless_font_add_f(FT_Library ft, suckless_font_t* font, uint8_t letter) {
	FT_Face face;
	if (FT_New_Face(ft, "fonts/arial.ttf", 0, &face))
		printf("err new face\n");

	FT_Set_Pixel_Sizes(face, 0, font->fontsize);
	FT_Load_Char(face, letter, FT_LOAD_RENDER);

	uint_t tx = face->glyph->bitmap.width;
	uint_t ty = face->glyph->bitmap.rows;

	letter_info_t letter_info;

	letter_info.width = tx;
	letter_info.height = ty;

	uint_t bx = face->glyph->bitmap_left;
	/* extreme hardcoded */
	uint_t by = (font->fontsize / 1.33) - face->glyph->bitmap_top;

	/* check if offsets are in limit */
	if ((font->offset.x + tx) > font->datasize) {
		font->offset.x = 0;
		font->offset.y += font->fontsize;
	}
	if ((font->offset.y + font->fontsize) > font->datasize) {
		fprintf(stderr, "vector too small\n");
		return { 0 };
	}

	letter_info.pos = font->offset;

	/* transfer face buffer to data */
	for (uint_t iy = by; iy < (by + ty); iy++) {
		for (uint_t ix = 0; ix < tx; ix++) {
			font->data[(font->datasize * (iy + font->offset.y)) + (ix + font->offset.x)] = face->glyph->bitmap.buffer[(tx * (iy - by)) + ix];
		}
	}

	/* calculate offset for next character */
	if ((font->offset.x + tx) > font->datasize) {
		font->offset.x = 0;
		font->offset.y += font->fontsize;
	}
	else {
		font->offset.x += tx;
	}

	FT_Done_Face(face);
	return letter_info;
}

void suckless_letter_render(suckless_font_t* font) {
	for (uint_t iy = 0; iy < font->datasize; iy++) {
		for (uint_t ix = 0; ix < font->datasize; ix++) {
			if (font->data[(font->datasize * iy) + ix] > 128) /* if more than half solid */
				printf("W");
			else
				printf(" ");
		}
		printf("\n");
	}
}

constexpr auto characters_begin(33);
constexpr auto characters_end(248);

std::array<f_t, 248> fan_gui::text_renderer::widths;
suckless_font_t fan_gui::text_renderer::font;

fan_gui::text_renderer::text_renderer()
	: m_shader(Shader(fan_2d::shader_paths::text_renderer_vs, fan_2d::shader_paths::text_renderer_fs))
{
	FT_Library ft;

	if (FT_Init_FreeType(&ft))
		std::cout << "Could not init FreeType Library" << std::endl;

	FT_Face face;
	if (FT_New_Face(ft, "fonts/arial.ttf", 0, &face))
		std::cout << "Failed to load font" << std::endl;

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	FT_Set_Pixel_Sizes(face, 0, font_size);

	font = suckless_font_fr(1024 * 2, font_size);

	std::array<letter_info_t, characters_end> infos;

	for (int i = characters_begin; i < characters_end; i++) {
		infos[i] = suckless_font_add_f(ft, &font, i);
	}

	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_RED,
		font.datasize,
		font.datasize,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		font.data.data()
	);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glBindTexture(GL_TEXTURE_2D, 0);

	FT_Done_Face(face);
	FT_Done_FreeType(ft);

	std::vector<vec2> texture_coordinates;

	letter_info_t _letter;

	std::wstring text;
	for (int i = characters_begin; i < characters_end; i++) {
		text.push_back(i);
	}

	for (int i = 0; i < text.size(); i++) {
		_letter = infos[text[i]];
		widths[i + 33] = infos[text[i]].width / font_size;
		letter_info_opengl_t letter = letter_to_opengl(font, _letter);
		f_t height = letter.pos.y + ((f_t)font.fontsize / font.datasize);

		texture_coordinates.push_back(vec2(letter.pos.x, letter.pos.y));
		texture_coordinates.push_back(vec2(letter.pos.x, height));
		texture_coordinates.push_back(vec2(letter.pos.x + letter.width, height));
		texture_coordinates.push_back(vec2(letter.pos.x, letter.pos.y));
		texture_coordinates.push_back(vec2(letter.pos.x + letter.width, height));
		texture_coordinates.push_back(vec2(letter.pos.x + letter.width, letter.pos.y));
	}

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_texture_id_ssbo);
	glGenBuffers(1, &m_text_ssbo);
	glGenBuffers(1, &m_colors_ssbo);
	glGenBuffers(1, &m_vertex_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_texture_id_ssbo);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		sizeof(texture_coordinates[0]) * texture_coordinates.size(),
		texture_coordinates.data(),
		GL_STATIC_DRAW
	);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glBindVertexArray(m_vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_text_ssbo);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 1, GL_FLOAT_T, GL_FALSE, sizeof(int), 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_colors_ssbo);
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT_T, GL_FALSE, vec4::size() * sizeof(vec4::type), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttribDivisor(4, 1);
	glVertexAttribDivisor(6, 1);
	glBindVertexArray(0);
}

fan_gui::text_renderer::~text_renderer()
{
	glDeleteVertexArrays(1, &m_vao);
	glDeleteBuffers(1, &m_texture_id_ssbo);
	glDeleteBuffers(1, &m_text_ssbo);
	glDeleteBuffers(1, &m_colors_ssbo);
	glDeleteBuffers(1, &m_vertex_ssbo);
	glDeleteTextures(1, &m_texture);
}

void fan_gui::text_renderer::alloc_storage(const std::vector<std::wstring>& vector)
{
	realloc_storage(vector);
}

void fan_gui::text_renderer::realloc_storage(const std::vector<std::wstring>& vector)
{
	m_colors.resize(vector.size());
	for (int i = 0; i < vector.size(); i++) {
		m_colors[i].resize(vector[i].size());
	}
}

void fan_gui::text_renderer::store_to_renderer(std::wstring& text, vec2 position, const Color& color, f_t scale, f_t max_width)
{
	m_characters.resize(m_characters.size() + 1);
	m_vertices.resize(m_vertices.size() + 1);
	m_characters[m_characters.size() - 1].resize(text.size());

	f_t width = 0;
	f_t begin = position.x;

	for (int i = 0; i < text.size(); i++) {
		emplace_vertex_data(m_vertices[m_vertices.size() - 1], position, vec2(widths[text[i]] * scale, scale));

		if (max_width != -1) {
			f_t next_step = 0;

			switch (text[i]) {
			case ' ': {
				next_step += fan_gui::font::properties::get_space(scale) * 2;
				break;
			}
			case '\n': {
				next_step = (position.x) * 2;
				break;
			}
			default: {
				next_step += (widths[text[i]] * scale + fan_gui::font::properties::space_between_characters) * 2;
			}
			}

			if (width + next_step >= max_width) {
				position.x = begin;
				m_characters[m_characters.size() - 1].resize(m_characters[m_characters.size() - 1].size() + 1);
				m_colors[m_colors.size() - 1].resize(m_colors[m_colors.size() - 1].size() + 1);
				position.y += scale;
				width = 0;
				//i--;
				//continue;
				goto skip;
			}
		}


		switch (text[i]) {
		case ' ': {
			if (width != 0) {
				position.x += fan_gui::font::properties::get_space(scale);
				width += fan_gui::font::properties::get_space(scale);
			}
			break;
		}
		case '\n': {
			position.x = begin;
			position.y += scale;
			break;
		}
		default: {
			position.x += widths[text[i]] * scale + fan_gui::font::properties::space_between_characters;
			width += widths[text[i]] * scale + fan_gui::font::properties::space_between_characters;
		}
		}
	skip:
		m_colors[m_colors.size() - 1][i] = color;
		m_characters[m_characters.size() - 1][i] = text[i];
	}
}

void fan_gui::text_renderer::edit_storage(uint64_t i, const std::wstring& text, vec2 position, const Color& color, f_t scale)
{
	m_vertices[i].clear();
	m_characters[i].resize(text.size());
	m_colors[i].resize(text.size());

	f_t width = 0;
	f_t begin = position.x;

	for (int character = 0; character < text.size(); character++) {
		emplace_vertex_data(m_vertices[i], position, vec2(widths[text[character]] * scale, scale));

		switch (text[character]) {
		case ' ': {
			if (width != 0) {
				position.x += fan_gui::font::properties::get_space(scale);
				width += fan_gui::font::properties::get_space(scale);
			}
			break;
		}
		case '\n': {
			position.x = begin;
			position.y += scale;
			break;
		}
		default: {
			position.x += widths[text[character]] * scale + fan_gui::font::properties::space_between_characters;
			width += widths[text[character]] * scale + fan_gui::font::properties::space_between_characters;
		}
		}
		m_colors[i][character] = color;
		m_characters[i][character] = text[character];
	}
}

void fan_gui::text_renderer::upload_vertices()
{
	std::vector<vec2> one_dimension_draw_data(vector_size(m_vertices));
	int copied = 0;
	for (int i = 0; i < m_vertices.size(); i++) {
		std::copy(m_vertices[i].begin(), m_vertices[i].end(), one_dimension_draw_data.begin() + copied);
		copied += m_vertices[i].size();
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_vertex_ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(one_dimension_draw_data[0]) * one_dimension_draw_data.size(), one_dimension_draw_data.data(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_vertex_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void fan_gui::text_renderer::upload_colors()
{
	std::vector<Color> one_dimension_colors(vector_size(m_colors));
	int copied = 0;
	for (int i = 0; i < m_colors.size(); i++) {
		std::copy(m_colors[i].begin(), m_colors[i].end(), one_dimension_colors.begin() + copied);
		copied += m_colors[i].size();
	}
	glBindBuffer(GL_ARRAY_BUFFER, m_colors_ssbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(one_dimension_colors[0]) * one_dimension_colors.size(), one_dimension_colors.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void fan_gui::text_renderer::upload_characters()
{
	std::vector<int> one_dimension_characters(vector_size(m_characters));
	int copied = 0;
	for (int i = 0; i < m_characters.size(); i++) {
		std::copy(m_characters[i].begin(), m_characters[i].end(), one_dimension_characters.begin() + copied);
		copied += m_characters[i].size();
	}
	glBindBuffer(GL_ARRAY_BUFFER, m_text_ssbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(one_dimension_characters[0]) * one_dimension_characters.size(), one_dimension_characters.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void fan_gui::text_renderer::upload_stored()
{
	this->upload_vertices();
	this->upload_colors();
	this->upload_characters();
}

void fan_gui::text_renderer::upload_stored(uint64_t i)
{
	std::vector<int> new_characters(m_characters[i].begin(), m_characters[i].end());

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_text_ssbo);
	glBufferSubData(
		GL_SHADER_STORAGE_BUFFER,
		i * new_characters.size() * sizeof(int),
		sizeof(int) * new_characters.size(),
		new_characters.data()
	);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_text_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

}

void fan_gui::text_renderer::render_stored()
{
	m_shader.use();
	m_shader.set_mat4("projection", ortho(0, window_size.x, window_size.y, 0));

	glBindVertexArray(m_vao);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 6, vector_size(m_characters));
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindVertexArray(0);
}

void fan_gui::text_renderer::set_scale(uint64_t i, f_t scale, vec2 position)
{
	m_vertices[i].clear();

	f_t begin = position.x;

	for (int index = 0; index < m_characters[i].size(); index++) {

		emplace_vertex_data(m_vertices[i], position, vec2(widths[m_characters[i][index]] * scale, scale));

		switch (m_characters[i][index]) {
		case ' ': {
			position.x += fan_gui::font::properties::get_space(scale);
			break;
		}
		case '\n': {
			position.x = begin;
			position.y += scale;
			break;
		}
		default: {
			position.x += fan_gui::font::properties::get_character_x_offset(widths[m_characters[i][index]], scale);
		}
		}
	}

	upload_vertices();
}

void fan_gui::text_renderer::clear_storage()
{
	m_characters.clear();
}

void fan_gui::text_renderer::render(const std::wstring& text, vec2 position, const Color& color, f_t scale, bool use_old) {
	static std::wstring old_str;

	m_shader.use();
	m_shader.set_mat4("projection", ortho(0, window_size.x, window_size.y, 0));

	f_t begin = position.x;

	if (use_old && old_str == text) {
		goto draw;
	}
	{

		m_vertices.clear();
		m_characters.clear();
		m_colors.clear();

		m_characters.resize(1);
		m_characters[0].resize(text.size());

		m_colors.resize(1);
		m_colors[0].resize(text.size());

		m_vertices.resize(1);

		for (int i = 0; i < text.size(); i++) {
			if (m_vertices.size() < 6 * text.size()) {
				emplace_vertex_data(m_vertices[m_vertices.size() - 1], position, vec2(widths[text[i]] * scale, scale));
			}
			else {
				edit_vertex_data(i * 6, m_vertices[i], position, vec2(widths[text[i]] * scale, scale));
			}

			switch (text[i]) {
			case ' ': {
				position.x += fan_gui::font::properties::get_space(scale);
				break;
			}
			case '\n': {
				position.x = begin;
				position.y += scale;
				break;
			}
			default: {
				position.x += widths[text[i]] * scale + fan_gui::font::properties::space_between_characters;
			}
			}
			m_colors[0][i] = color;
			m_characters[0][i] = text[i];
		}
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_vertex_ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(m_vertices[0][0]) * m_vertices[0].size(), m_vertices[0].data(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_vertex_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	
	glBindBuffer(GL_ARRAY_BUFFER, m_text_ssbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_characters[0][0]) * m_characters[0].size(), m_characters[0].data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_colors_ssbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_colors[0][0]) * m_colors[0].size(), m_colors[0].data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	old_str = text;



draw:
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_vertex_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);

	glBindVertexArray(m_vao);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 6, text.size());
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindVertexArray(0);
}

vec2 fan_gui::text_renderer::get_length(const std::wstring& text, f_t scale)
{
	vec2 string_size;

	string_size.y = scale;

	f_t biggest_width = -1;

	for (int i = 0; i < text.size(); i++) {
		if (text[i] == ' ') {
			string_size.x += fan_gui::font::properties::get_space(scale);
		}
		else if (text[i] == '\n') {
			string_size.y += scale;
			biggest_width = std::max(string_size.x, biggest_width);
			string_size.x = 0;
		}
		else {
			if (i != text.size() - 1) {
				string_size.x += widths[text[i]] * scale + fan_gui::font::properties::space_between_characters;
			}
			else {
				string_size.x += widths[text[i]] * scale;
			}
		}
	}

	biggest_width = std::max(string_size.x, biggest_width);

	if (biggest_width == -1) {
		biggest_width = string_size.x;
	}

	return vec2(biggest_width, string_size.y);
}

std::vector<vec2> fan_gui::text_renderer::get_length(const std::vector<std::wstring>& texts, const std::vector<f_t>& scales, bool half)
{
	f_t width;
	std::vector<vec2> string_size(texts.size());

	for (int text = 0; text < texts.size(); text++) {
		for (int i = 0; i < texts[text].size(); i++) {
			width = widths[texts[text][i]];

			if (texts[text][i] == ' ') {
				if (half) {
					string_size[text].x += 2.5;
				}
				else {
					string_size[text].x += 5;
				}
			}
			else {
				vec2 size(width, font.fontsize);
				size = size / size.max() * scales[text];

				if (half) {
					string_size[text].x += (size.x + 2) * 0.5;
				}
				else {
					string_size[text].x += size.x + 2;
				}
			}

		}
		if (half) {
			string_size[text].y = -(int)font.fontsize / 2;
		}
		else {
			string_size[text].y = font.fontsize / 2;
		}
	}

	return string_size;
}

fan_gui::font::basic_methods::basic_text_button_vector::basic_text_button_vector() : fan_gui::text_renderer() {}

vec2 fan_gui::font::basic_methods::basic_text_button_vector::edit_size(uint64_t i, const std::wstring& text, f_t scale)
{
	std::vector<std::wstring> lines;
	int offset = 0;
	for (int i = 0; i < text.size(); i++) {
		if (text[i] == '\n') {
			lines.push_back(text.substr(offset, i - offset));
			offset = i + 1;
		}
	}
	lines.push_back(text.substr(offset, text.size()));
	f_t largest = -9999999;
	for (auto i : lines) {
		f_t size = get_length(i, scale).x;
		if (size > largest) {
			largest = size;
		}
	}
	m_texts[i] = text;
	return vec2(fan_gui::font::properties::get_gap_scale_x(largest), fan_gui::font::properties::get_gap_scale_y(scale) * lines.size());
}

fan_gui::font::text_button_vector::text_button_vector() : basic_text_button_vector() { }

fan_gui::font::text_button_vector::text_button_vector(const std::wstring& text, const vec2& position, const Color& color, float_t scale)
	: basic_text_button_vector() {
	this->add(text, position, color, scale);
}

fan_gui::font::text_button_vector::text_button_vector(const std::wstring& text, const vec2& position, const Color& color, float_t scale, const vec2& box_size)
	: basic_text_button_vector() {
	m_scales.push_back(scale);
	std::vector<std::wstring> all_strings(m_texts.begin(), m_texts.end());
	all_strings.push_back(text);
	m_texts.push_back(text);
	realloc_storage(all_strings);
	push_back(position, box_size, color); // * 2 for both sides
	auto rtext = text;
	store_to_renderer(rtext, position + box_size / 2 - get_length(text, scale) / 2, default_text_color, scale);
	upload_stored();
}

void fan_gui::font::text_button_vector::add(const std::wstring& text, const vec2& position, const Color& color, float_t scale)
{
	m_scales.push_back(scale);
	std::vector<std::wstring> all_strings(m_texts.begin(), m_texts.end());
	all_strings.push_back(text);
	m_texts.push_back(text);
	realloc_storage(all_strings);
	push_back(position, get_length(text, scale) + fan_gui::font::properties::get_gap_scale(scale) * 2, color); // * 2 for both sides
	auto rtext = text;
	store_to_renderer(rtext, position + fan_gui::font::properties::get_gap_scale(scale), default_text_color, scale);
	upload_stored();
}

void fan_gui::font::text_button_vector::add(const std::wstring& text, const vec2& position, const Color& color, float_t scale, const vec2& box_size)
{
	m_scales.push_back(scale);
	std::vector<std::wstring> all_strings(m_texts.begin(), m_texts.end());
	all_strings.push_back(text);
	m_texts.push_back(text);
	realloc_storage(all_strings);
	push_back(position, box_size, color); // * 2 for both sides
	auto rtext = text;
	store_to_renderer(rtext, position + box_size / 2 - get_length(text, scale) / 2, default_text_color, scale);
	upload_stored();
}

void fan_gui::font::text_button_vector::edit_string(uint64_t i, const std::wstring& text, f_t scale)
{
	m_scales[i] = scale;
	m_texts[i] = text;
	auto len = get_length(text, scale);
	set_size(i, len + fan_gui::font::properties::get_gap_scale(scale) * 2);
	edit_storage(i, text, get_position(i) + fan_gui::font::properties::get_gap_scale(scale), default_text_color, scale);
	upload_stored();
}

vec2 fan_gui::font::text_button_vector::get_string_length(const std::wstring& text, f_t scale)
{
	return get_length(text, scale);
}

f_t fan_gui::font::text_button_vector::get_scale(uint64_t i)
{
	return m_scales[i];
}

void fan_gui::font::text_button_vector::set_font_size(uint64_t i, f_t scale)
{
	m_scales[i] = scale;
	auto str = std::wstring(m_characters[i].begin(), m_characters[i].end());
	auto len = get_length(str, scale);
	auto text_size = edit_size(i, str, scale);
	set_size(i, len + fan_gui::font::properties::get_gap_scale(scale) * 2);
	auto pos = get_position(i);
	set_scale(i, scale, pos + fan_gui::font::properties::get_gap_scale(scale));
	upload_stored();
}

void fan_gui::font::text_button_vector::set_position(uint64_t i, const vec2& position)
{
	f_t scale = get_scale(i);
	fan_2d::square_vector::set_position(i, position);
	auto len = get_length(m_texts[i], scale);
	set_size(i, len + fan_gui::font::properties::get_gap_scale(scale) * 2);
	edit_storage(i, m_texts[i], get_position(i) + fan_gui::font::properties::get_gap_scale(scale), default_text_color, scale);
	upload_stored();
}

void fan_gui::font::text_button_vector::set_press_callback(int key, const std::function<void()>& function)
{
	callback::key.add(key, true, function);
}

void fan_gui::font::text_button_vector::draw()
{
	fan_2d::square_vector::draw();
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_vertex_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);
	glDisable(GL_DEPTH_TEST);
	render_stored();
	glEnable(GL_DEPTH_TEST);
}

bool fan_gui::font::text_button_vector::inside(std::uint64_t i)
{
	vec2 position = get_position(i);
	vec2 size = get_size(i);
	if (cursor_position.x >= position.x && cursor_position.x <= position.x + size.x &&
		cursor_position.y >= position.y && cursor_position.y <= position.y + size.y)
	{
		return true;
	}
	return false;
}

void begin_render(const Color& background_color)
{
	GetFps(true, true);

	glClearColor(
		background_color.r,
		background_color.g,
		background_color.b,
		background_color.a
	);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void end_render()
{
	glfwSwapBuffers(window);
	glfwPollEvents();
}