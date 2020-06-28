#include <FAN/Graphics.hpp>
#include <functional>
#include <numeric>

float delta_time = 0;
GLFWwindow* window;
bool window_init = WindowInit();

void GetFps(bool print) {
	static int fps = 0;
	static Timer timer(Timer::start(), 1000);
	static _Timer<microseconds> frame_time(Timer::start());
	static int old_fps = 0;
	float current_frame = glfwGetTime();
	static float last_frame = 0;
	delta_time = current_frame - last_frame;
	last_frame = current_frame;
	if (timer.finished()) {
		old_fps = fps;
		fps = 0;
		if (print) {
			glfwSetWindowTitle(window, (
				std::string("FPS: ") + 
				std::to_string(old_fps) + 
				std::string(" frame time: ") + 
				std::to_string(static_cast<float_t>(frame_time.elapsed()) / static_cast<float_t>(1000)) + 
				std::string(" ms")
			).c_str());
		}
		timer.restart();
	}
	frame_time.restart();
	fps++;
}

Texture::Texture() : texture(0), width(0), height(0), VBO(0), VAO(0) { }

_vec2<int> window_position() {
	_vec2<int> position;
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
	this->updateCameraVectors();
}

void Camera::move(bool noclip, float_t movement_speed)
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
		const vec2 direction(DirectionVector(Radians(this->yaw)));
		this->velocity.x += direction.x * (movement_speed * delta_time);
		this->velocity.z += direction.y * (movement_speed * delta_time);
	}
	if (glfwGetKey(window, GLFW_KEY_S)) {
		const vec2 direction(DirectionVector(Radians(this->yaw)));
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
	this->updateCameraVectors();
}

void Camera::rotate_camera()
{
	static bool firstMouse = true;
	static float lastX, lastY;
	float xpos = cursor_position.x;
	float ypos = cursor_position.y;

	float& yaw = camera3d.yaw;
	float& pitch = camera3d.pitch;

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

	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;
}

mat4 Camera::get_view_matrix() {
	return LookAt(this->position, (this->position + (this->front)), (this->up));
}

mat4 Camera::get_view_matrix(mat4 m) {
	return m * LookAt(this->position, (this->position + (this->front)).rounded(), (this->up).rounded());
}

vec3 Camera::get_position() const {
	return this->position;
}

void Camera::set_position(const vec3& position) {
	this->position = position;
}

void Camera::updateCameraVectors() {
	vec3 front;
	front.x = cos(Radians(this->yaw)) * cos(Radians(this->pitch));
	front.y = sin(Radians(this->pitch));
	front.z = sin(Radians(this->yaw)) * cos(Radians(this->pitch));
	this->front = Normalize(front);
	// Also re-calculate the Right and Up vector
	this->right = Normalize(Cross(this->front, this->worldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
	this->up = Normalize(Cross(this->right, this->front));
}

Camera camera2d(vec3(), vec3(0, 1, 0), -90, 0);
Camera camera3d;

Shader shape_shader2d("GLSL/shapes.vs", "GLSL/shapes.frag");
Shader sprite_shader2d("GLSL/sprite.vs", "GLSL/sprite.fs");

default_2d_base_vector::default_2d_base_vector() :
	matrix_allocator_size(0), matrix_allocated(false), shapes_size(0) {}

template<typename _Type, uint64_t N>
default_2d_base_vector::default_2d_base_vector(const std::array<_Type, N>& init_vertices) : 
	matrix_allocator_size(0), matrix_allocated(false), shapes_size(0)
{
	glGenVertexArrays(1, &shape_vao);
	std::array<unsigned int*, 2> vbos{
		&matrix_vbo, &vertex_vbo
	};
	glGenBuffers(vbos.size(), *vbos.data());
	glBindVertexArray(shape_vao);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Type) * init_vertices.size(), init_vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	for (int i = 3; i < 7; i++) {
		glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), reinterpret_cast<void*>((i - 3) * sizeof(vec4)));
		glEnableVertexAttribArray(i);
		glVertexAttribDivisor(i, 1);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

default_2d_base_vector::~default_2d_base_vector()
{
	glDeleteVertexArrays(1, &shape_vao);
	glDeleteBuffers(1, &matrix_vbo);
	glDeleteBuffers(1, &vertex_vbo);
	if (this->matrix_allocated) {
		glDeleteBuffers(1, &matrix_allocator_vbo);
	}

}

void default_2d_base_vector::free_queue()
{
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	this->copy_data(matrix_allocator_vbo, GL_ARRAY_BUFFER, this->matrix_allocator_size, GL_DYNAMIC_DRAW, matrix_vbo);
	glDeleteBuffers(1, &matrix_allocator_vbo);
	this->matrix_allocated = false;
	this->matrix_allocator_size = 0;
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

std::vector<vec2> default_2d_base_vector::get_positions() const
{
	int size = this->size();
	std::vector<mat4> matrix(size);
	std::vector<vec2> positions(size);
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, size * sizeof(mat4), matrix.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	for (int i = 0; i < size; i++) {
		positions[i].x = matrix[i][3][0];
		positions[i].y = matrix[i][3][1];
	}
	return positions;
}

vec2 default_2d_base_vector::get_position(uint64_t index) const
{
	mat4 matrix;
	vec2 position;
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, index * sizeof(mat4), sizeof(mat4), matrix.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	position.x = matrix[3][0];
	position.y = matrix[3][1];

	return position;
}

void default_2d_base_vector::set_position(uint64_t index, const vec2& position, bool queue)
{
	mat4 matrix(1);
	matrix = Translate(matrix, position);
	matrix = Scale(matrix, get_size(index));
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * index, sizeof(mat4), matrix.data());
}

std::vector<vec2> default_2d_base_vector::get_sizes(bool half) const
{
	int size = this->size();
	std::vector<mat4> matrix(size);
	std::vector<vec2> sizes(size);
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, size * sizeof(mat4), matrix.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	for (int i = 0; i < size; i++) {
		if (half) {
			sizes[i].x = matrix[i][0][0] * 0.5;
			sizes[i].y = matrix[i][1][1] * 0.5;
		}
		else {
			sizes[i].x = matrix[i][0][0];
			sizes[i].y = matrix[i][1][1];
		}
	}

	return sizes;
}

vec2 default_2d_base_vector::get_size(uint64_t index) const
{
	mat4 matrix;
	vec2 size;
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, index * sizeof(mat4), sizeof(mat4), matrix.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	size.x = matrix[0][0];
	size.y = matrix[1][1];
	return size;
}

void default_2d_base_vector::set_size(uint64_t index, const vec2& size, bool queue)
{
	mat4 matrix(1);
	matrix = Translate(matrix, get_position(index));
	matrix = Scale(matrix, size);
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * index, sizeof(mat4), matrix.data());
}

int default_2d_base_vector::get_buffer_size(int buffer) const
{
	int size = 0;

	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return size;
}

void default_2d_base_vector::realloc_copy_data(unsigned int& buffer, uint64_t buffer_type, int size, GLenum usage, unsigned int& allocator) const
{
	int old_buffer_size = 0;
	glBindBuffer(buffer_type, buffer);
	glGetBufferParameteriv(buffer_type, GL_BUFFER_SIZE, (int*)&old_buffer_size);

	glGenBuffers(1, &allocator);
	glBindBuffer(GL_COPY_WRITE_BUFFER, allocator);
	glBufferData(GL_COPY_WRITE_BUFFER, size, nullptr, usage);

	glBindBuffer(GL_COPY_READ_BUFFER, buffer);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, old_buffer_size);
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void default_2d_base_vector::copy_data(unsigned int& buffer, uint64_t buffer_type, int size, GLenum usage, unsigned int& allocator) const
{
	glBindBuffer(GL_COPY_WRITE_BUFFER, allocator);
	glBindBuffer(GL_COPY_READ_BUFFER, buffer);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, size);
	if (buffer_type == GL_SHADER_STORAGE_BUFFER) {
		glBindBufferBase(buffer_type, 1, buffer);
	}
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void default_2d_base_vector::realloc_buffer(unsigned int& buffer, uint64_t buffer_type, int location, int size, GLenum usage, unsigned int& allocator) const
{
	int old_buffer_size = 0;

	glBindBuffer(buffer_type, buffer);
	glGetBufferParameteriv(buffer_type, GL_BUFFER_SIZE, (int*)&old_buffer_size);

	glGenBuffers(1, &allocator);

	glBindBuffer(GL_COPY_WRITE_BUFFER, allocator);
	glBufferData(GL_COPY_WRITE_BUFFER, old_buffer_size, nullptr, usage);

	glBindBuffer(GL_COPY_READ_BUFFER, buffer);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, old_buffer_size);
	const int new_size = size + old_buffer_size;

	glBufferData(GL_COPY_READ_BUFFER, new_size, nullptr, usage);

	if (buffer_type == GL_SHADER_STORAGE_BUFFER) {
		glBindBufferBase(GL_COPY_READ_BUFFER, location, buffer);
	}

	glCopyBufferSubData(GL_COPY_WRITE_BUFFER, GL_COPY_READ_BUFFER, 0, 0, old_buffer_size);

	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	glBindBuffer(buffer_type, 0);
}

int default_2d_base_vector::size() const
{
	return shapes_size;
}

void default_2d_base_vector::erase(uint64_t first, uint64_t last)
{
	std::vector<mat4> matrices(this->size());
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, this->size() * sizeof(mat4), matrices.data());
	if (last != -1) {
		matrices.erase(matrices.begin() + first, matrices.begin() + last);
	}
	else {
		matrices.erase(matrices.begin() + first);
	}
	glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * matrices.size(), matrices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shapes_size -= (last == -1 ? 1 : (last - first));
}

void default_2d_base_vector::basic_shape_draw(unsigned int mode, uint64_t points, const Shader& shader) const
{
	int amount_of_objects = size();
	if (!amount_of_objects) {
		return;
	}

	mat4 view(1);
	mat4 projection(1);

	view = camera2d.get_view_matrix(Translate(view, vec3(window_size.x / 2, window_size.y / 2, -700.0f)));

	projection = Ortho(window_size.x / 2, window_size.x + window_size.x * 0.5f, window_size.y + window_size.y * 0.5f, window_size.y / 2.f, 0.1f, 1000.0f);


	shader.use();
	shader.set_mat4("projection", projection);
	shader.set_mat4("view", view);

	static float pos;
	shader.set_float("position_multiplier", pos);
	pos += delta_time * 5;

	glBindVertexArray(shape_vao);
	glDrawArraysInstanced(mode, 0, points, amount_of_objects);
	glBindVertexArray(0);

	//for (int i = 0; i < 2; i++) { // for bloom
	//	shader.use();
	//	shader.set_mat4("projection", projection);
	//	shader.set_mat4("view", view);
	//	shader.set_int("vert", 0);
	//	static float pos;
	//	shader.set_float("position_multiplier", pos);
	//	pos += delta_time;

	//	glBindVertexArray(shape_vao);
	//	glDrawArraysInstanced(mode, 0, points, amount_of_objects);
	//	glBindVertexArray(0);
	//}
}

basic_2dshape_vector::basic_2dshape_vector() : 
	color_allocated(false), color_allocator_size(0), default_2d_base_vector() {}

template <typename _Type, uint64_t N>
basic_2dshape_vector::basic_2dshape_vector(const std::array<_Type, N>& init_vertices) :
	color_allocated(false), color_allocator_size(0), default_2d_base_vector(init_vertices)
{
	glGenBuffers(1, &color_vbo);

	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	glVertexAttribDivisor(1, 1);

	glBindVertexArray(0);
}

basic_2dshape_vector::~basic_2dshape_vector()
{
	glDeleteBuffers(1, &color_vbo);
	if (this->color_allocated) {
		glDeleteBuffers(1, &color_allocator_vbo);
	}
}

std::vector<Color> basic_2dshape_vector::get_colors() const
{
	std::vector<Color> colors(this->size());
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, size() * sizeof(Color), colors.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return colors;
}

Color basic_2dshape_vector::get_color(uint64_t index) const
{
	Color color;
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, index * sizeof(Color), sizeof(Color), color.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return color;
}

void basic_2dshape_vector::set_color(uint64_t index, const Color& color, bool queue)
{
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, index * sizeof(Color), sizeof(Color), &color.r);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void basic_2dshape_vector::free_queue(bool colors, bool matrices)
{
	if (colors) {
		glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Color) * this->size(), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		this->copy_data(color_allocator_vbo, GL_ARRAY_BUFFER, this->color_allocator_size, GL_DYNAMIC_DRAW, color_vbo);
		glDeleteBuffers(1, &color_allocator_vbo);
		this->color_allocated = false;
		this->color_allocator_size = 0;
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	if (matrices) {
		glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		this->copy_data(matrix_allocator_vbo, GL_ARRAY_BUFFER, this->matrix_allocator_size, GL_DYNAMIC_DRAW, matrix_vbo);
		glDeleteBuffers(1, &matrix_allocator_vbo);
		this->matrix_allocated = false;
		this->matrix_allocator_size = 0;
	}
}

void basic_2dshape_vector::erase(uint64_t first, uint64_t last)
{
	std::vector<Color> colors(this->size());
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, this->size() * sizeof(Color), colors.data());
	if (last != -1) {
		colors.erase(colors.begin() + first, colors.begin() + last + 1);
	}
	else {
		colors.erase(colors.begin() + first);
	}
	glBufferData(GL_ARRAY_BUFFER, sizeof(Color) * colors.size(), colors.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	default_2d_base_vector::erase(first, last);
}

void basic_2dshape_vector::push_back(const vec2& position, const vec2& size, const Color& color, bool queue)
{
	mat4 model(1);
	model = Translate(model, position);
	model = Scale(model, size);

	if (queue) {
	request_allocate:
		if (!this->color_allocated) {
			this->realloc_copy_data(color_vbo, GL_ARRAY_BUFFER, sizeof(Color) * copy_buffer + sizeof(Color) * this->size(), GL_DYNAMIC_DRAW, color_allocator_vbo);
			this->color_allocated = true;
		}

		if (!this->matrix_allocated) {
			this->realloc_copy_data(matrix_vbo, GL_ARRAY_BUFFER, sizeof(mat4) * copy_buffer + sizeof(mat4) * this->size(), GL_DYNAMIC_DRAW, matrix_allocator_vbo);
			this->matrix_allocated = true;
		}

		if (this->color_allocator_size + sizeof(Color) > sizeof(Color) * copy_buffer) {
			free_queue(true, false);
			this->color_allocated = false;
			goto request_allocate;
		}

		if (this->matrix_allocator_size + sizeof(mat4) > sizeof(mat4) *  copy_buffer) {
			free_queue(false, true);
			this->matrix_allocated = false;
			goto request_allocate;
		}

		glBindBuffer(GL_ARRAY_BUFFER, color_allocator_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(Color) * this->size(), sizeof(Color), &color);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		this->color_allocator_size += sizeof(Color);

		glBindBuffer(GL_ARRAY_BUFFER, matrix_allocator_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), sizeof(mat4), &model);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		this->matrix_allocator_size += sizeof(mat4);
	}
	else {
		realloc_buffer(color_vbo, GL_ARRAY_BUFFER, 0, sizeof(Color), GL_DYNAMIC_DRAW, color_allocator_vbo);

		glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(Color) * this->size(), sizeof(Color), &color);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDeleteBuffers(1, &color_allocator_vbo);

		realloc_buffer(matrix_vbo, GL_ARRAY_BUFFER, 0, sizeof(mat4), GL_DYNAMIC_DRAW, matrix_allocator_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), sizeof(mat4), model.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDeleteBuffers(1, &matrix_allocator_vbo);
	}

	shapes_size++;
}

void basic_2dshape_vector::insert(const vec2& position, const vec2& size, const Color& color, uint64_t how_many)
{
	mat4 model(1);
	model = Translate(model, position);
	model = Scale(model, size);

	std::vector<Color> colors(how_many, color);
	std::vector<mat4> matrices(how_many, model);

	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Color) * how_many, colors.data(), GL_DYNAMIC_DRAW);
	//glBufferSubData(GL_ARRAY_BUFFER, sizeof(Color) * how_many, sizeof(Color), &color);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &color_allocator_vbo);

	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * how_many, matrices.data(), GL_DYNAMIC_DRAW);
	//glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * how_many, sizeof(mat4), model.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &matrix_allocator_vbo);

	shapes_size += how_many;
}

constexpr std::array<float_t, 12> square_2d_vertices{
	0, 0,
	0, 1,
	1, 1,
	1, 1,
	1, 0,
	0, 0
};

constexpr std::array<float_t, 4> line_2d_vertices{
	0, 0,
	1, 1
};

line_vector2d::line_vector2d() : basic_2dshape_vector(line_2d_vertices) {}

line_vector2d::line_vector2d(const mat2& position, const Color& color) :
	basic_2dshape_vector(line_2d_vertices)
{
	line_vector2d::push_back(position, color);
}

void line_vector2d::push_back(const mat2& position, const Color& color, bool queue)
{
	if (position[0] < position[1]) {
		basic_2dshape_vector::push_back(position[0], position[1], color, queue);
	}
	else {
		basic_2dshape_vector::push_back(position[1], position[0] - position[1], color, queue); // TODO KORJAJAAJAJADSJAJ
	}
}

mat2 line_vector2d::get_position(uint64_t index) const
{
	return mat2(
		default_2d_base_vector::get_position(index),
		default_2d_base_vector::get_size(index)
	);
}

void line_vector2d::set_position(uint64_t index, const mat2& position, bool queue)
{
	if (position[0] < position[1]) {
		basic_2dshape_vector::set_position(index, position[0]);
		basic_2dshape_vector::set_size(index, position[1]);
	}
	else {
		basic_2dshape_vector::set_position(index, position[1]);
		basic_2dshape_vector::set_size(index, position[0] - position[1]);
	}
}

void line_vector2d::draw() const
{
	this->basic_shape_draw(GL_LINES, 2);
}

square_vector2d::square_vector2d() : basic_2dshape_vector(square_2d_vertices) {}

square_vector2d::square_vector2d(
	const vec2& position,
	const vec2& size,
	const Color& color
) : basic_2dshape_vector(square_2d_vertices)
{
	this->push_back(position, size, color);
}

void square_vector2d::draw() const
{
	this->basic_shape_draw(GL_TRIANGLES, 6);
}

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

auto get_texture_onsided(const vec2i& size, const vec2i& position) {
	vec2 b(1.f / size.x, 1.f / size.y);
	vec2 x(position.x * b.x, position.y * b.y);

	return std::array<float_t, 12>{
		x.x, 1.f - x.y,
		x.x, 1.f - (x.y + b.y),
		x.x + b.x, 1.f - (x.y + b.y),
		x.x + b.x, 1.f - (x.y + b.y),
		x.x + b.x, 1.f - x.y,
		x.x, 1.f - x.y
	};
}

sprite_vector2d::sprite_vector2d() : default_2d_base_vector(square_2d_vertices),
	texture_allocated(0), texture_allocator_size(0) {}

sprite_vector2d::sprite_vector2d(const char* path, const vec2& position, const vec2& size) : 
	default_2d_base_vector(square_2d_vertices), texture_allocated(0), texture_allocator_size(0)
{
	texture_id = load_texture(path, "", true, true);

	auto textures = get_texture_onsided(vec2i(1, 1), vec2i());
	glGenBuffers(1, &texture_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, texture_ssbo);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		sizeof(textures[0]) * textures.size(),
		textures.data(),
		GL_STATIC_DRAW
	);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, texture_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	glGenBuffers(1, &texture_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, texture_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, texture_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	this->push_back(position, size);
}

sprite_vector2d::~sprite_vector2d()
{
	glDeleteBuffers(1, &texture_ssbo);
	glDeleteTextures(1, &texture_id);
}

void sprite_vector2d::free_queue(bool textures, bool matrices)
{
	if (textures) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, texture_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * this->size(), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		this->copy_data(texture_allocator, GL_SHADER_STORAGE_BUFFER, this->texture_allocator_size, GL_DYNAMIC_DRAW, texture_ssbo);
		glDeleteBuffers(1, &texture_allocator);
		this->texture_allocated = false;
		this->texture_allocator_size = 0;
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
	if (matrices) {
		default_2d_base_vector::free_queue();
	}
}

void sprite_vector2d::draw() const
{
	sprite_shader2d.use();
	glBindTexture(GL_TEXTURE_2D, 1);
	sprite_shader2d.set_int("texture_sampler", 0);
	this->basic_shape_draw(GL_TRIANGLES, 6, sprite_shader2d);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void sprite_vector2d::push_back(const vec2& position, const vec2& size, bool queue)
{
	mat4 model(1);
	model = Translate(model, position);
	model = Scale(model, size);

	if (queue) {
	request_allocate:
		if (!this->texture_allocated) {
			this->realloc_copy_data(texture_ssbo, GL_SHADER_STORAGE_BUFFER, sizeof(int) * copy_buffer, GL_DYNAMIC_DRAW, texture_allocator);
			this->texture_allocated = true;
		}

		if (!this->matrix_allocated) {
			this->realloc_copy_data(matrix_vbo, GL_ARRAY_BUFFER, sizeof(mat4) * copy_buffer, GL_DYNAMIC_DRAW, matrix_allocator_vbo);
			this->matrix_allocated = true;
		}

		int texture = 0;
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, texture_allocator);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * this->size(), sizeof(int), &texture);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		this->texture_allocator_size += sizeof(int);

		glBindBuffer(GL_ARRAY_BUFFER, matrix_allocator_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), sizeof(mat4), &model);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		this->matrix_allocator_size += sizeof(mat4);
	}
	else {
		realloc_buffer(texture_ssbo, GL_SHADER_STORAGE_BUFFER, 1, sizeof(int), GL_DYNAMIC_DRAW, texture_allocator);

		int texture = 0;
		
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, texture_ssbo);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * this->size(), sizeof(int), &texture);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 1);
		glDeleteBuffers(1, &texture_ssbo);

		realloc_buffer(matrix_vbo, GL_ARRAY_BUFFER, 0, sizeof(mat4), GL_DYNAMIC_DRAW, matrix_allocator_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), sizeof(mat4), model.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDeleteBuffers(1, &matrix_allocator_vbo);
	}

	shapes_size++;
}

void sprite_vector2d::erase(uint64_t first, uint64_t last)
{
	std::vector<int> textures(this->size());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, texture_ssbo);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, this->size() * sizeof(int), textures.data());
	if (last != -1) {
		textures.erase(textures.begin() + first, textures.begin() + last);
	}
	else {
		textures.erase(textures.begin() + first);
	}
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * textures.size(), textures.data(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, texture_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	default_2d_base_vector::erase(first, last);
}


template class vertice_handler<shapes::line>;
template class vertice_handler<shapes::square>;

template<shapes shape>
void vertice_handler<shape>::init(const std::vector<float>& l_vertices, const std::vector<float>& l_colors, bool queue) {
	for (int i = 0; i < l_vertices.size(); i++) {
		vertices.push_back(l_vertices[i]);
	}
	for (int i = 0; i < l_colors.size(); i++) {
		colors.push_back(l_colors[i]);
	}
	switch (shape) {
	case shapes::line: {
		mode = GL_LINES;
		point_size = 2;
		break;
	}
	case shapes::square: {
		mode = GL_QUADS;
		point_size = 4;
		break;
	}
	}
	points += point_size;
	static bool once = false;
	if (!once) {
		camera = &camera2d;
		shader = Shader("GLSL/shapes.vs", "GLSL/shapes.frag");
		glGenBuffers(1, &vertice_buffer.VBO);
		glBindBuffer(GL_ARRAY_BUFFER, vertice_buffer.VBO);
		glBufferData(
			GL_ARRAY_BUFFER,
			sizeof(vertices[0]) *
			vertices.size(),
			vertices.data(),
			GL_STATIC_DRAW
		);
		glGenBuffers(1, &color_buffer.VBO);
		glBindBuffer(GL_ARRAY_BUFFER, color_buffer.VBO);
		glBufferData(
			GL_ARRAY_BUFFER,
			sizeof(colors[0]) *
			colors.size(),
			colors.data(),
			GL_STATIC_DRAW
		);
		glGenVertexArrays(1, &shape_buffer.VAO);
		glBindVertexArray(shape_buffer.VAO);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
		once = true;
	}
	else if (!queue) {
		write(true, true);
	}
}

template <shapes shape>
void vertice_handler<shape>::write(bool _EditVertices, bool _EditColor) {
	if (_EditVertices) {
		glBindBuffer(GL_ARRAY_BUFFER, vertice_buffer.VBO);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vertices[0]) *
			vertices.size(),
			vertices.data(),
			GL_STATIC_DRAW
		);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	if (_EditColor) {
		glBindBuffer(GL_ARRAY_BUFFER, color_buffer.VBO);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(colors[0]) *
			colors.size(),
			colors.data(),
			GL_STATIC_DRAW
		);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

template<shapes shape>
void vertice_handler<shape>::draw(uint64_t shape_id) const {
	if (vertices.empty() || colors.empty()) {
		return;
	}
	shader.use();
	mat4 view(1);
	mat4 projection(1);

	view = camera->get_view_matrix(Translate(
		view,
		vec3(
			window_size.x / 2,
			window_size.y / 2,
			-700.0f
		)
	));
	projection = Ortho(
		window_size.x / 2,
		window_size.x + window_size.x * 0.5f,
		window_size.y + window_size.y * 0.5f,
		window_size.y / 2.f,
		0.1f,
		1000.0f
	);
	static int projLoc = glGetUniformLocation(shader.ID, "projection");
	static int viewLoc = glGetUniformLocation(shader.ID, "view");
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);

	glBindVertexArray(shape_buffer.VAO);
	glBindBuffer(GL_ARRAY_BUFFER, vertice_buffer.VBO);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, color_buffer.VBO);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glDrawArrays(mode, shape_id == -1 ? 0 : shape_id * point_size, shape_id == -1 ? points : point_size);
	glBindVertexArray(0);
}

template <shapes shape>
void default_shape<shape>::draw() {
	if constexpr (shape == shapes::line) {
		line_handler.draw(draw_id);
	}
	else if constexpr (shape == shapes::square) {
		square_handler.draw(draw_id);
	}
}

Line::Line(const mat2x2& m, const Color& color) {
	position = m[0];
	size = m[1] - m[0];
	this->color = color;
	std::vector<float> vertices, colors;
	for (int i = 0; i < 4; i++) {
		vertices.push_back(m[(i & 2) >> 1][i & 1]);
	}
	for (int i = 0; i < COLORSIZE * 2; i++) {
		colors.push_back(color[i % 4]);
	}
	draw_id = line_handler.draw_id++;
	line_handler.init(vertices, colors);
}

vec2 Line::get_position() const {
	return position;
}

void Line::set_position(const mat2x2& m) {
	position = m[0];
	size = m[1] - m[0];
	for (int i = 0; i < 4; i++) {
		line_handler.vertices[i + draw_id * 4] = m[(i & 2) >> 1][i & 1];
	}
	line_handler.write(true, false);
}

Color Line::get_color() const {
	return color;
}

void Line::set_color(const Color& color) {
	this->color = color;
	for (int i = 0; i < COLORSIZE * 4; i++) {
		line_handler.colors[i + (COLORSIZE * 2 * draw_id)] = color[i % 4];
	}
	line_handler.write(false, true);
}

vec2 Line::get_size() const {
	return size;
}

void Line::set_size(const vec2& size) {
	vec2 expand(size - this->size);
	this->size = size;
	for (int i = 2; i < 4; i++) {
		line_handler.vertices[i + draw_id * 4] += expand[i & 1];
	}
	line_handler.write(true, false);
}

Square::Square(const vec2& position, const vec2& size, const Color& color, bool queue) {
	this->position = position;
	this->size = size;
	this->color = color;
	std::vector<float> vertices;
	std::vector<float> colors;
	vertices.push_back(position.x);
	vertices.push_back(position.y);
	vertices.push_back(position.x + size.x);
	vertices.push_back(position.y);
	vertices.push_back(position.x + size.x);
	vertices.push_back(position.y + size.y);
	vertices.push_back(position.x);
	vertices.push_back(position.y + size.y);

	for (int i = 0; i < COLORSIZE * 4; i++) {
		colors.push_back(color[i % 4]);
	}
	draw_id = square_handler.draw_id++;
	square_handler.init(vertices, colors, queue);
}

Square::~Square() {
	//square_handler.draw_id--;
}

vec2 Square::get_position() const {
	return position;
}

void Square::set_position(const vec2& position, bool queue) {
	{
		square_handler.vertices[0 + draw_id * 8] = position.x;
		square_handler.vertices[1 + draw_id * 8] = position.y;
		square_handler.vertices[2 + draw_id * 8] = position.x + size.x;
		square_handler.vertices[3 + draw_id * 8] = position.y;
		square_handler.vertices[4 + draw_id * 8] = position.x + size.x;
		square_handler.vertices[5 + draw_id * 8] = position.y + size.y;
		square_handler.vertices[6 + draw_id * 8] = position.x;
		square_handler.vertices[7 + draw_id * 8] = position.y + size.y;
		if (!queue) {
			square_handler.write(true, false);
		}
		this->position = position;
	}
}

Color Square::get_color() const {
	return color;
}

void Square::set_color(const Color& color) {
	this->color = color;
	for (int i = 0; i < COLORSIZE * 4; i++) {
		square_handler.colors[i + (COLORSIZE * 4 * draw_id)] = color[i % 4];
	}
	square_handler.write(false, true);
}

vec2 Square::get_size() const {
	return size;
}

void Square::set_size(const vec2& size, bool queue) {
	square_handler.vertices[0 + draw_id * 8] = position.x;
	square_handler.vertices[1 + draw_id * 8] = position.y;
	square_handler.vertices[2 + draw_id * 8] = position.x + size.x;
	square_handler.vertices[3 + draw_id * 8] = position.y;
	square_handler.vertices[4 + draw_id * 8] = position.x + size.x;
	square_handler.vertices[5 + draw_id * 8] = position.y + size.y;
	square_handler.vertices[6 + draw_id * 8] = position.x;
	square_handler.vertices[7 + draw_id * 8] = position.y + size.y;
	this->size = size;
	if (!queue) {
		square_handler.write(true, false);
	}
}

void Square::break_queue() {
	square_handler.write(true, true);
}

Sprite::Sprite() : texture(), position(0) {}

Sprite::Sprite(const char* path, vec2 position, vec2 size, float angle, Shader shader) :
	shader(shader), angle(angle), position(position), size(size), texture() {
	this->camera = (Camera*)glfwGetWindowUserPointer(window);
	init_image();
	load_image(path, texture);
	if (size.x != 0 && size.y != 0) {
		texture.width = size.x;
		texture.height = size.y;
	}
}

Sprite::Sprite(unsigned char* pixels, const vec2& image_size, const vec2& position, const vec2& size, Shader shader) :
	shader(shader), position(position), size(size), texture() {
	this->camera = (Camera*)glfwGetWindowUserPointer(window);
	init_image();
	texture.width = image_size.x;
	texture.height = image_size.y;
	load_image(pixels, texture);
	if (size.x != 0 && size.y != 0) {
		texture.width = size.x;
		texture.height = size.y;
	}
}

void Sprite::draw() {
	shader.use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture.texture);
	glUniform1i(glGetUniformLocation(shader.ID, "ourTexture1"), 0);
	mat4 view(1);
	mat4 projection(1);
	view = camera->get_view_matrix(Translate(view, vec3(window_size.x / 2, window_size.y / 2, -700.0f)));
	projection = Ortho(window_size.x / 2, window_size.x + window_size.x * 0.5f, window_size.y + window_size.y * 0.5f, window_size.y / 2.f, 0.1f, 1000.0f);
	GLint projLoc = glGetUniformLocation(shader.ID, "projection");
	GLint viewLoc = glGetUniformLocation(shader.ID, "view");
	GLint modelLoc = glGetUniformLocation(shader.ID, "model");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
	mat4 model(1);
	model = Translate(model, position);
	if (size.x || size.y) {
		model = Scale(model, vec3(size.x, size.y, 0));
	}
	else {
		model = Scale(model, vec3(texture.width, texture.height, 0));
	}
	if (angle) {
		model = Rotate(model, angle, vec3(0, 0, 1));
	}
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model[0][0]);
	glBindVertexArray(texture.VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
}

void Sprite::init_image() {
	const float vertices[] = {
		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
		 0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
		 0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
		 0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f
	};
	_Vertices.resize(30);
	for (int i = 0; i < _Vertices.size(); i++) {
		_Vertices[i] = vertices[i];
	}
}

void Sprite::load_image(const char* path, Texture& object) {
	std::ifstream file(path);
	if (!file.good()) {
		printf("File path does not exist\n");
		return;
	}
	glGenVertexArrays(1, &object.VAO);
	glGenBuffers(1, &object.VBO);
	glBindVertexArray(object.VAO);
	glBindBuffer(GL_ARRAY_BUFFER, object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.size(), _Vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);
	glBindVertexArray(0);
	glGenTextures(1, &object.texture);
	glBindTexture(GL_TEXTURE_2D, object.texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	stbi_set_flip_vertically_on_load(true);
	unsigned char* data = SOIL_load_image(path, &object.width, &object.height, 0, SOIL_LOAD_RGBA);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, object.width, object.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	SOIL_free_image_data(data);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Sprite::load_image(unsigned char* pixels, Texture& object) {
	glGenVertexArrays(1, &object.VAO);
	glGenBuffers(1, &object.VBO);
	glBindVertexArray(object.VAO);
	glBindBuffer(GL_ARRAY_BUFFER, object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.size(), _Vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);
	glBindVertexArray(0);
	glGenTextures(1, &object.texture);
	glBindTexture(GL_TEXTURE_2D, object.texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, object.width, object.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, pixels);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Sprite::reload_image(unsigned char* pixels) {
	glBindTexture(GL_TEXTURE_2D, texture.texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width, texture.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, pixels);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);
}

Texture& Sprite::get_texture() {
	return texture;
}

vec2 Sprite::get_size() const {
	return this->size;
}

vec2 Sprite::get_position() const {
	return this->position;
}

void Sprite::set_position(const vec2& position) {
	this->position = position;
}

float Sprite::get_angle() const {
	return angle;
}

void Sprite::set_angle(float angle) {
	this->angle = angle;
}

//LineVector3D::LineVector3D() {
//	_Mode = GL_LINES;
//	_Points = 0;
//	_PointSize = 2 * 2;
//	init();
//}
//
//LineVector3D::LineVector3D(const matrix<3, 2>& _M, const Color& color) {
//	_Mode = GL_LINES;
//	_Points = 2;
//	_PointSize = _Points * 2;
//	for (int i = 0; i < 6; i++) {
//		_Vertices.push_back(_M[i / 3][i % 3]);
//	}
//	for (int i = 0; i < COLORSIZE * 2; i++) {
//		_Colors.push_back(color[i % 4]);
//	}
//	init();
//	this->_Camera = (Camera*)&camera3d;
//}
//
//matrix<2, 3> LineVector3D::get_position(uint64_t _Index) const {
//	return matrix<2, 3>(
//		_Vertices[_Index * _PointSize],
//		_Vertices[_Index * _PointSize + 1],
//		_Vertices[_Index * _PointSize + 2],
//		_Vertices[_Index * _PointSize + 3],
//		_Vertices[_Index * _PointSize + 4],
//		_Vertices[_Index * _PointSize + 5]
//	);
//}
//
//void LineVector3D::set_position(uint64_t _Index, const matrix<3, 2>& _M, bool _Queue) {
//	for (int i = 0; i < 6; i++) {
//		_Vertices[_Index * _PointSize + i] = _M[i / 3][i % 3];
//	}
//	if (!_Queue) {
//		write(true, false);
//	}
//}
//
//void LineVector3D::push_back(const matrix<3, 2>& _M, Color _Color, bool _Queue) {
//	for (int i = 0; i < 6; i++) {
//		_Vertices.push_back(_M[i / 3][i % 3]);
//	}
//	if (_Color.r == -1) {
//		if (_Colors.size() > COLORSIZE) {
//			_Color = Color(_Colors[0], _Colors[1], _Colors[2], _Colors[3]);
//		}
//		else {
//			_Color = Color(1, 1, 1, 1);
//		}
//		for (int i = 0; i < COLORSIZE * 2; i++) {
//			_Colors.push_back(_Color[i % 4]);
//		}
//	}
//	else {
//		for (int i = 0; i < COLORSIZE * 2; i++) {
//			_Colors.push_back(_Color[i % 4]);
//		}
//	}
//	_Points += 2;
//	if (!_Queue) {
//		write(true, true);
//	}
//}

//LineVector::LineVector(const mat2x2& _M, const Color& color) {
//	_Mode = GL_LINES;
//	_Points = 2;
//	_PointSize = _Points * 2;
//	_Length.push_back(vec2(_M[1] - _M[0]));
//	for (int i = 0; i < 4; i++) {
//		_Vertices.push_back(_M[(i & 2) >> 1][i & 1]);
//	}
//	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
//		_Colors.push_back(color[i % 4]);
//	}
//	init();
//}

model_mesh::model_mesh(
	const std::vector<mesh_vertex>& vertices,
	const std::vector<unsigned int>& indices,
	const std::vector<mesh_texture>& textures
) : vertices(vertices), indices(indices), textures(textures) {
	initialize_mesh();
}

void model_mesh::initialize_mesh() {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(mesh_vertex), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(mesh_vertex), 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(mesh_vertex), reinterpret_cast<void*>(offsetof(mesh_vertex, normal)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(mesh_vertex), reinterpret_cast<void*>(offsetof(mesh_vertex, texture_coordinates)));
	glBindVertexArray(0);
}

model_loader::model_loader(const std::string& path) {
	load_model(path);
}

void model_loader::load_model(const std::string& path) {
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

	if (scene == nullptr || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr) {
		std::cout << "assimp error: " << importer.GetErrorString() << '\n';
		return;
	}

	directory = path.substr(0, path.find_last_of('/'));

	process_node(scene->mRootNode, scene);
}

void model_loader::process_node(aiNode* node, const aiScene* scene) {
	for (GLuint i = 0; i < node->mNumMeshes; i++) {
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

		meshes.emplace_back(process_mesh(mesh, scene));
	}

	for (GLuint i = 0; i < node->mNumChildren; i++) {
		process_node(node->mChildren[i], scene);
	}
}

model_mesh model_loader::process_mesh(aiMesh* mesh, const aiScene* scene) {
	std::vector<mesh_vertex> vertices;
	std::vector<GLuint> indices;
	std::vector<mesh_texture> textures;

	for (GLuint i = 0; i < mesh->mNumVertices; i++)
	{
		mesh_vertex vertex;
		vec3 vector;

		vector.x = mesh->mVertices[i].x / 2;
		vector.y = mesh->mVertices[i].y / 2;
		vector.z = mesh->mVertices[i].z / 2;
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

std::vector<mesh_texture> model_loader::load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name) {
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

void basic_3d::init(const std::string& vs, const std::string& fs) {
	_Shader = Shader(vs.c_str(), fs.c_str());

	glGenBuffers(1, &_Shape_Matrix_VBO);
}

void basic_3d::init_matrices() {
	glBindBuffer(GL_ARRAY_BUFFER, _Shape_Matrix_VBO);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)0);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)(sizeof(vec4)));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)(2 * sizeof(vec4)));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)(3 * sizeof(vec4)));
	glEnableVertexAttribArray(6);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttribDivisor(3, 1);
	glVertexAttribDivisor(4, 1);
	glVertexAttribDivisor(5, 1);
	glVertexAttribDivisor(6, 1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void basic_3d::set_projection() {
	if (this->object_matrix.empty()) {
		return;
	}
	mat4 projection(1);
	mat4 view(1);

	view = camera3d.get_view_matrix();
	projection = Perspective(Radians(90.f), (float_t)window_size.x / (float_t)window_size.y, 0.1f, 1000.0f);

	_Shader.use();
	_Shader.set_mat4("view", view);
	_Shader.set_mat4("projection", projection);
}

void basic_3d::set_position(uint64_t index, const vec3& position, bool queue) {
	this->object_matrix[index] = Translate(mat4(1), position);
	if (!queue) {
		glBindBuffer(GL_ARRAY_BUFFER, _Shape_Matrix_VBO);
		glBufferSubData(
			GL_ARRAY_BUFFER,
			index * sizeof(this->object_matrix[index]),
			sizeof(this->object_matrix[index]),
			&this->object_matrix[index]
		);
	}
}

vec3 basic_3d::get_position(uint64_t i) const {
	return vec3(
		object_matrix[i][3][0],
		object_matrix[i][3][1],
		object_matrix[i][3][2]
	);
}

vec3 basic_3d::get_size(uint64_t i) const {
	vec3 size;
	size.x = object_matrix[i][0][0];
	size.y = object_matrix[i][1][1];
	size.z = object_matrix[i][2][2];
	return size;
}


void basic_3d::push_back(const vec3& position, const vec3& size, bool queue) {
	mat4 object(1);
	object = Translate(object, position);
	
	if (size != vec3(1)) {
		object = Scale(object, size);
	}

	object_matrix.push_back(object);

	if (!queue) {
#ifdef RAM_SAVER
		basic_3d::free_queue();
#else 
		glBufferSubData(
			GL_ARRAY_BUFFER,
			sizeof(this->object_matrix[0]) * (object_matrix.size() - 1),
			sizeof(this->object_matrix[0]),
			&this->object_matrix[object_matrix.size() - 1]
		);
#endif
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void basic_3d::insert(const std::vector<mat4> positions, const vec3& size, bool queue) {
	object_matrix.insert(object_matrix.end(), positions.begin(), positions.end());
	if (!queue) {
#ifdef RAM_SAVER 
		basic_3d::free_queue();
#else
		glBufferSubData(
			GL_ARRAY_BUFFER,
			sizeof(this->object_matrix[0]) * (object_matrix.size() - objects.size()),
			sizeof(this->object_matrix[0]) * objects.size(),
			&this->object_matrix[object_matrix.size() - objects.size()]
		);
#endif
	}
}

void basic_3d::free_queue() {
	glBindBuffer(GL_ARRAY_BUFFER, _Shape_Matrix_VBO);
	glBufferData(
		GL_ARRAY_BUFFER,
		sizeof(this->object_matrix[0]) * object_matrix.size(),
		&this->object_matrix[0],
		GL_DYNAMIC_DRAW
	);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

uint64_t basic_3d::size() const {
	return this->object_matrix.size();
}

SquareVector3D::SquareVector3D(std::string_view path) {
	this->init(path);
}

void SquareVector3D::init(std::string_view path)  {
	glGenVertexArrays(1, &_Shape_VAO);
	glGenBuffers(1, &_Shape_Vertices_VBO);
	glBindVertexArray(_Shape_VAO);
	glBindBuffer(GL_ARRAY_BUFFER, _Shape_Vertices_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(square_vertices), square_vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vec3::size() * sizeof(float_t), 0);

	basic_3d::init("GLSL/instancing.vs", "GLSL/instancing.frag");
	basic_3d::init_matrices();

	generate_textures(path);

	glGenBuffers(1, &_Texture_Id_SSBO);
	glGenBuffers(1, &_Texture_SSBO);

	std::vector<vec2> textures(texture_coordinate_size * ceil(_Textures.size() / 2));
	for (int j = 0; j < texture_coordinate_size * _Textures.size(); j++) {
		textures[j / 2][j & 1] = _Textures[j / texture_coordinate_size][j % texture_coordinate_size];
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, _Texture_SSBO);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		sizeof(textures[0]) * textures.size(),
		textures.data(),
		GL_STATIC_DRAW
	);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _Texture_SSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#ifndef RAM_SAVER
	constexpr auto gpu_prealloc = sizeof(float) * 100000000;
	glBindBuffer(GL_ARRAY_BUFFER, _Shape_Matrix_VBO);
	glBufferData(
		GL_ARRAY_BUFFER,
		gpu_prealloc,
		nullptr,
		GL_DYNAMIC_DRAW
	);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, _Texture_Id_SSBO);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		gpu_prealloc,
		nullptr,
		GL_DYNAMIC_DRAW
	);
#endif

	glBindVertexArray(0);
}

void SquareVector3D::free_queue(bool vertices, bool texture) {
	if (vertices) {
		basic_3d::free_queue();
	}
	if (texture) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, _Texture_Id_SSBO);
		glBufferData(
			GL_SHADER_STORAGE_BUFFER,
			sizeof(_Texture_Ids[0]) * _Texture_Ids.size(),
			_Texture_Ids.data(),
			GL_DYNAMIC_DRAW
		);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _Texture_Id_SSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
}

void SquareVector3D::draw() {
	basic_3d::set_projection();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _Texture_VBO);
	_Shader.set_int("texture_sampler", 0);

	glBindVertexArray(_Shape_VAO);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 36, this->size());
	glBindVertexArray(0);
}

template <typename T>
std::vector<T> SquareVector3D::get_texture_onsided(_vec2<uint32_t> size, _vec2<uint32_t> position) {
	vec2 b(1.f / size.x, 1.f / size.y);
	vec2 x(position.x * b.x, position.y * b.y);

	return std::vector<T>{
		x.x, 1.f - x.y,
			x.x, 1.f - (x.y + b.y),
			x.x + b.x, 1.f - (x.y + b.y),
			x.x + b.x, 1.f - (x.y + b.y),
			x.x + b.x, 1.f - x.y,
			x.x, 1.f - x.y
	};
}

void SquareVector3D::insert(const std::vector<mat4> objects, const vec3& size, const vec2& texture_id, bool queue) {
	object_matrix.insert(object_matrix.end(), objects.begin(), objects.end());
	_Texture_Ids.insert(_Texture_Ids.end(), objects.size(), texturepack_size.x / 6 * texture_id.y + texture_id.x);

	if (!queue) {
#ifdef RAM_SAVER 
		basic_3d::free_queue();

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, _Texture_Id_SSBO);
		glBufferData(
			GL_SHADER_STORAGE_BUFFER,
			sizeof(_Texture_Ids[0]) * _Texture_Ids.size(),
			&_Texture_Ids[0],
			GL_DYNAMIC_DRAW
		);
#else
		glBufferSubData(
			GL_ARRAY_BUFFER,
			sizeof(this->object_matrix[0]) * (object_matrix.size() - objects.size()),
			sizeof(this->object_matrix[0]) * objects.size(),
			&this->object_matrix[object_matrix.size() - objects.size()]
		);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, _Texture_Id_SSBO);
		glBufferSubData(
			GL_SHADER_STORAGE_BUFFER,
			sizeof(this->_Texture_Ids[0]) * (_Texture_Ids.size() - objects.size()),
			sizeof(this->_Texture_Ids[0]) * objects.size(),
			&this->_Texture_Ids[_Texture_Ids.size() - objects.size()]
		);
#endif
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _Texture_Id_SSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
}

void SquareVector3D::push_back(const vec3& position, const vec3& size, const vec2& texture_id, bool queue) {
	basic_3d::push_back(position, size, queue);
	_Texture_Ids.push_back(texturepack_size.x / 6 * texture_id.y + texture_id.x);
	if (!queue) {
#ifdef RAM_SAVER 
		basic_3d::free_queue();

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, _Texture_Id_SSBO);
		glBufferData(
			GL_SHADER_STORAGE_BUFFER,
			sizeof(_Texture_Ids[0]) * _Texture_Ids.size(),
			_Texture_Ids.data(),
			GL_DYNAMIC_DRAW
		);
#else
		glBufferSubData(
			GL_ARRAY_BUFFER,
			sizeof(this->object_matrix[0]) * (object_matrix.size() - 1),
			sizeof(this->object_matrix[0]),
			&this->object_matrix[object_matrix.size() - 1]
		);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, _Texture_Id_SSBO);
		glBufferSubData(
			GL_SHADER_STORAGE_BUFFER,
			sizeof(this->_Texture_Ids[0]) * (_Texture_Ids.size() - 1),
			sizeof(this->_Texture_Ids[0]),
			&this->_Texture_Ids[_Texture_Ids.size() - 1]
		);
#endif
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _Texture_Id_SSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void SquareVector3D::erase(uint64_t first, uint64_t last, bool queue) {
	if (last != -1) {
		this->object_matrix.erase(
			this->object_matrix.begin() + first, 
			this->object_matrix.begin() + last
		);

		glBufferSubData(
			GL_ARRAY_BUFFER,
			first * sizeof(this->object_matrix[first]),
			sizeof(this->object_matrix[first]) * (last - first),
			0
		);
	}
	else {
		this->object_matrix.erase(this->object_matrix.begin() + first);

		glBufferSubData(
			GL_ARRAY_BUFFER,
			first * sizeof(this->object_matrix[first]),
			sizeof(this->object_matrix[first]),
			0
		);
	}
}

void SquareVector3D::generate_textures(std::string_view path) {
	glGenTextures(1, &_Texture_VBO);
	glBindTexture(GL_TEXTURE_2D, _Texture_VBO);

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

	texturepack_size = vec2(image_size.x / texture_size.x, image_size.y / texture_size.y);
	vec2 amount_of_textures = vec2(texturepack_size.x / 6, texturepack_size.y);
	constexpr int side_order[] = { 0, 1, 4, 5, 3, 2 };
	_Textures.resize(amount_of_textures.x * amount_of_textures.y);
	int current_texture = 0;
	for (vec2i texture; texture.y < amount_of_textures.y; texture.y++) {
		const vec2 begin(1.f / texturepack_size.x, 1.f / texturepack_size.y);
		const float up = 1 - begin.y * texture.y;
		const float down = 1 - begin.y * (texture.y + 1);
		for (texture.x = 0; texture.x < amount_of_textures.x; texture.x++) {
			for (int side = 0; side < ArrLen(side_order); side++) {
				const float left = begin.x * side_order[side] + ((begin.x * (texture.x)) * 6);
				const float right = begin.x * (side_order[side] + 1) + ((begin.x * (texture.x)) * 6);
				const float texture_coordinates[] = {
					left,  up,
					left,  down,
					right, down,
					right, down,
					right, up,
					left,  up
				};
				for (auto coordinate : texture_coordinates) {
					_Textures[current_texture].push_back(coordinate);
				}
			}
			current_texture++;
		}		
	}
}

void add_chunk(SquareVector3D& square_vector, const vec3& position, const vec3& chunk_size, const vec2& texture_id, bool queue) {
	const matrix<4, 4> base_matrix(1);
	std::vector<matrix<4, 4>> objects(chunk_size.x * chunk_size.y * chunk_size.z, matrix<4, 4>(1));
	int index = 0;
	for (int i = 0; i < chunk_size.x; i++) {
		for (int j = 0; j < chunk_size.y; j++) {
			for (int k = 0; k < chunk_size.z; k++) {
				const vec3 temp_vector = vec3(i, j, k) + position;
				objects[index][3] =
					(base_matrix[0] * temp_vector.x) +
					(base_matrix[1] * temp_vector.y) +
					(base_matrix[2] * temp_vector.z) +
					base_matrix[3];
				index++;
			}
		}
	}
	square_vector.insert(objects, vec3(1), texture_id, queue);
}

void remove_chunk(SquareVector3D& square_vector, uint64_t chunk) {
	square_vector.erase(chunk, chunk + 16 * 16 * 16);
}

Model::Model(
	const std::string& path, 
	const std::string& vs,
	const std::string& frag
) : model_loader(path) {

	basic_3d::init(vs, frag);

	for (int i = 0; i < this->meshes.size(); i++) {
		glBindVertexArray(this->meshes[i].vao);
		init_matrices();
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

void Model::draw() {
	basic_3d::set_projection();

	_Shader.set_int("texture_sampler", 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->textures_loaded[0].id);

	_Shader.set_vec3("light_position", camera3d.get_position());
	_Shader.set_vec3("view_position", camera3d.get_position());
	_Shader.set_vec3("light_color", vec3(1, 1, 1));
	_Shader.set_int("texture_diffuse", 0);
	//_Shader.set_vec3("sky_color", vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));
	glDepthFunc(GL_LEQUAL);
	for (int i = 0; i < this->meshes.size(); i++) {
		glBindVertexArray(this->meshes[i].vao);
		glDrawElementsInstanced(GL_TRIANGLES, this->meshes[i].indices.size(), GL_UNSIGNED_INT, 0, this->size());
	}
	glDepthFunc(GL_LESS);

	glBindVertexArray(0);
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

	const vec3 n = Normalize(Cross((a - p0), (b - p0)));

	const vec3 l = DirectionVector(camera3d.yaw, camera3d.pitch);

	const float nl_dot(Dot(n, l));

	if (!nl_dot) {
		return vec3(-1);
	}

	const float d = Dot(p0 - l0, n) / nl_dot;
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

skybox::skybox(
	const std::string& left, 
	const std::string& right, 
	const std::string& front, 
	const std::string back, 
	const std::string bottom, 
	const std::string& top
) : shader("GLSL/skybox.vs", "GLSL/skybox.frag"), camera(&camera3d) {
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
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glBindVertexArray(0);
}

skybox::~skybox() {
	glDeleteVertexArrays(1, &skybox_vao);
	glDeleteBuffers(1, &skybox_vbo);
	glDeleteTextures(1, &texture_id);
}

void skybox::draw() {
	shader.use();

	mat4 view(1);
	mat4 projection(1);

	view = mat4(mat3(camera->get_view_matrix()));
	projection = Perspective(Radians(90.f), (float_t)window_size.x / (float_t)window_size.y, 0.1f, 1000.0f);

	shader.set_mat4("view", view);
	shader.set_mat4("projection", projection);
	shader.set_vec3("fog_color", vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));

	glDepthFunc(GL_LEQUAL);
	glBindVertexArray(skybox_vao);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glDepthFunc(GL_LESS);
}

model_skybox::model_skybox(const std::string& path) : Model(path, "GLSL/skybox_model.vs", "GLSL/skybox_model.frag") {}

void model_skybox::draw() {
	if (this->object_matrix.empty()) {
		return;
	}
	mat4 projection(1);
	mat4 view(1);

	view = mat4(mat3(camera3d.get_view_matrix()));
	projection = Perspective(Radians(90.f), (float_t)window_size.x / (float_t)window_size.y, 0.1f, 1000.0f);

	_Shader.use();
	_Shader.set_mat4("view", view);
	_Shader.set_mat4("projection", projection);

	_Shader.set_int("texture_sampler", 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->textures_loaded[0].id);

	glDepthFunc(GL_LEQUAL);
	for (int i = 0; i < this->meshes.size(); i++) {
		glBindVertexArray(this->meshes[i].vao);
		glDrawElementsInstanced(GL_TRIANGLES, this->meshes[i].indices.size(), GL_UNSIGNED_INT, 0, this->size());
	}
	glDepthFunc(GL_LESS);

	glBindVertexArray(0);
}

#include <ft2build.h>
#include FT_FREETYPE_H
#include <FAN/File.hpp>

std::vector<unsigned int> advance_data;

//void load_to_ram() {
//	FT_Library ft;
//
//	if (FT_Init_FreeType(&ft))
//		std::cout << "Could not init FreeType Library" << std::endl;
//
//	FT_Face face;
//	if (FT_New_Face(ft, "fonts/arial.ttf", 0, &face))
//		std::cout << "Failed to load font" << std::endl;
//
//	for (int i = 0; i < 100; i++) {
//		for (int j = 0; j < max_ascii; j++) {
//			FT_Set_Pixel_Sizes(face, i, i);
//			advance_data.push_back(suckless_getwidth(ft, face, j, i));
//		}
//	}
//}
//
//void update_advance_data() {
//	FT_Library ft;
//
//	if (FT_Init_FreeType(&ft))
//		std::cout << "Could not init FreeType Library" << std::endl;
//
//	FT_Face face;
//	if (FT_New_Face(ft, "fonts/arial.ttf", 0, &face))
//		std::cout << "Failed to load font" << std::endl;
//
//	std::vector<unsigned int> data;
//	for (int i = 0; i < max_font_size; i++) {
//		for (int j = 0; j < max_ascii; j++) {
//			FT_Set_Pixel_Sizes(face, i, i);
//			data.push_back(suckless_getwidth(ft, face, j, i));
//		}
//	}
//
//	File::write<unsigned int>("fonts/advance", data, std::ios::binary);
//}


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
	if (FT_New_Face(ft, "fonts/calibri.ttf", 0, &face))
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
		fprintf(stderr, "data cant hold more\n");
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

text_renderer::text_renderer(float_t font_size) : shader(Shader("GLSL/text.vs", "GLSL/text.frag")) {
	FT_Library ft;

	if (FT_Init_FreeType(&ft))
		std::cout << "Could not init FreeType Library" << std::endl;

	FT_Face face;
	if (FT_New_Face(ft, "fonts/arial.ttf", 0, &face))
		std::cout << "Failed to load font" << std::endl;

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	FT_Set_Pixel_Sizes(face, 0, font_size);

	font = suckless_font_fr(1024, font_size);

	for (int i = 33; i < 248; i++) {
		infos[i] = suckless_font_add_f(ft, &font, i);
	}

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
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

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

//void TextRenderer::update(const std::string& text, float_t font_size)
//{
//	if (text == current_str) {
//		return;
//	}
//
//	current_str = text;
//
//	static FT_Library ft;
//	static FT_Face face;
//
//	static bool once = false;
//
//	if (!once) {
//		if (FT_Init_FreeType(&ft))
//			std::cout << "Could not init FreeType Library" << std::endl;
//		if (FT_New_Face(ft, "fonts/arial.ttf", 0, &face))
//			std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
//		once = true;
//	}
//
//	FT_Set_Pixel_Sizes(face, 0, font_size);
//	suckless_font_t font = suckless_font_fr(ft, face, font_size, text.data());
//
//	glBindTexture(GL_TEXTURE_2D, texture);
//	glTexImage2D(
//		GL_TEXTURE_2D,
//		0,
//		GL_RED,
//		font.datasize,
//		font.datasize,
//		0,
//		GL_RED,
//		GL_UNSIGNED_BYTE,
//		font.data.data()
//	);
//
//	glBindTexture(GL_TEXTURE_2D, 0);
//
//	string = Character{
//		texture,
//		vec2(font.datasize),
//		vec2i(0, 0),
//		(unsigned int)font.datasize
//	};
//}

void text_renderer::render(const std::string& text, vec2 position, const Color& color, float_t scale) {
	shader.use();
	shader.set_mat4("projection", Ortho(0, window_size.x, window_size.y, 0));
	shader.set_vec4("textColor", color);

	glBindVertexArray(VAO);

	std::vector<std::array<std::array<float_t, 4>, 6>> datas;

	letter_info_t _letter;

	const float_t begin = position.x;

	for (int i = 0; i < text.size(); i++) {
		_letter = infos[text[i]];

		float_t xpos = position.x;
		float_t ypos = position.y - 1;
		vec2 size(_letter.width, font.fontsize);
		size = size / size.max() * scale;

		if (text[i] == ' ') {
			position.x += 10;
		}
		else {
			position.x += size.x + 2;
		}

		if (text[i] == '\n') {
			position.x = begin;
			position.y += button_text_gap;
		}

		letter_info_opengl_t letter = letter_to_opengl(font, _letter);
		float_t height = letter.pos.y + ((float_t)font.fontsize / font.datasize);

		std::array<std::array<float_t, 4>, 6> vertices = { {
			{ xpos,          ypos,                     letter.pos.x, letter.pos.y },
			{ xpos,          ypos + size.y,            letter.pos.x, height },
			{ xpos + size.x, ypos + size.y,            letter.pos.x + letter.width, height },

			{ xpos,          ypos,                     letter.pos.x, letter.pos.y },
			{ xpos + size.x, ypos + size.y,            letter.pos.x + letter.width, height },
			{ xpos + size.x, ypos,                     letter.pos.x + letter.width, letter.pos.y }
		}
		};
		datas.push_back(vertices);
	}

	glBindTexture(GL_TEXTURE_2D, texture);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(datas[0]) * datas.size(), datas.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDrawArrays(GL_TRIANGLES, 0, 6 * datas.size());

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);
}

void text_renderer::render(const std::vector<std::string>& text, std::vector<vec2> position, const std::vector<Color>& color, const std::vector<float_t>& scale)
{
	shader.use();
	shader.set_mat4("projection", Ortho(0, window_size.x, window_size.y, 0));
	shader.set_vec4("textColor", color[0]);

	std::vector<std::array<std::array<float_t, 4>, 6>> datas(text.size());
	glBindVertexArray(VAO);

	for (int instance = 0; instance < text.size(); instance++) {

		letter_info_t _letter;

		float_t x_position = position[instance].x;

		float_t begin = x_position;

		for (int i = 0; i < text[instance].size(); i++) {
			_letter = infos[text[instance][i]];

			vec2 size(_letter.width, font.fontsize);
			size = size / size.max() * scale[instance];

			float_t xpos = x_position;
			float_t ypos = position[instance].y;

			if (text[instance][i] == ' ') {
				x_position += 10;
			}
			else {
				x_position += size.x + 2;
			}

			if (text[instance][i] == '\n') {
				x_position = begin;
				position[instance].y += button_text_gap;
			}

			letter_info_opengl_t letter = letter_to_opengl(font, _letter);
			float_t height = letter.pos.y + ((float_t)font.fontsize / font.datasize);

			std::array<std::array<float_t, 4>, 6> vertices = { {
				{ xpos,          ypos,                     letter.pos.x,     letter.pos.y },
				{ xpos,          ypos + size.y,            letter.pos.x, height },
				{ xpos + size.x, ypos + size.y,            letter.pos.x + letter.width, height },

				{ xpos,          ypos,                     letter.pos.x, letter.pos.y },
				{ xpos + size.x, ypos + size.y,            letter.pos.x + letter.width, height },
				{ xpos + size.x, ypos,                     letter.pos.x + letter.width, letter.pos.y }
			}
			};
			datas.push_back(vertices);
		}
	}

	glBindTexture(GL_TEXTURE_2D, texture);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(datas[0]) * datas.size(), datas.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDrawArrays(GL_TRIANGLES, 0, 6 * datas.size());

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

vec2 text_renderer::get_length(const std::string& text, float_t scale)
{
	letter_info_t _letter;

	vec2 string_size;

	float_t new_line = 0;

	string_size.y = -1;

	for (int i = 0; i < text.size(); i++) {
		_letter = infos[text[i]];
		if (text[i] == ' ') {
			string_size.x += 10;
		}
		else if (text[i] == '\n') {
			new_line += button_text_gap;
		}
		else {
			vec2 size(_letter.width, font.fontsize);
			size = size / size.max() * scale;

			if (i != text.size() - 1) {
				string_size.x += size.x + 2;
			}
			else {
				string_size.x += size.x;
			}
		}
		string_size.y = std::max(static_cast<float_t>(_letter.height), string_size.y);
	}

	return string_size + vec2(0, new_line);
}

std::vector<vec2> text_renderer::get_length(const std::vector<std::string>& texts, const std::vector<float_t>& scales, bool half)
{
	letter_info_t _letter;
	std::vector<vec2> string_size(texts.size());

	for (int text = 0; text < texts.size(); text++) {
		for (int i = 0; i < texts[text].size(); i++) {
			_letter = infos[texts[text][i]];

			if (texts[text][i] == ' ') {
				if (half) {
					string_size[text].x += 2.5;
				}
				else {
					string_size[text].x += 5;
				}
			}
			else {
				vec2 size(_letter.width, font.fontsize);
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

fan_gui::button_vector::button_vector() : square_vector2d(), renderer(font_size) {}

fan_gui::button_vector::button_vector(const vec2& position, const Color& color)
	: square_vector2d(position, button_text_gap, color), renderer(font_size), max_width(INF)
{
	texts.push_back("");
	button_character_callbacks.push_back([this](int button_id, int key) {
		texts[button_id].push_back(key);
		edit_size(button_id, texts[button_id]);
	});
}

fan_gui::button_vector::button_vector(const std::string& text, const vec2& position, const Color& color)
	: square_vector2d(position, button_text_gap, color), renderer(font_size), max_width(INF)
{
	texts.push_back(text);
	button_character_callbacks.push_back([this](int button_id, int key) {
		texts[button_id].push_back(key);
		edit_size(button_id, texts[button_id]);
	});
}

fan_gui::button_vector::button_vector(const vec2& position, float_t max_width, const Color& color)
	: renderer(text_renderer(font_size)), max_width(max_width), square_vector2d(position, vec2(), color)
{
	set_size(0, button_text_gap);
	texts.push_back("");
	button_character_callbacks.push_back([&](int button_id, int key) {
		auto found = texts[button_id].find_last_of('\n');
		if (get_length(texts[button_id].substr(found == std::string::npos ? 0 : found + 1) + (char)key, font_size).x + button_text_gap > this->max_width) {
			texts[button_id].push_back('\n');
		}
		texts[button_id].push_back(key);
		edit_size(button_id, texts[button_id]);
	});
}

fan_gui::button_vector::button_vector(const std::string& text, const vec2& position, float_t max_width, const Color& color)
	: renderer(text_renderer(font_size)), max_width(max_width), square_vector2d(position, vec2(), color)
{
	set_size(0, vec2(renderer.get_length(text, font_size).x, font_size) + button_text_gap);
	texts.push_back(text);
	button_character_callbacks.push_back([&] (int button_id, int key) {
		auto found = texts[button_id].find_last_of('\n');
		if (get_length(texts[button_id].substr(found == std::string::npos ? 0 : found + 1) + (char)key, font_size).x + button_text_gap > this->max_width) {
			texts[button_id].push_back('\n');
		}
		texts[button_id].push_back(key);
		edit_size(button_id, texts[button_id]);
	});
}

fan_gui::button_vector::button_vector(const std::string& text, const vec2& position, const vec2& size, const Color& color)
	: square_vector2d(position, size, color), renderer(font_size), max_width(INF) {
	button_character_callbacks.push_back([&](int button_id, int key) {
		texts[button_id].push_back(key);
		edit_size(button_id, texts[button_id]);
	});
	texts.push_back(text);
}

void fan_gui::button_vector::add(const std::string& text, const vec2& position, const vec2& size, const Color& color)
{
	push_back(position, size, color);
	texts.push_back(text);
}

void fan_gui::button_vector::add(const vec2& position, const Color& color)
{
	button_character_callbacks.push_back(std::function([&](int button_id, int key) {
		texts[button_id].push_back(key);
		edit_size(button_id, texts[button_id]);
	}));
	push_back(position, button_text_gap, color);
	texts.push_back("");
}

void fan_gui::button_vector::add(const std::string& text, const vec2& position, const Color& color)
{
	push_back(position, renderer.get_length(text, font_size) + button_text_gap, color);
	texts.push_back(text);
}

void fan_gui::button_vector::edit_size(uint64_t i, const std::string& text)
{
	std::vector<std::string> lines;
	int offset = 0;
	for (int i = 0; i < text.size(); i++) {
		if (text[i] == '\n') {
			lines.push_back(text.substr(offset, i - offset));
			offset = i + 1;
		}
	}
	lines.push_back(text.substr(offset, text.size()));
	float_t largest = -9999999;
	for (auto i : lines) {
		float_t size = renderer.get_length(i, font_size).x;
		if (size > largest) {
			largest = size;
		}
	}
	vec2 square_size(vec2(largest, font_size * lines.size()) + vec2(button_text_gap, button_text_gap));
	set_size(i, square_size);
	texts[i] = text;
}

void fan_gui::button_vector::draw(uint64_t i)
{
	square_vector2d::draw();
	glDisable(GL_DEPTH_TEST);
	auto pos = get_position(i) + button_text_gap / 2;
	renderer.render(texts[i], pos, default_text_color, font_size);
	glEnable(GL_DEPTH_TEST);
}

void fan_gui::button_vector::draw()
{
	square_vector2d::draw();
	glDisable(GL_DEPTH_TEST);
	std::vector<vec2> positions = get_positions();
	std::vector<vec2> gaps(size(), button_text_gap / 2);
	std::transform(positions.begin(), positions.end(), gaps.begin(), positions.begin(), std::plus<vec2>());
	renderer.render(texts, positions, std::vector<Color>(size(), default_text_color), std::vector<float_t>(size(), font_size));
	glEnable(GL_DEPTH_TEST);
}

bool fan_gui::button_vector::inside(uint64_t i) const
{
	const vec2 size = get_size(i);
	const vec2 position = get_position(i);
	return cursor_position.x > position.x && cursor_position.x < position.x + size.x &&
		cursor_position.y > position.y && cursor_position.y < position.y + size.y;
}

void fan_gui::button_vector::on_click(uint64_t i, int key, std::function<void()> function) const
{
	if (inside(i) && key_press(key)) {
		function();
	}
}

void fan_gui::button_vector::on_click(const std::string& button_id, int key, std::function<void()> function) const
{
	auto found = std::find(texts.begin(), texts.end(), button_id);

	if (found != texts.end()) {
		if (inside(std::distance(texts.begin(), found)) && key_press(key)) {
			function();
		}
	}

}

vec2 fan_gui::button_vector::get_length(const std::string& text, float_t font_size)
{
	return renderer.get_length(text, font_size);
}

std::function<void(int, int)> fan_gui::button_vector::get_character_callback(uint64_t i) const
{
	return this->button_character_callbacks[i];
}

std::function<void(int)> fan_gui::button_vector::get_newline_callback() const
{
	return this->button_newline_callback;
}

std::function<void(int)> fan_gui::button_vector::get_erase_callback() const
{
	return this->button_erase_callback;
}