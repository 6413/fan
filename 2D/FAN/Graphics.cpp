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

Texture::Texture() : texture(0), width(0), height(0), VBO(0), VAO(0), EBO(0) { }

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

mat4 Camera::get_view_matrix(mat4 m) {
	return m * LookAt(this->position, (this->position + (this->front)).rounded(), (this->up).rounded());
}

mat4 Camera::get_view_matrix() {
	return LookAt(this->position, (this->position + (this->front)), (this->up));
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

template <typename _Type, uint64_t N>
basic_2dshape_vector::basic_2dshape_vector(const std::array<_Type, N>& init_vertices) :
	color_allocated(false), matrix_allocated(false), shapes_size(0)
{
	glGenVertexArrays(1, &shape_vao);
	std::array<unsigned int*, 3> vbos{
		&matrix_vbo, &vertex_vbo, &color_vbo
	};
	glGenBuffers(vbos.size(), *vbos.data());

	glBindVertexArray(shape_vao);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Type) * init_vertices.size(), init_vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
	glVertexAttribDivisor(1, 1);

	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	for (int i = 3; i < 7; i++) {
		glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), reinterpret_cast<void*>((i - 3) * sizeof(vec4)));
		glEnableVertexAttribArray(i);
		glVertexAttribDivisor(i, 1);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

basic_2dshape_vector::~basic_2dshape_vector()
{
	glDeleteBuffers(1, &color_vbo);
	glDeleteBuffers(1, &matrix_vbo);
	glDeleteBuffers(1, &vertex_vbo);
	if (this->color_allocated) {
		glDeleteBuffers(1, &color_allocator_vbo);
	}
	if (this->matrix_allocated) {
		glDeleteBuffers(1, &matrix_allocator_vbo);
	}
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
		this->copy_data(color_allocator_vbo, GL_ARRAY_BUFFER, sizeof(Color) * this->size(), GL_DYNAMIC_DRAW, color_vbo);

		glBindBuffer(GL_ARRAY_BUFFER, color_allocator_vbo);
		Color data;
		glGetBufferSubData(GL_ARRAY_BUFFER, sizeof(Color), sizeof(Color), data.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		printf("%f %f %f %f\n", data.r, data.g, data.b, data.a);

		glDeleteBuffers(1, &color_allocator_vbo);
		this->color_allocated = false;
	}

	if (matrices) {
		glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		this->copy_data(matrix_allocator_vbo, GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), GL_DYNAMIC_DRAW, matrix_vbo);
		glDeleteBuffers(1, &matrix_allocator_vbo);
		this->matrix_allocated = false;
	}
}

constexpr auto copy_buffer = 5000000;

void basic_2dshape_vector::realloc_copy_data(unsigned int& buffer, uint64_t buffer_type, int size, GLenum usage, unsigned int& allocator)
{
	int old_buffer_size = 0;
	glBindBuffer(buffer_type, buffer);
	glGetBufferParameteriv(buffer_type, GL_BUFFER_SIZE, (int*)&old_buffer_size);

	const int new_size = size * copy_buffer;

	glGenBuffers(1, &allocator);
	glBindBuffer(GL_COPY_WRITE_BUFFER, allocator);
	glBufferData(GL_COPY_WRITE_BUFFER, new_size, nullptr, usage);

	glBindBuffer(GL_COPY_READ_BUFFER, buffer);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, old_buffer_size);
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void basic_2dshape_vector::copy_data(unsigned int& buffer, uint64_t buffer_type, int size, GLenum usage, unsigned int& allocator)
{
	glBindBuffer(GL_COPY_WRITE_BUFFER, allocator);
	glBindBuffer(GL_COPY_READ_BUFFER, buffer);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, size);
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void basic_2dshape_vector::realloc_buffer(unsigned int& buffer, uint64_t buffer_type, int location, int size, GLenum usage, unsigned int& allocator) {
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

int basic_2dshape_vector::size() const
{
	return shapes_size;
}

// shape functions
void basic_2dshape_vector::erase(uint64_t first, uint64_t last)
{
	std::vector<Color> colors(this->size());
	std::vector<mat4> matrices(this->size());
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, this->size() * sizeof(Color), colors.data());
	if (last != -1) {
		colors.erase(colors.begin() + first, colors.begin() + last + 1);
	}
	else {
		colors.erase(colors.begin() + first);
	}
	glBufferData(GL_ARRAY_BUFFER, sizeof(Color) * colors.size(), colors.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, this->size() * sizeof(mat4), matrices.data());
	if (last != -1) {
		matrices.erase(matrices.begin() + first, matrices.begin() + last + 1);
	}
	else {
		matrices.erase(matrices.begin() + first);
	}
	glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * matrices.size(), matrices.data(), GL_DYNAMIC_DRAW);
	shapes_size -= (last == -1 ? 1 : (last - first));
}

void basic_2dshape_vector::push_back(const vec2& position, const vec2& size, const Color& color, bool queue)
{
	mat4 model(1);
	model = Translate(model, position);
	model = Scale(model, size);

	if (queue) {
		if (!this->color_allocated) {
			this->realloc_copy_data(color_vbo, GL_ARRAY_BUFFER, sizeof(Color), GL_DYNAMIC_DRAW, color_allocator_vbo);
			this->color_allocated = true;
			this->realloc_copy_data(matrix_vbo, GL_ARRAY_BUFFER, sizeof(mat4), GL_DYNAMIC_DRAW, matrix_allocator_vbo);
			this->matrix_allocated = true;
		}
		glBindBuffer(GL_ARRAY_BUFFER, color_allocator_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(Color) * this->size(), sizeof(Color), &color);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, matrix_allocator_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * this->size(), sizeof(mat4), &model);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
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

vec2 basic_2dshape_vector::get_position(uint64_t index) const
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

void basic_2dshape_vector::set_position(uint64_t index, const vec2& position, bool queue)
{
	mat4 matrix(1);
	matrix = Translate(matrix, position);
	matrix = Scale(matrix, get_size(index));
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * index, sizeof(mat4), matrix.data());
}

vec2 basic_2dshape_vector::get_size(uint64_t index) const
{
	vec2 size;
	mat4 matrix;
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, index * sizeof(mat4), sizeof(mat4), matrix.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	size.x = matrix[0][0];
	size.y = matrix[1][1];
	return size;
}

void basic_2dshape_vector::set_size(uint64_t index, const vec2& size, bool queue)
{
	mat4 matrix(1);
	matrix = Translate(matrix, get_position(index));
	matrix = Scale(matrix, size);
	glBindBuffer(GL_ARRAY_BUFFER, matrix_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(mat4) * index, sizeof(mat4), matrix.data());
}

void basic_2dshape_vector::basic_shape_draw(unsigned int mode, uint64_t points) const
{
	int amount_of_objects = size();
	if (!amount_of_objects) {
		return;
	}

	mat4 view(1);
	mat4 projection(1);

	view = camera2d.get_view_matrix(Translate(view, vec3(window_size.x / 2, window_size.y / 2, -700.0f)));
	projection = Ortho(window_size.x / 2, window_size.x + window_size.x * 0.5f, window_size.y + window_size.y * 0.5f, window_size.y / 2.f, 0.001f, 1000.0f);

	shape_shader2d.use();
	shape_shader2d.set_mat4("projection", projection);
	shape_shader2d.set_mat4("view", view);


	glBindVertexArray(shape_vao);
	glDrawArraysInstanced(mode, 0, points, amount_of_objects);
	glBindVertexArray(0);
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
		basic_2dshape_vector::push_back(position[1], position[0] - position[1], color, queue);
	}
}

mat2 line_vector2d::get_position(uint64_t index) const
{
	return mat2(
		basic_2dshape_vector::get_position(index),
		basic_2dshape_vector::get_size(index)
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

void square_vector2d::draw()
{
	this->basic_shape_draw(GL_TRIANGLES, 6);
}

template class vertice_handler<shapes::line>;
template class vertice_handler<shapes::square>;

template<shapes shape>
vertice_handler<shape>::~vertice_handler()
{
	if (!this->vertices.empty()) {
		glDeleteVertexArrays(1, &vertice_buffer.VAO);
		glDeleteVertexArrays(1, &color_buffer.VAO);
		glDeleteVertexArrays(1, &shape_buffer.VAO);
		glDeleteBuffers(1, &vertice_buffer.VBO);
		glDeleteBuffers(1, &color_buffer.VBO);
		glDeleteBuffers(1, &shape_buffer.VBO);
	}
}

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
		camera = (Camera*)glfwGetWindowUserPointer(window);
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

//Particles::Particles(uint64_t particles_amount, vec2 particle_size, vec2 particle_speed, float life_time, Color begin, Color end) :
//	particles(particles_amount, -particle_size,
//		vec2(particle_size), begin), particle(), particleIndex(particles_amount - 1), begin(begin), end(end), life_time(life_time) {
//	for (int i = 0; i < particles_amount; i++) {
//		particle.push_back({ life_time, Timer(chrono_t::now(), 0), 0, particle_speed * vec2(cosf(i), sinf(i)) });
//	}
//}

void Particles::add(vec2 position) {
	static Timer click_timer = {
		chrono_t::now(),
		particles_per_second ? uint64_t(1000 / particles_per_second) : uint64_t(1e+10)
	};
	if (particle[particleIndex].time.finished() && click_timer.finished()) {
		particles.set_position(particleIndex, position - particles.get_size(0) / 2);
		particle[particleIndex].time.start(life_time);
		particle[particleIndex].display = true;
		if (--particleIndex <= -1) {
			particleIndex = particles.size() - 1;
		}
		click_timer.restart();
	}
}

void Particles::draw() {
	for (int i = 0; i < particles.size(); i++) {
		if (!particle[i].display) {
			continue;
		}
		if (particle[i].time.finished()) {
			particles.set_position(i, vec2(-particles.get_size(0)), true);
			particle[i].display = false;
			particle[i].time.start(life_time);
			continue;
		}
		Color color = particles.get_color(i);
		const float passed_time = particle[i].time.elapsed();
		float life_time = particle[i].life_time;

		color.r = ((end.r - begin.r) / life_time) * passed_time + begin.r;
		color.g = ((end.g - begin.g) / life_time) * passed_time + begin.g;
		color.b = ((end.b - begin.b) / life_time) * passed_time + begin.b;
		color.a = (particle[i].life_time - passed_time / 1.f) / particle[i].life_time;
		particles.set_color(i, color, true);
		particles.set_position(i, particles.get_position(i) + particle[i].particle_speed * delta_time, true);
	}
	particles.free_queue();
	particles.draw();
}

button::button(const vec2& position, const vec2& size, const Color& color, std::function<void()> lambda) :
	square_vector2d(position, size, color), count(1) {
	callbacks.push_back(lambda);
}

void button::add(const vec2& _Position, vec2 _Length, Color color, std::function<void()> lambda, bool queue) {
	this->push_back(_Position, _Length, color, queue);
	callbacks.push_back(lambda);
	count++;
}

void button::add(const button& button) {
	*this = button;
}

void button::button_press_callback(uint64_t index) {
	if (inside(index)) {
		if (callbacks[index]) {
			callbacks[index]();
		}
	}
}

bool button::inside(uint64_t index) const {
	return cursor_position.x >= get_position(index).x &&
		cursor_position.x <= get_position(index).x + get_size(index).x &&
		cursor_position.y >= get_position(index).y &&
		cursor_position.y <= get_position(index).y + get_size(index).y;
}

uint64_t button::amount() const {
	return count;
}

button_single::button_single(const vec2& position, const vec2& size, const Color& color, std::function<void()> lambda, bool queue)
	: Square(position, size, color, queue), callback(lambda) { }

void button_single::button_press_callback(uint64_t index) {
	if (inside()) {
		if (callback) {
			callback();
		}
	}
}

bool button_single::inside() const {
	return cursor_position.x >= get_position().x &&
		cursor_position.x <= get_position().x + get_size().x &&
		cursor_position.y >= get_position().y &&
		cursor_position.y <= get_position().y + get_size().y;
}

Box::Box(const vec2& position, const vec2& size, const Color& color) :
	box_lines(
		line_vector2d(
			mat2x2(
				position,
				vec2(position.x + size.x, position.y)
			), color
		)
	) {
	box_lines.push_back(
		mat2x2(
			position,
			vec2(position.x, position.y + size.y)
		),
		color
	);
	box_lines.push_back(
		mat2x2(
			position + size - vec2(0, 1),
			vec2(position.x - 1, position.y + size.y - 1)
		),
		color
	);
	box_lines.push_back(
		mat2x2(position + size,
			vec2(position.x + size.x, position.y)
		),
		color
	);
	this->size.push_back(size);
}

void Box::set_position(uint64_t index, const vec2& position) {
	box_lines.set_position(
		index,
		mat2x2(
			position,
			vec2(position.x + size[index].x, position.y)
		), true
	);
	box_lines.set_position(
		index + 1,
		mat2x2(
			position,
			vec2(position.x, position.y + size[index].y)
		), true
	);
	box_lines.set_position(
		index + 2,
		mat2x2(
			position + size[index],
			vec2(position.x, position.y + size[index].y)
		), true
	);
	box_lines.set_position(
		index + 3,
		mat2x2(position + size[index],
			vec2(position.x + size[index].x, position.y)
		)
	);
	box_lines.free_queue();
}

void Box::draw() const {
	box_lines.draw();
}

#ifdef FT_FREETYPE_H

TextRenderer::TextRenderer() : shader(Shader("GLSL/text.vs", "GLSL/text.frag")) {
	shader.Use();
	mat4 projection = Ortho(0, window_size.x, window_size.y, 0);
	glUniformMatrix4fv(glGetUniformLocation(shader.ID, "projection"), 1, GL_FALSE, &projection[0][0]);
	// FreeType
	FT_Library ft;
	// All functions return a value different than 0 whenever an error occurred
	if (FT_Init_FreeType(&ft))
		std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;

	// Load font as face
	FT_Face face;
	if (FT_New_Face(ft, "fonts/calibri.ttf", 0, &face))
		std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;

	// Set size to load glyphs as
	FT_Set_Pixel_Sizes(face, 0, 48);

	// Disable byte-alignment restriction
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// 246 = � in unicode
	for (GLubyte c = 0; c < 247; c++)
	{
		// Load character glyph 
		if (FT_Load_Char(face, c, FT_LOAD_RENDER))
		{
			std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
			continue;
		}
		// Generate texture
		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RED,
			face->glyph->bitmap.width,
			face->glyph->bitmap.rows,
			0,
			GL_RED,
			GL_UNSIGNED_BYTE,
			face->glyph->bitmap.buffer
		);
		// Set texture options
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// Now store character for later use
		Character character = {
			texture,
			_vec2<int>(face->glyph->bitmap.width, face->glyph->bitmap.rows),
			_vec2<int>(face->glyph->bitmap_left, face->glyph->bitmap_top),
			(unsigned int)face->glyph->advance.x
		};
		Characters.insert(std::pair<GLchar, Character>(c, character));
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	// Destroy FreeType once we're finished
	FT_Done_Face(face);
	FT_Done_FreeType(ft);


	// Configure VAO/VBO for texture quads
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

vec2 TextRenderer::get_size(std::string text, float scale, bool include_endl) {
	vec2 end_position;

	std::string::const_iterator c;

	float length = 0;

	for (c = text.begin(); c != text.end(); c++)
	{
		if (*c == '\n' && !include_endl) {
			continue;
		}
		else if (*c == '\n') {
			end_position.y += text_gap;
		}
		Character ch = Characters[*c];

		GLfloat w = ch.Size.x * scale;
		GLfloat h = ch.Size.y * scale;

		GLfloat xpos = ch.Bearing.x * scale + w;
		GLfloat ypos = (ch.Size.y - ch.Bearing.y) * scale - h;

		end_position.x += (ch.Advance >> 6) * scale;
		length = end_position.x + ch.Bearing.x * scale;
		if (ch.Size.y * scale > end_position.y) {
			end_position.y = ch.Size.y * scale;
		}
	}

	//float end = 0;
	return vec2(length, end_position.y);
}

void TextRenderer::render(const std::string& text, vec2 position, float scale, const Color& color) {
	if (text.empty()) {
		return;
	}
	shader.Use();
	mat4 projection = Ortho(0, window_size.x, window_size.y, 0);
	glUniformMatrix4fv(glGetUniformLocation(shader.ID, "projection"), 1, GL_FALSE, &projection[0][0]);
	glUniform4f(glGetUniformLocation(shader.ID, "textColor"), color.r, color.g, color.b, color.a);
	glActiveTexture(GL_TEXTURE0);
	glBindVertexArray(VAO);

	std::string::const_iterator c;

	float originalX = position.x;

	for (c = text.begin(); c != text.end(); c++)
	{
		Character ch = Characters[*c];

		GLfloat w = ch.Size.x * scale;
		GLfloat h = ch.Size.y * scale;

		/*if (position.y - (ch.Size.y - ch.Bearing.y) * scale + h < title_bar_height) {
			continue;
		}
		else if (position.y - (ch.Size.y - ch.Bearing.y) * scale + h > window_size.y) {
			continue;
		}*/
		if (*c == '\n') {
			position.x = originalX;
			position.y += text_gap;
			continue;
		}
		else if (*c == '\b') {
			position.x = originalX;
			position.y -= text_gap;
			continue;
		}

		GLfloat xpos = position.x + ch.Bearing.x * scale;
		GLfloat ypos = position.y + (ch.Size.y - ch.Bearing.y) * scale;
		// Update VBO for each character
		GLfloat vertices[6][4] = {
			{ xpos,     ypos - h,   0.0, 0.0 },
			{ xpos,     ypos,       0.0, 1.0 },
			{ xpos + w, ypos,       1.0, 1.0 },

			{ xpos,     ypos - h,   0.0, 0.0 },
			{ xpos + w, ypos,       1.0, 1.0 },
			{ xpos + w, ypos - h,   1.0, 0.0 }
		};
		// Render glyph texture over quad
		glBindTexture(GL_TEXTURE_2D, ch.TextureID);
		// Update content of VBO memory
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// Render quad
		glDrawArrays(GL_TRIANGLES, 0, 6);
		// Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
		position.x += (ch.Advance >> 6) * scale; // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
	}
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

fan_gui::text_box::text_box() : button_single(vec2(), vec2(), Color()) {}

fan_gui::text_box::text_box(TextRenderer* renderer, std::string text, const vec2& position, const Color& color)
	: text(text), button_single(position, vec2(), color, [] {}, true) {
	this->renderer = renderer;
	int offset = 0;
	int counter = 0;
	float size = 0;
	for (int i = 0; i < this->text.size(); i++) {
		size = this->renderer->get_size(this->text.substr(offset, i - offset), font_size).x;
		if (chat_box_max_width <= size + this->renderer->get_size(this->text.substr(i, 1), font_size).x) {
			this->text.insert(this->text.begin() + i, '\n');
			offset = i;
			counter++;
		}
	}
	if (counter) {
		set_size(
			vec2(
				floor(chat_box_max_width + gap_between_text_and_box.x),
				chat_box_height + ceil(text_gap * counter)
			)
		);
	}
	else {
		set_size(
			vec2(
				ceil(size + gap_between_text_and_box.x + gap_between_text_and_box.x),
				chat_box_height
			)
		);
	}
}

std::string fan_gui::text_box::get_text() const {
	return text;
}

void fan_gui::text_box::set_text(std::string new_text, std::deque<std::string>& messages, uint64_t at) {
	auto found = messages[at].find_last_of("-");
	messages[at] = new_text += messages[at].substr(found - 1);
	*this = text_box::text_box(renderer, messages[at], position, color);
}

vec2 fan_gui::text_box::get_position() const {
	return position;
}

void fan_gui::text_box::set_position(const vec2& position, bool queue) {
	this->position = position;
	Square::set_position(this->position, queue);
}

void fan_gui::text_box::draw() {
	Square::draw();

	renderer->render(
		text,
		position + vec2(gap_between_text_and_box.x / 2, ceil(renderer->get_size(text, font_size).y) * 1.8),
		font_size,
		white_color
	);
}

std::string fan_gui::text_box::get_finished_string(TextRenderer* renderer, std::string text) {
	int offset = 0;
	float size = 0;
	for (int i = 0; i < text.size(); i++) {
		size = renderer->get_size(text.substr(offset, i - offset), font_size).x;
		if (chat_box_max_width <= size) {
			text.insert(text.begin() + i, '\n');
			offset = i;
		}
	}
	return text;
}

vec2 fan_gui::text_box::get_size_all(TextRenderer* renderer, std::string text) {
	int offset = 0;
	int counter = 0;
	float size = 0;
	for (int i = 0; i < text.size(); i++) {
		size = renderer->get_size(text.substr(offset, i - offset), font_size).x;
		if (chat_box_max_width <= size) {
			offset = i;
			counter++;
		}
	}
	if (counter) {
		return(
			vec2(
				floor(chat_box_max_width + gap_between_text_and_box.x),
				chat_box_height + ceil(text_gap * counter)
			)
			);
	}
	else {
		return(
			vec2(
				ceil(size + gap_between_text_and_box.x + gap_between_text_and_box.x),
				chat_box_height
			)
			);
	}
}

constexpr auto how_many_boxes_visible = 15;

void fan_gui::text_box::refresh(
	std::vector<text_box>& chat_boxes,
	const std::deque<std::string>& messages,
	TextRenderer* tr,
	text_box_side side,
	int offset
) {
	chat_boxes.erase(chat_boxes.begin(), chat_boxes.end());
	float y_pos = window_size.y - type_box_height;
	for (int i = 0; i < messages.size(); i++) {
		chat_boxes.push_back(
			text_box(
				tr,
				messages[i],
				vec2(),
				select_color
			)
		);

		y_pos -= chat_boxes[i].get_size().y + chat_boxes_gap;
		if (y_pos + chat_boxes[i].get_size().y <= title_bar_height - offset) {
			chat_boxes.erase(chat_boxes.begin() + i);
			break;
		}
		chat_boxes[i].set_position(
			vec2(
				side == text_box_side::LEFT ?
				user_box_size.x
				+ chat_boxes_gap / 2 :
				window_size.x - chat_boxes_gap / 2
				- chat_boxes[i].get_size().x,
				y_pos + (offset != -1 ? offset : 0)
			),
			true
		);
	}
	if (!messages.empty()) {
		break_queue();
	}
}

void fan_gui::text_box::refresh(
	std::vector<text_box>& chat_boxes,
	const std::deque<std::string>& messages,
	std::vector<text_box>& second_boxes,
	const std::deque<std::string>& second_messages,
	TextRenderer* tr,
	int offset
)
{
	refresh(chat_boxes, messages, tr, text_box_side::RIGHT, offset);
	refresh(second_boxes, second_messages, tr, text_box_side::LEFT, offset);
}

fan_gui::Titlebar::Titlebar() {
	exit_cross.push_back(
		mat2x2(
			vec2(buttons.get_position(eti(e_button::exit)).x +
				title_bar_shapes_size,
				title_bar_button_size.y - title_bar_shapes_size
			),
			vec2(buttons.get_position(eti(e_button::exit)).x +
				title_bar_button_size.x - title_bar_shapes_size,
				buttons.get_position(eti(e_button::exit)).y +
				title_bar_shapes_size
			)
		)
	);
	buttons.add( // close
		vec2(window_size.x - title_bar_button_size.x, 0),
		title_bar_button_size,
		title_bar_color,
		[] {
			glfwSetWindowShouldClose(window, true);
		}
	);
	buttons.add( // maximize
		vec2(window_size.x - title_bar_button_size.x * 2, 0),
		title_bar_button_size,
		title_bar_color,
		[&] {
			if (m_bMaximized) {
				glfwRestoreWindow(window);
			}
			else {
				glfwMaximizeWindow(window);
			}
			m_bMaximized = !m_bMaximized;
		}
	);
	buttons.add( // minimize
		vec2(window_size.x - title_bar_button_size.x * 3, 0),
		title_bar_button_size,
		title_bar_color,
		[] {
			glfwIconifyWindow(window);
		}
	);
}

void fan_gui::Titlebar::cursor_update() {
	if (buttons.inside(eti(e_button::exit))) {
		if (buttons.get_color(eti(e_button::exit)) != Color(1, 0, 0)) {
			buttons.set_color(eti(e_button::exit), Color(1, 0, 0));
		}
	}
	else {
		if (buttons.get_color(eti(e_button::exit)) != title_bar_color) {
			buttons.set_color(eti(e_button::exit), title_bar_color);
			glfwSwapBuffers(window);
		}
	}
	if (buttons.inside(eti(e_button::minimize))) {
		if (buttons.get_color(eti(e_button::minimize)) != highlight_color) {
			buttons.set_color(eti(e_button::minimize), highlight_color);
		}
	}
	else {
		if (buttons.get_color(eti(e_button::minimize)) != title_bar_color) {
			buttons.set_color(eti(e_button::minimize), title_bar_color);
		}
	}
	if (buttons.inside(eti(e_button::maximize))) {
		if (buttons.get_color(eti(e_button::maximize)) != highlight_color) {
			buttons.set_color(eti(e_button::maximize), highlight_color);
		}
	}
	else {
		if (buttons.get_color(eti(e_button::maximize)) != title_bar_color) {
			buttons.set_color(eti(e_button::maximize), title_bar_color);
		}
	}

	if (allow_move()) {
#ifdef FAN_WINDOWS
		vec2 new_pos(cursor_screen_position() - old_cursor_offset);
#else
		vec2 new_pos(cursor_position - old_cursor_offset);
#endif
		glfwSetWindowPos(window, new_pos.x, new_pos.y);
	}
}

void fan_gui::Titlebar::resize_update() {
	buttons.set_size(eti(e_button::title_bar), vec2(window_size.x, title_bar_height));

	buttons.set_position(eti(e_button::exit), vec2(window_size.x - title_bar_button_size.x, 0));
	exit_cross.set_position(
		0,
		mat2x2(
			buttons.get_position(eti(e_button::exit)) + title_bar_shapes_size,
			buttons.get_position(eti(e_button::exit)) + title_bar_button_size - title_bar_shapes_size
		)
	);
	exit_cross.set_position(
		1,
		mat2x2(
			vec2(buttons.get_position(eti(e_button::exit)).x + title_bar_shapes_size,
				title_bar_button_size.y - title_bar_shapes_size
			),
			vec2(buttons.get_position(eti(e_button::exit)).x + title_bar_button_size.x - title_bar_shapes_size,
				buttons.get_position(eti(e_button::exit)).y + title_bar_shapes_size
			)
		)
	);
	buttons.set_position(
		eti(e_button::minimize),
		vec2(window_size.x - title_bar_button_size.x * 3, 0)
	);
	minimize_line.set_position(
		0,
		mat2x2(
			vec2(buttons.get_position(eti(e_button::minimize)).x +
				title_bar_shapes_size,
				title_bar_button_size.y / 2
			),
			vec2(buttons.get_position(eti(e_button::minimize)).x +
				title_bar_button_size.x - title_bar_shapes_size,
				title_bar_button_size.y / 2
			)
		)
	);
	buttons.set_position(
		eti(e_button::maximize),
		vec2(window_size.x - title_bar_button_size.x * 2, 0)
	);
	maximize_box.set_position(
		0,
		buttons.get_position(eti(e_button::maximize)) +
		exit_cross.get_size() / 2
	);
}

vec2 fan_gui::Titlebar::get_position(e_button button) {
	return buttons.get_position(eti(button));
}

void fan_gui::Titlebar::move_window() {
	if (buttons.inside(eti(e_button::title_bar)) &&
		!buttons.inside(eti(e_button::minimize)) &&
		!buttons.inside(eti(e_button::maximize)) &&
		!buttons.inside(eti(e_button::exit))) {
		_vec2<int> l_window_position;
		glfwGetWindowPos(window, &l_window_position.x, &l_window_position.y);
#ifdef FAN_WINDOWS
		old_cursor_offset = cursor_screen_position() - l_window_position;
#else
		old_cursor_offset = cursor_position - l_window_position;
#endif
		move_window(true);
	}
}

void fan_gui::Titlebar::callbacks() {
	for (int i = buttons.size(); i--; ) {
		buttons.button_press_callback(i);
	}
}

void fan_gui::Titlebar::move_window(bool state) {
	m_bAllowMoving = state;
}

bool fan_gui::Titlebar::allow_move() const {
	return m_bAllowMoving;
}

void fan_gui::Titlebar::draw() {
	buttons.draw();
	exit_cross.draw();
	minimize_line.draw();
	maximize_box.draw();
}

fan_gui::Users::Users(const std::string& username, message_t chat)
	: user_divider(
		mat2x2(
			vec2(user_divider_x, title_bar_height),
			vec2(user_divider_x, window_size.y)
		),
		Color()
	), user_boxes(vec2(0, user_box_size.y), user_box_size, user_box_color),
	background(vec2(0, title_bar_height),
		vec2(user_box_size.x, window_size.y - type_box_height + title_bar_height), user_box_color) {
	usernames.push_back(username);
	background.push_back(
		vec2(user_box_size.x, title_bar_height),
		vec2(
			window_size.x - user_box_size.x,
			user_box_size.y - title_bar_height
		),
		user_box_color
	);
	chat[username].push_back(std::string());
}

void fan_gui::Users::add(const std::string& username, message_t chat) {
	user_boxes.add(
		user_boxes.get_position(user_boxes.size() - 1) +
		vec2(0, user_box_size.y),
		user_box_size,
		user_box_color
	);
	usernames.push_back(username);
	chat[username].push_back(std::string());
}

void fan_gui::Users::draw() {
	background.draw(
		0,
		selected() ?
		background.size() : background.size() - 1
	);
	user_boxes.draw();
	user_divider.draw();
}

void fan_gui::Users::color_callback() {
	bool selected = false;
	Color color = lighter_color(user_box_color, 0.1);
	for (int i = 0; i < user_boxes.size(); i++) {
		if (user_boxes.inside(i) && !selected) {
			Color temp_color = user_boxes.get_color(i);
			if (temp_color != color && temp_color != select_color) {
				user_boxes.set_color(i, color);
				selected = true;
			}
		}
		else {
			Color temp_color = user_boxes.get_color(i);
			if (temp_color != user_box_color && temp_color != select_color) {
				user_boxes.set_color(i, user_box_color);
			}
		}
	}
}

void fan_gui::Users::resize_update() {
	background.set_position(
		0,
		vec2(0, title_bar_height)
	);
	background.set_size(
		0,
		vec2(user_box_size.x, window_size.y - type_box_height + title_bar_height)
	);
	background.set_position(
		1,
		vec2(user_box_size.x, title_bar_height)
	);
	background.set_size(
		1,
		vec2(window_size.x - user_box_size.x,
			user_box_size.y - title_bar_height)
	);
	user_divider.set_position(
		mat2x2(
			vec2(user_divider_x, title_bar_height),
			vec2(user_divider_x, window_size.y)
		)
	);
}

void fan_gui::Users::render_text(TextRenderer& renderer) {
	for (int i = 0; i < usernames.size(); i++) {
		renderer.render(
			usernames[i],
			user_boxes.get_position(i) + vec2(20, 50),
			font_size,
			white_color
		);
	}
	renderer.render(current_user,
		vec2(
			user_box_size.x + 20,
			title_bar_height +
			renderer.get_size(current_user, font_size).y + 20
		),
		font_size,
		white_color
	);
}

void fan_gui::Users::select() {
	for (int i = 0; i < user_boxes.size(); i++) {
		if (user_boxes.inside(i)) {
			current_user = usernames[i];
			user_boxes.set_color(get_user_i(), user_box_color);
			high_light(i);
			break;
		}
	}
}

void fan_gui::Users::high_light(int i) {
	current_user_i = i;
	user_boxes.set_color(i, select_color);
}

int fan_gui::Users::get_user_i() const {
	return current_user_i;
}

bool fan_gui::Users::selected() const {
	return !current_user.empty();
}

std::string fan_gui::Users::get_username(int i) {
	return usernames[i];
}

void fan_gui::Users::reset() {
	current_user = std::string();
	user_boxes.set_color(get_user_i(), user_box_color);
}

std::string fan_gui::Users::get_user() const {
	return current_user;
}

uint64_t fan_gui::Users::size() const {
	return usernames.size();
}

#endif

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

int load_texture(const std::string_view path, const std::string& directory) {
	std::string file_name = std::string(directory + '/' + path.data());
	GLuint texture_id;
	glGenTextures(1, &texture_id);

	int width, height;

	stbi_set_flip_vertically_on_load(false);
	unsigned char* image = SOIL_load_image(file_name.c_str(), &width, &height, 0, SOIL_LOAD_RGBA);

	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
	glGenerateMipmap(GL_TEXTURE_2D);
	SOIL_free_image_data(image);
	glBindTexture(GL_TEXTURE_2D, 0);

	return texture_id;
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

vec3 basic_3d::get_size(uint64_t i) const {
	vec3 size;
	size.x = object_matrix[i][0][0];
	size.y = object_matrix[i][1][1];
	size.z = object_matrix[i][2][2];
	return size;
}

vec3 basic_3d::get_position(uint64_t i) const {
	vec3 position;
	position.x = object_matrix[i][3][0];
	position.y = object_matrix[i][3][1];
	position.z = object_matrix[i][3][2];
	return position;
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

void move_camera(bool noclip, float movement_speed) {
	constexpr double accel = -40;

	constexpr double jump_force = 100;
	if (!noclip) {
		camera3d.velocity.x /= camera3d.friction * delta_time + 1;
		camera3d.velocity.z /= camera3d.friction * delta_time + 1;
	}
	else {
		camera3d.velocity /= camera3d.friction * delta_time + 1;
	}
	static constexpr auto magic_number = 0.001;
	if (camera3d.velocity.x < magic_number && camera3d.velocity.x > -magic_number) {
		camera3d.velocity.x = 0;
	}
	if (camera3d.velocity.y < magic_number && camera3d.velocity.y > -magic_number) {
		camera3d.velocity.y = 0;
	} 
	if (camera3d.velocity.z < magic_number && camera3d.velocity.z > -magic_number) {
		camera3d.velocity.z = 0;
	}
	if (glfwGetKey(window, GLFW_KEY_W)) {
		const vec2 direction(DirectionVector(Radians(camera3d.yaw)));
		camera3d.velocity.x += direction.x * (movement_speed * delta_time);
		camera3d.velocity.z += direction.y * (movement_speed * delta_time);
	}
	if (glfwGetKey(window, GLFW_KEY_S)) {
		const vec2 direction(DirectionVector(Radians(camera3d.yaw)));
		camera3d.velocity.x -= direction.x * (movement_speed * delta_time);
		camera3d.velocity.z -= direction.y * (movement_speed * delta_time);
	}
	if (glfwGetKey(window, GLFW_KEY_A)) {
		camera3d.velocity -= camera3d.right * (movement_speed * delta_time);
	}
	if (glfwGetKey(window, GLFW_KEY_D)) {
		camera3d.velocity += camera3d.right * (movement_speed * delta_time);
	}
	if (!noclip) {
		if (glfwGetKey(window, GLFW_KEY_SPACE)) {
			camera3d.velocity.y += jump_force * delta_time;
		}
		camera3d.velocity.y += accel * delta_time;
	}
	else {
		if (glfwGetKey(window, GLFW_KEY_SPACE)) {
			camera3d.velocity.y += movement_speed * delta_time;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) {
			camera3d.velocity.y -= movement_speed * delta_time;
		}
	}
	camera3d.position += camera3d.velocity * delta_time;
	camera3d.updateCameraVectors();
}

void rotate_camera() {
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

std::initializer_list<e_cube> e_cube_loop = {
	e_cube::left,
	e_cube::right,
	e_cube::front,
	e_cube::back,
	e_cube::down,
	e_cube::up
};

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

static Alloc<size_t> ind;

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