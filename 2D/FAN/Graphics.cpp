#include <FAN/Graphics.hpp>
#include <functional>
#include <numeric>


float delta_time = 0;
GLFWwindow* window;

void GetFps(bool print) {
	static int fps = 0;
	static double start = glfwGetTime();
	float currentFrame = glfwGetTime();
	static float lastFrame = 0;
	delta_time = currentFrame - lastFrame;
	lastFrame = currentFrame;
	if ((glfwGetTime() - start) > 1.0) {
		if (print) {
			printf("%d\n", fps);
		}
		fps = 0;
		start = glfwGetTime();
	}
	fps++;
}

Texture::Texture() : texture(0), width(0), height(0), VBO(0), VAO(0), EBO(0) { }

_vec2<int> window_position() {
	_vec2<int> position;
	glfwGetWindowPos(window, &position.x, &position.y);
	return position;
}

unsigned char* LoadBMP(const char* path, Texture& texture) {
	FILE* file = fopen(path, "rb");
	if (!file) {
		printf("wrong path %s", path);
		exit(1);
	}
	fseek(file, 0, SEEK_END);
	uint64_t size = ftell(file);
	fseek(file, 0, SEEK_SET);
	unsigned char* data = (unsigned char*)malloc(size);
	if (data) {
		fread(data, 1, size, file);
	}
	fclose(file);

	uint32_t pixelOffset = *(uint32_t*)(data + BMP_Offsets::PIXELDATA);
	texture.width = *(uint32_t*)(data + BMP_Offsets::WIDTH);
	texture.height = *(uint32_t*)(data + BMP_Offsets::HEIGHT);

	return data + pixelOffset;
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

matrix<4, 4> Camera::GetViewMatrix(matrix<4, 4> m) {
	return m * LookAt(this->position, Round(this->position + (this->front)), Round(this->up));
}

matrix<4, 4> Camera::GetViewMatrix() {
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

Camera2D::Camera2D(vec3 position, vec3 up, float yaw, float pitch) : front(vec3(0.0f, 0.0f, -1.0f)) {
	this->position = position;
	this->worldUp = up;
	this->yaw = yaw;
	this->pitch = pitch;
	this->updateCameraVectors();
}

matrix<4, 4> Camera2D::GetViewMatrix(matrix<4, 4> m) {
	return m * LookAt(this->position, Round(this->position + (this->front)), Round(this->up));
}

matrix<4, 4> Camera2D::GetViewMatrix() {
	return LookAt(this->position, (this->position + (this->front)), (this->up));
}

vec3 Camera2D::get_position() const {
	return this->position;
}

void Camera2D::set_position(const vec3& position) {
	this->position = position;
}

void Camera2D::updateCameraVectors() {
	front.x = std::cos(Radians(this->yaw)) * std::cos(Radians(this->pitch));
	front.y = std::sin(Radians(this->pitch));
	front.z = std::sin(Radians(this->yaw)) * std::cos(Radians(this->pitch));
	this->front = Normalize(front);
	this->right = Normalize(Cross(this->front, this->worldUp));
	this->up = Normalize(Cross(this->right, this->front));
}

template class vertice_handler<shapes::line>;
template class vertice_handler<shapes::square>;

template<shapes shape>
vertice_handler<shape>::~vertice_handler() {
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
	shader.Use();
	matrix<4, 4> view(1);
	matrix<4, 4> projection(1);

	view = camera->GetViewMatrix(Translate(
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

DefaultShapeVector::~DefaultShapeVector() {
	glDeleteVertexArrays(1, &_VerticeBuffer.VAO);
	glDeleteVertexArrays(1, &_ColorBuffer.VAO);
	glDeleteVertexArrays(1, &_ShapeBuffer.VAO);
	glDeleteBuffers(1, &_VerticeBuffer.VBO);
	glDeleteBuffers(1, &_ColorBuffer.VBO);
	glDeleteBuffers(1, &_ShapeBuffer.VBO);
}

Color DefaultShapeVector::get_color(uint64_t _Index) const {
	return Color(
		_Colors[_Index * COLORSIZE * (_PointSize / 2)],
		_Colors[_Index * COLORSIZE * (_PointSize / 2) + 1],
		_Colors[_Index * COLORSIZE * (_PointSize / 2) + 2],
		_Colors[_Index * COLORSIZE * (_PointSize / 2) + 3]
	);
}

auto& DefaultShapeVector::get_color_ptr() const {
	return _Colors;
}

void DefaultShapeVector::set_color(uint64_t _Index, const Color& color, bool queue) {
	for (int i = 0; i < COLORSIZE * (_PointSize / 2); i++) {
		_Colors[_Index * (COLORSIZE * (_PointSize / 2)) + i] = color[i % 4];
	}
	if (!queue) {
		write(false, true);
	}
}

void DefaultShapeVector::draw(uint64_t first, uint64_t last) const {
	if (_Vertices.empty()) {
		return;
	}
	_Shader.Use();
	matrix<4, 4> view(1);
	matrix<4, 4> projection(1);

	view = _Camera->GetViewMatrix(Translate(view, vec3(window_size.x / 2, window_size.y / 2, -700.0f)));
	projection = Ortho(window_size.x / 2, window_size.x + window_size.x * 0.5f, window_size.y + window_size.y * 0.5f, window_size.y / 2.f, 0.1f, 1000.0f);

	static int projLoc = glGetUniformLocation(_Shader.ID, "projection");
	static int viewLoc = glGetUniformLocation(_Shader.ID, "view");
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
	glBindVertexArray(_ShapeBuffer.VAO);
	glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glDrawArrays(_Mode, first * (_PointSize / 2), !last ? _Points : last * (_PointSize / 2));
	glBindVertexArray(0);
}

void DefaultShapeVector::break_queue(bool vertices, bool color) {
	write(vertices, color);
}

void DefaultShapeVector::init() {
	this->_Camera = (Camera*)glfwGetWindowUserPointer(window);
	this->_Shader = Shader("GLSL/shapes.vs", "GLSL/shapes.frag");
	glGenBuffers(1, &_VerticeBuffer.VBO);
	glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.size(), _Vertices.data(), GL_STATIC_DRAW);
	glGenBuffers(1, &_ColorBuffer.VBO);
	glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Colors[0]) * _Colors.size(), _Colors.data(), GL_STATIC_DRAW);
	glGenVertexArrays(1, &_ShapeBuffer.VAO);
	glBindVertexArray(_ShapeBuffer.VAO);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void DefaultShapeVector::write(bool _EditVertices, bool _EditColor) {
	if (_EditVertices) {
		glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.size(), _Vertices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	if (_EditColor) {
		glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(_Colors[0]) * _Colors.size(), _Colors.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

LineVector::LineVector() {
	_Mode = GL_LINES;
	_Points = 0;
	_PointSize = 2 * 2;
	init();
}

LineVector::LineVector(const mat2x2& _M, const Color& color) {
	_Mode = GL_LINES;
	_Points = 2;
	_PointSize = _Points * 2;
	_Length.push_back(vec2(_M[1] - _M[0]));
	for (int i = 0; i < 4; i++) {
		_Vertices.push_back(_M[(i & 2) >> 1][i & 1]);
	}
	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.push_back(color[i % 4]);
	}
	init();
}

LineVector::LineVector(const LineVector& line) {
	*this = line;
}

mat2x2 LineVector::get_position(uint64_t _Index) const {
	return mat2x2(
		_Vertices[_Index * _PointSize],
		_Vertices[_Index * _PointSize + 1],
		_Vertices[_Index * _PointSize + 2],
		_Vertices[_Index * _PointSize + 3]
	);
}

void LineVector::set_position(uint64_t _Index, const mat2x2& _M, bool _Queue) {
	for (int i = 0; i < 4; i++) {
		_Vertices[_Index * _PointSize + i] = _M[(i & 2) >> 1][i & 1];
	}
	if (!_Queue) {
		write(true, false);
	}
	_Length[_Index] = vec2(_M[1] - _M[0]);
}

void LineVector::push_back(const mat2x2& _M, Color _Color, bool _Queue) {
	_Length.push_back(vec2(_M[1] - _M[0]));
	for (int i = 0; i < 4; i++) {
		_Vertices.push_back(_M[(i & 2) >> 1][i & 1]);
	}
	if (_Color.r == -1) {
		if (_Colors.size() > COLORSIZE) {
			_Color = Color(_Colors[0], _Colors[1], _Colors[2], _Colors[3]);
		}
		else {
			_Color = Color(1, 1, 1, 1);
		}
		for (int i = 0; i < COLORSIZE * 2; i++) {
			_Colors.push_back(_Color[i % 4]);
		}
	}
	else {
		for (int i = 0; i < COLORSIZE * 2; i++) {
			_Colors.push_back(_Color[i % 4]);
		}
	}
	_Points += 2;
	if (!_Queue) {
		write(true, true);
	}
}

vec2 LineVector::get_length(uint64_t _Index) const {
	return _Length[_Index];
}

TriangleVector::TriangleVector() {
	_Mode = GL_TRIANGLES;
	_Points = 0;
	_PointSize = 3 * 2;
	init();
}

TriangleVector::TriangleVector(const vec2& _Position, const vec2& _Length, const Color& _Color) {
	_Mode = GL_TRIANGLES;
	_Points = 3;
	_PointSize = _Points * 2;
	this->_Length.push_back(_Length);
	this->_Position.push_back(_Position);
	_Vertices.push_back(_Position.x - (_Length.x / 2));
	_Vertices.push_back(_Position.y + (_Length.y / 2));
	_Vertices.push_back(_Position.x + (_Length.x / 2));
	_Vertices.push_back(_Position.y + (_Length.y / 2));
	_Vertices.push_back(_Position.x);
	_Vertices.push_back(_Position.y - (_Length.y / 2));
	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.push_back(_Color[i % 4]);
	}
	init();
}

void TriangleVector::set_position(uint64_t _Index, const vec2& _Position) {
	_Vertices[_Index * _PointSize] = (_Position.x - (_Length[_Index].x / 2));
	_Vertices[_Index * _PointSize + 1] = (_Position.y + (_Length[_Index].y / 2));
	_Vertices[_Index * _PointSize + 2] = (_Position.x + (_Length[_Index].x / 2));
	_Vertices[_Index * _PointSize + 3] = (_Position.y + (_Length[_Index].y / 2));
	_Vertices[_Index * _PointSize + 4] = (_Position.x);
	_Vertices[_Index * _PointSize + 5] = (_Position.y - (_Length[_Index].y / 2));
	write(true, false);
}

vec2 TriangleVector::get_position(uint64_t _Index) const {
	return _Position[_Index];
}

void TriangleVector::push_back(const vec2 _Position, vec2 _Length, Color _Color) {
	if (!_Length.x) {
		_Length = this->_Length[0];
	}
	this->_Length.push_back(_Length);
	this->_Position.push_back(_Position);
	_Vertices.push_back(_Position.x - (_Length.x / 2));
	_Vertices.push_back(_Position.y + (_Length.y / 2));
	_Vertices.push_back(_Position.x + (_Length.x / 2));
	_Vertices.push_back(_Position.y + (_Length.y / 2));
	_Vertices.push_back(_Position.x);
	_Vertices.push_back(_Position.y - (_Length.y / 2));
	if (_Color.r == -1) {
		if (_Colors.size() > COLORSIZE) {
			_Color = Color(_Colors[0], _Colors[1], _Colors[2], _Colors[3]);
		}
		else {
			_Color = Color(1, 1, 1, 1);
		}
		for (int i = 0; i < COLORSIZE * 2; i++) {
			_Colors.push_back(_Color[i % 4]);
		}
	}
	else {
		for (int i = 0; i < COLORSIZE * 2; i++) {
			_Colors.push_back(_Color[i % 4]);
		}
	}
	_Points += 3;
	write(true, true);
}

SquareVector::SquareVector() {
	_Mode = GL_QUADS;
	_Points = 0;
	_PointSize = 4 * 2;
	init();
}

SquareVector::SquareVector(const vec2& _Position, const vec2& _Length, const Color& color) {
	_Mode = GL_QUADS;
	_Points = 4;
	_PointSize = _Points * 2;
	this->_Length.push_back(_Length);
	this->_Position.push_back(_Position);
	_Vertices.push_back(_Position.x);
	_Vertices.push_back(_Position.y);
	_Vertices.push_back(_Position.x + _Length.x);
	_Vertices.push_back(_Position.y);
	_Vertices.push_back(_Position.x + _Length.x);
	_Vertices.push_back(_Position.y + _Length.y);
	_Vertices.push_back(_Position.x);
	_Vertices.push_back(_Position.y + _Length.y);

	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.push_back(color[i % 4]);
	}
	init();
}

SquareVector::SquareVector(uint64_t _Reserve, const vec2& _Position, const vec2& _Length, const Color& color) : SquareVector() {
	for (int i = 0; i < _Reserve; i++) {
		push_back(_Position, _Length, color, true);
	}
	break_queue();
}

uint64_t SquareVector::amount() const {
	return _Points / 4;
}

bool SquareVector::empty() const {
	return amount();
}

void SquareVector::erase(uint64_t _Index) {
#ifdef FAN_PERFORMANCE
	for (int i = 0; i < _PointSize; i++) {
		_Vertices.erase(_Index * _PointSize);
	}
	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.erase(_Index * _PointSize);
	}
#else
	for (int i = 0; i < _PointSize; i++) {
		_Vertices.erase(_Vertices.begin() + _Index * _PointSize);
	}
	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.erase(_Colors.begin() + _Index * _PointSize);
	}
#endif
	_Points -= 4;
	write(true, true);
}

void SquareVector::erase_all(uint64_t _Index) {
#ifdef FAN_PERFORMANCE
	_Vertices.erase_all();
	_Colors.erase_all();
#else
	_Vertices.erase(_Vertices.begin(), _Vertices.end());
	_Colors.erase(_Colors.begin(), _Colors.end());
#endif
	_Points -= 4;
	write(true, true);
}

vec2 SquareVector::get_length(uint64_t _Index) const {
	return _Length[_Index];
}

mat2x4 SquareVector::get_corners(uint64_t _Index) const {
	if (_Index >= amount()) {
		return mat2x4();
	}
	uint64_t _Multiplier = _Index * _PointSize;
	return mat2x4(
		vec2(_Vertices[_Multiplier], _Vertices[_Multiplier + 1]),
		vec2(_Vertices[_Multiplier + 2], _Vertices[_Multiplier + 3]),
		vec2(_Vertices[_Multiplier + 4], _Vertices[_Multiplier + 5]),
		vec2(_Vertices[_Multiplier + 6], _Vertices[_Multiplier + 7])
	);
}

vec2 SquareVector::get_position(uint64_t _Index) const {
	if (_Index >= amount()) {
		return vec2(-1);
	}
	return vec2(_Vertices[_PointSize * _Index], _Vertices[_PointSize * _Index + 1]);
}

void SquareVector::set_position(uint64_t _Index, const vec2& _Position, bool _Queue) {
	vec2 _Distance(_Position[0] - _Vertices[_Index * _PointSize + 0], _Position[1] - _Vertices[_Index * _PointSize + 1]);
	for (int i = 0; i < _PointSize; i++) {
		_Vertices[_Index * _PointSize + i] += _Distance[i % 2];
	}
	if (!_Queue) {
		write(true, false);
	}
	this->_Position[_Index] = _Position;
}

vec2 SquareVector::get_size(uint64_t _Index) const {
	return this->_Length[_Index];
}

void SquareVector::set_size(uint64_t _Index, const vec2& _Size, bool _Queue) {
	_Vertices[_Index * _PointSize + 0] = _Position[_Index].x;
	_Vertices[_Index * _PointSize + 1] = _Position[_Index].y;
	_Vertices[_Index * _PointSize + 2] = _Position[_Index].x + _Size.x;
	_Vertices[_Index * _PointSize + 3] = _Position[_Index].y;
	_Vertices[_Index * _PointSize + 4] = _Position[_Index].x + _Size.x;
	_Vertices[_Index * _PointSize + 5] = _Position[_Index].y + _Size.y;
	_Vertices[_Index * _PointSize + 6] = _Position[_Index].x;
	_Vertices[_Index * _PointSize + 7] = _Position[_Index].y + _Size.y;
	if (!_Queue) {
		write(true, false);
	}
	this->_Length[_Index] = _Size;
}

void SquareVector::push_back(const SquareVector& square) {
	for (int i = 0; i < square.amount(); i++) {
		push_back(square.get_position(i), square.get_length(i), square.get_color(i), true);
	}
	break_queue();
}

void SquareVector::push_back(const vec2& _Position, vec2 _Length, Color color, bool _Queue) {
	if (!_Length.x && !_Position.x && !_Position.y) {
		_Length = this->_Length[0];
	}
	this->_Length.push_back(_Length);
	this->_Position.push_back(_Position);
	_Vertices.push_back(_Position.x);
	_Vertices.push_back(_Position.y);
	_Vertices.push_back(_Position.x + _Length.x);
	_Vertices.push_back(_Position.y);
	_Vertices.push_back(_Position.x + _Length.x);
	_Vertices.push_back(_Position.y + _Length.y);
	_Vertices.push_back(_Position.x);
	_Vertices.push_back(_Position.y + _Length.y);

	if (color.r == -1) {
		if (_Colors.size() > COLORSIZE) {
			color = Color(_Colors[0], _Colors[1], _Colors[2], _Colors[3]);
		}
		else {
			color = Color(1, 1, 1, 1);
		}
		for (int i = 0; i < COLORSIZE * 2; i++) {
			_Colors.push_back(color[i % 4]);
		}
	}
	else {
		for (int i = 0; i < COLORSIZE * _PointSize; i++) {
			_Colors.push_back(color[i % 4]);
		}
	}
	_Points += 4;
	if (!_Queue) {
		write(true, true);
	}
}

void SquareVector::rotate(uint64_t _Index, double _Angle, bool _Queue) {
	constexpr double offset = 3 * PI / 4;
	const vec2 position(get_position(_Index));
	const vec2 _Radius(_Length[_Index] / 2);
	const double r = Distance(get_position(_Index), get_position(_Index) + _Length[_Index] / 2);

	mat2x4 corners(
		vec2(r * cos(Radians(_Angle) + offset), r * sin(Radians(_Angle) + offset)),
		vec2(r * cos(Radians(_Angle) + offset - PI / 2.f), r * sin(Radians(_Angle) + offset - PI / 2.f)),
		vec2(r * cos(Radians(_Angle) + offset - PI), r * sin(Radians(_Angle) + offset - PI)),
		vec2(r * cos(Radians(_Angle) + offset - PI * 3.f / 2.f), r * sin(Radians(_Angle) + offset - PI * 3.f / 2.f))
	);

	for (int i = 0; i < _PointSize; i++) {
		_Vertices[_Index * _PointSize + i] = corners[(i & (_PointSize - 1)) >> 1][i & 1] + position[i & 1] + _Radius[i & 1];
	}

	if (!_Queue) {
		write(true, false);
	}
}

CircleVector::CircleVector(uint64_t _Number_Of_Points, float _Radius) {
	_Mode = GL_LINES;
	_Points = 0;
	_PointSize = _Number_Of_Points * 2;
	this->_Radius.push_back(_Radius);
	init();
}

CircleVector::CircleVector(const vec2& _Position, float _Radius, uint64_t _Number_Of_Points, const Color& _Color) {
	_Mode = GL_LINE_LOOP;
	_Points = _Number_Of_Points;
	_PointSize = _Number_Of_Points * 2;
	this->_Position.push_back(_Position);
	this->_Radius.push_back(_Radius);
	for (int i = 0; i < _Points; i++) {
		float theta = 2.0f * 3.1415926f * float(i) / float(_Points);

		float x = _Radius * cosf(theta);
		float y = _Radius * sinf(theta);

		_Vertices.push_back(_Position.x + x);
		_Vertices.push_back(_Position.y + y);
	}

	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.push_back(_Color[i % 4]);
	}
	init();
}

void CircleVector::set_radius(float _Radius, uint64_t index) {
	for (int i = 0; i < _Points * 2; i++) {
		float theta = 2.0f * 3.1415926f * float(i) / float(_Points * 2);

		if (i & 1) {
			float x = _Radius * cosf(theta);
			_Vertices[i] = (_Position[index].x + x);
		}
		else {
			float y = _Radius * sinf(theta);
			_Vertices[i] = (_Position[index].y + y);
		}
	}
	write(true, false);
}

void CircleVector::set_position(uint64_t _Index, const vec2& _Position) {
	for (int ii = 0; ii < _PointSize; ii += 2) {
		float theta = _Double_Pi * float(ii) / float(_PointSize);

		float x = _Radius[_Index] * cosf(theta);
		float y = _Radius[_Index] * sinf(theta);
		_Vertices[_Index * _PointSize + ii] = _Position.x + x;
		_Vertices[_Index * _PointSize + ii + 1] = _Position.y + y;
	}
	this->_Position[_Index] = _Position;
	write(true, false);
}

void CircleVector::push_back(vec2 _Position, float _Radius, Color _Color, bool _Queue) {
	const uint64_t _LPoints = _PointSize / 2;
	this->_Position.push_back(_Position);
	this->_Radius.push_back(_Radius);
	for (int ii = 0; ii < _Points; ii++) {
		float theta = 2.0f * 3.1415926f * float(ii) / float(_Points);

		float x = _Radius * cosf(theta);
		float y = _Radius * sinf(theta);

		_Vertices.push_back(_Position.x + x);
		_Vertices.push_back(_Position.y + y);
	}
	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.push_back(_Color[i % 4]);
	}
	_Points += _PointSize / 2;
	if (_Queue) {
		write(true, true);
	}
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

void Sprite::draw() {
	shader.Use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture.texture);
	glUniform1i(glGetUniformLocation(shader.ID, "ourTexture"), 0);
	matrix<4, 4> view(1);
	matrix<4, 4> projection(1);
	view = camera->GetViewMatrix(Translate(view, vec3(window_size.x / 2, window_size.y / 2, -700.0f)));
	projection = Ortho(window_size.x / 2, window_size.x + window_size.x * 0.5f, window_size.y + window_size.y * 0.5f, window_size.y / 2.f, 0.1f, 1000.0f);
	GLint projLoc = glGetUniformLocation(shader.ID, "projection");
	GLint viewLoc = glGetUniformLocation(shader.ID, "view");
	GLint modelLoc = glGetUniformLocation(shader.ID, "model");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
	matrix<4, 4> model(1);
	model = Translate(model, Vec2ToVec3(position));
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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, object.width, object.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, LoadBMP(path, object));
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

Particles::Particles(uint64_t particles_amount, vec2 particle_size, vec2 particle_speed, float life_time, Color begin, Color end) :
	particles(particles_amount, -particle_size,
		vec2(particle_size), begin), particle(), particleIndex(particles_amount - 1), begin(begin), end(end), life_time(life_time) {
	for (int i = 0; i < particles_amount; i++) {
		particle.push_back({ life_time, Timer(high_resolution_clock::now(), 0), 0, particle_speed * vec2(cosf(i), sinf(i)) });
	}
}

void Particles::add(vec2 position) {
	static Timer click_timer = {
		high_resolution_clock::now(),
		particles_per_second ? uint64_t(1000 / particles_per_second) : uint64_t(1e+10)
	};
	if (particle[particleIndex].time.finished() && click_timer.finished()) {
		particles.set_position(particleIndex, position - particles.get_length(0) / 2);
		particle[particleIndex].time.start(life_time);
		particle[particleIndex].display = true;
		if (--particleIndex <= -1) {
			particleIndex = particles.amount() - 1;
		}
		click_timer.restart();
	}
}

void Particles::draw() {
	for (int i = 0; i < particles.amount(); i++) {
		if (!particle[i].display) {
			continue;
		}
		if (particle[i].time.finished()) {
			particles.set_position(i, vec2(-particles.get_length(0)), true);
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
	particles.break_queue();
	particles.draw();
}

vec2 Raycast(const vec2& start, const vec2& end, const SquareVector& squares, bool map[grid_size.x][grid_size.y]) {
	const float angle = -Degrees(AimAngle(start, end));
	const bool left = angle < 90 && angle > -90;
	const bool top = angle > 0 && angle < 180;

	for (int i = start.x / block_size; left ? i < end.x / block_size : i > end.x / block_size - 1; left ? i++ : i--) {
		for (int j = start.y / block_size; top ? j > end.y / block_size - 1 : j < end.y / block_size; top ? j-- : j++) {
			if (i < 0 || j < 0 || i > grid_size.x || j > grid_size.y) {
				return vec2(-1);
			}
			if (map[i][j]) {
				const mat2x4 corners = squares.get_corners(_2d_1d(vec2(i * block_size, j * block_size)));
				bool left_right = start.x < end.x;
				bool up_down = end.y < start.y;
				vec2 point = IntersectionPoint(start, end, corners[!left_right], corners[left_right ? 3 : 2]);
				if (ray_hit(point)) { return point; }
				point = IntersectionPoint(start, end, corners[up_down ? 2 : 0], corners[up_down ? 3 : 1]);
				if (ray_hit(point)) { return point; }
			}
		}
	}
	return vec2(-1);
}

button::button(const vec2& position, const vec2& size, const Color& color, std::function<void()> lambda) :
	SquareVector(position, size, color), count(1) {
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
		cursor_position.x <= get_position(index).x + get_length(index).x &&
		cursor_position.y >= get_position(index).y &&
		cursor_position.y <= get_position(index).y + get_length(index).y;
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
		LineVector(
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
		)
	);
	box_lines.push_back(
		mat2x2(
			position + size - vec2(0, 1),
			vec2(position.x - 1, position.y + size.y - 1)
		)
	);
	box_lines.push_back(
		mat2x2(position + size,
			vec2(position.x + size.x, position.y)
		)
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
	box_lines.break_queue();
}

void Box::draw() const {
	box_lines.draw();
}

TextRenderer::TextRenderer() : shader(Shader("GLSL/text.vs", "GLSL/text.frag")) {
	shader.Use();
	matrix<4, 4> projection = Ortho(0, window_size.x, window_size.y, 0);
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

vec2 TextRenderer::get_length(std::string text, float scale, bool include_endl) {
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
	matrix<4, 4> projection = Ortho(0, window_size.x, window_size.y, 0);
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
		size = this->renderer->get_length(this->text.substr(offset, i - offset), font_size).x;
		if (chat_box_max_width <= size + this->renderer->get_length(this->text.substr(i, 1), font_size).x) {
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
		position + vec2(gap_between_text_and_box.x / 2, ceil(renderer->get_length(text, font_size).y) * 1.8),
		font_size,
		white_color
	);
}

std::string fan_gui::text_box::get_finished_string(TextRenderer* renderer, std::string text) {
	int offset = 0;
	float size = 0;
	for (int i = 0; i < text.size(); i++) {
		size = renderer->get_length(text.substr(offset, i - offset), font_size).x;
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
		size = renderer->get_length(text.substr(offset, i - offset), font_size).x;
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
		exit_cross.get_length() / 2
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
	for (int i = buttons.amount(); i--; ) {
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
		user_boxes.get_position(user_boxes.amount() - 1) +
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
		background.amount() : background.amount() - 1
	);
	user_boxes.draw();
	user_divider.draw();
}

void fan_gui::Users::color_callback() {
	bool selected = false;
	Color color = lighter_color(user_box_color, 0.1);
	for (int i = 0; i < user_boxes.amount(); i++) {
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
			renderer.get_length(current_user, font_size).y + 20
		),
		font_size,
		white_color
	);
}

void fan_gui::Users::select() {
	for (int i = 0; i < user_boxes.amount(); i++) {
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