#include <FAN/Graphics.hpp>
#include <functional>

Texture::Texture() : texture(0), width(0), height(0), VBO(0), VAO(0), EBO(0) { }

unsigned char* LoadBMP(const char* path, Texture& texture) {
	FILE* file = fopen(path, "rb");
	if (!file) {
		printf("wrong path %s", path);
		exit(1);
	}
	fseek(file, 0, SEEK_END);
	std::size_t size = ftell(file);
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

size_t _2d_1d(vec2 position) {
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

Color DefaultShape::get_color(std::size_t _Index) const {
	return Color(
		_Colors[_Index * COLORSIZE * (_PointSize / 2)    ],
		_Colors[_Index * COLORSIZE * (_PointSize / 2) + 1],
		_Colors[_Index * COLORSIZE * (_PointSize / 2) + 2],
		_Colors[_Index * COLORSIZE * (_PointSize / 2) + 3]
	);
}

auto& DefaultShape::get_color_ptr() const {
	return _Colors;
}

void DefaultShape::set_color(size_t _Index, const Color& color, bool queue) {
	for (int i = 0; i < COLORSIZE * (_PointSize / 2); i++) {
		_Colors[_Index * (COLORSIZE * (_PointSize / 2)) + i] = color[i % 4];
	}
	if (!queue) {
		write(false, true);
	}
}

void DefaultShape::draw(std::size_t first) {
	if (_Vertices.empty()) {
		return;
	}
	_Shader.Use();
	matrix<4, 4> view(1);
	matrix<4, 4> projection(1);

	view = _Camera->GetViewMatrix(Translate(view, vec3(window_size.x / 2, window_size.y / 2, -700.0f)));
	projection = Ortho(window_size.x / 2, window_size.x + window_size.x * 0.5f, window_size.y + window_size.y * 0.5f, window_size.y / 2.f, 0.1f, 1000.0f);

	int projLoc = glGetUniformLocation(_Shader.ID, "projection");
	int viewLoc = glGetUniformLocation(_Shader.ID, "view");
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
	glBindVertexArray(_ShapeBuffer.VAO);
	glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glDrawArrays(_Mode, first, _Points);
	glBindVertexArray(0);
}

void DefaultShape::break_queue() {
	write(true, true);
}

void DefaultShape::init() {
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

void DefaultShape::write(bool _EditVertices, bool _EditColor) {
	if (!_Vertices.empty() && !_Colors.empty()) {
		if (_EditVertices) {
			glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.current(), _Vertices.data(), GL_STATIC_DRAW);
		}
		if (_EditColor) {
			glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(_Colors[0]) * _Colors.current(), _Colors.data(), GL_STATIC_DRAW);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

Line::Line() {
	_Mode = GL_LINES;
	_Points = 0;
	_PointSize = 2 * 2;
	init();
}

Line::Line(const mat2x2& _M, const Color& color) {
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

mat2x2 Line::get_position(size_t _Index) const {
	return mat2x2(
		_Vertices[_Index * _PointSize], 
		_Vertices[_Index * _PointSize + 1],
		_Vertices[_Index * _PointSize + 2], 
		_Vertices[_Index * _PointSize + 3]
	);
}

void Line::set_position(size_t _Index, const mat2x2& _M, bool _Queue) {
	for (int i = 0; i < 4; i++) {
		_Vertices[_Index * _PointSize + i] = _M[(i & 2) >> 1][i & 1];
	}
	if (!_Queue) {
		write(true, false);
	}
	_Length[_Index] = vec2(_M[1] - _M[0]);
}

void Line::push_back(const mat2x2& _M, Color _Color, bool _Queue) {
	_Length.push_back(vec2(_M[1] - _M[0]));
	for (int i = 0; i < 4; i++) {
		_Vertices.push_back(_M[(i & 2) >> 1][i & 1]);
	}
	if (_Color.r == -1) {
		if (_Colors.current() > COLORSIZE) {
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

vec2 Line::get_length(size_t _Index) const {
	return _Length[_Index];
}

Triangle::Triangle() {
	_Mode = GL_TRIANGLES;
	_Points = 0;
	_PointSize = 3 * 2;
	init();
}

Triangle::Triangle(const vec2& _Position, const vec2& _Length, const Color& _Color) {
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

void Triangle::set_position(size_t _Index, const vec2& _Position) {
	_Vertices[_Index * _PointSize] = (_Position.x - (_Length[_Index].x / 2));
	_Vertices[_Index * _PointSize + 1] = (_Position.y + (_Length[_Index].y / 2));
	_Vertices[_Index * _PointSize + 2] = (_Position.x + (_Length[_Index].x / 2));
	_Vertices[_Index * _PointSize + 3] = (_Position.y + (_Length[_Index].y / 2));
	_Vertices[_Index * _PointSize + 4] = (_Position.x);
	_Vertices[_Index * _PointSize + 5] = (_Position.y - (_Length[_Index].y / 2));
	write(true, false);
}

vec2 Triangle::get_position(std::size_t _Index) const {
	return _Position[_Index];
}

void Triangle::push_back(const vec2 _Position, vec2 _Length, Color _Color) {
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
		if (_Colors.current() > COLORSIZE) {
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

Square::Square() {
	_Mode = GL_QUADS;
	_Points = 0;
	_PointSize = 4 * 2;
	init();
}

Square::Square(const vec2& _Position, const vec2& _Length, const Color& color) {
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

Square::Square(std::size_t _Reserve, const vec2& _Position, const vec2& _Length, const Color& color) : Square() {
	for (int i = 0; i < _Reserve; i++) {
		push_back(_Position, _Length, color, true);
	}
	break_queue();
}

std::size_t Square::amount() const {
	return _Points / 4;
}

void Square::erase(std::size_t _Index) {
	for (int i = 0; i < _PointSize; i++) {
		_Vertices.erase(_Index * _PointSize);
	}
	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.erase(_Index * _PointSize);
	}
	_Points -= 4;
	write(true, true);
}

vec2 Square::get_length(std::size_t _Index) const {
	return _Length[_Index];
}

mat2x4 Square::get_corners(std::size_t _Index) const {
	if (_Index >= amount()) {
		return mat2x4();
	}
	std::size_t _Multiplier = _Index * _PointSize;
	return mat2x4(
		vec2(_Vertices[_Multiplier], _Vertices[_Multiplier + 1]),
		vec2(_Vertices[_Multiplier + 2], _Vertices[_Multiplier + 3]),
		vec2(_Vertices[_Multiplier + 4], _Vertices[_Multiplier + 5]),
		vec2(_Vertices[_Multiplier + 6], _Vertices[_Multiplier + 7])
	);
}

vec2 Square::get_position(std::size_t _Index) const {
	if (_Index >= amount()) {
		return vec2(-1);
	}
	return _Position[_Index];
}

void Square::set_position(std::size_t _Index, const vec2& _Position, bool _Queue) {
	vec2 _Distance(_Position[0] - _Vertices[_Index * _PointSize + 0], _Position[1] - _Vertices[_Index * _PointSize + 1]);
	for (int i = 0; i < _PointSize; i++) {
		_Vertices[_Index * _PointSize + i] += _Distance[i % 2];
	}
	if (!_Queue) {
		write(true, false);
	}
	this->_Position[_Index] = _Position;
}

void Square::push_back(const vec2& _Position, vec2 _Length, Color color, bool _Queue) {
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
		if (_Colors.current() > COLORSIZE) {
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

void Square::rotate(std::size_t _Index, double _Angle, bool _Queue) {
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
		_Vertices[_Index * _PointSize + i] = corners[(i & 7) >> 1][i & 1] + position[i & 1] + _Radius[i & 1];
	}

	if (!_Queue) {
		write(true, false);
	}
}

Circle::Circle(std::size_t _Number_Of_Points, float _Radius) {
	_Mode = GL_LINES;
	_Points = 0;
	_PointSize = _Number_Of_Points * 2;
	this->_Radius.push_back(_Radius);
	init();
}

Circle::Circle(const vec2& _Position, float _Radius, std::size_t _Number_Of_Points, const Color& _Color) {
	_Mode = GL_TRIANGLE_FAN;
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

void Circle::set_position(std::size_t _Index, const vec2& _Position) {
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

void Circle::push_back(vec2 _Position, float _Radius, Color _Color, bool _Queue) {
	const std::size_t _LPoints = _PointSize / 2;
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
	std::copy(std::begin(vertices), std::end(vertices), _Vertices.begin());
}

void Sprite::load_image(const char* path, Texture& texture) {
	std::ifstream file(path);
	if (!file.good()) {
		printf("File path does not exist\n");
		return;
	}
	glGenVertexArrays(1, &texture.VAO);
	glGenBuffers(1, &texture.VBO);
	glBindVertexArray(texture.VAO);
	glBindBuffer(GL_ARRAY_BUFFER, texture.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.size(), _Vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);
	glBindVertexArray(0);
	glGenTextures(1, &texture.texture);
	glBindTexture(GL_TEXTURE_2D, texture.texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width, texture.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, LoadBMP(path, texture));
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

Particles::Particles(std::size_t particles_amount, vec2 particle_size, vec2 particle_speed, float life_time, Color begin, Color end) :
	particles(particles_amount, -particle_size,
		vec2(particle_size), begin), particle(), particleIndex(particles_amount - 1), begin(begin), end(end), life_time(life_time) {
	for (int i = 0; i < particles_amount; i++) {
		particle.push_back({ life_time, Timer(high_resolution_clock::now(), 0), 0, particle_speed * vec2(cosf(i), sinf(i)) });
	}
}

void Particles::add(vec2 position) {
	static Timer click_timer = {
		high_resolution_clock::now(),
		particles_per_second ? 1000 / particles_per_second : std::size_t(1e+10)
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
		const float passed_time = particle[i].time.passed();
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

Entity::Entity(const char* path, vec2 position, vec2 size, float angle, Shader shader) :
	Sprite(path, position, size, angle, shader), velocity(0) { }

void Entity::move(bool mouse) {
	velocity /= (delta_time * friction) + 1;

	if (KeyPress(GLFW_KEY_W)) velocity.y -= movement_speed * delta_time;
	if (KeyPress(GLFW_KEY_S)) velocity.y += movement_speed * delta_time;
	if (KeyPress(GLFW_KEY_A)) velocity.x -= movement_speed * delta_time;
	if (KeyPress(GLFW_KEY_D)) velocity.x += movement_speed * delta_time;

	position += (velocity * delta_time);

	if (mouse) {
		angle = AimAngle(position, cursor_position) + PI / 2;
	}
}

vec2 Raycast(const vec2& start, const vec2& end, const Square& squares, bool map[grid_size.x][grid_size.y]) {
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