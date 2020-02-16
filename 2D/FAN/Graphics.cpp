#include <FAN/Graphics.hpp>
#include <FAN/Bmp.hpp>
#include <functional>

Texture::Texture() : texture(0), width(0), height(0), VBO(0), VAO(0), EBO(0) { }

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

void DefaultShape::draw() {
	if (_Vertices.empty()) {
		return;
	}
	_Shader.Use();
	Mat4x4 view(1);
	Mat4x4 projection(1);
	view = _Camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
	projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5f, windowSize.y / 2, 0.1f, 1000.0f);
	int projLoc = glGetUniformLocation(_Shader.ID, "projection");
	int viewLoc = glGetUniformLocation(_Shader.ID, "view");
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
	glBindVertexArray(_ShapeBuffer.VAO);
	glBindBuffer(GL_ARRAY_BUFFER, _VerticeBuffer.VBO);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glDrawArrays(_Mode, 0, _Points);
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Vertices[0]) * _Vertices.current(), _Vertices.data(), GL_STATIC_DRAW);
	glGenBuffers(1, &_ColorBuffer.VBO);
	glBindBuffer(GL_ARRAY_BUFFER, _ColorBuffer.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_Colors[0]) * _Colors.current(), _Colors.data(), GL_STATIC_DRAW);
	glGenVertexArrays(1, &_ShapeBuffer.VAO);
	glBindVertexArray(_ShapeBuffer.VAO);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void DefaultShape::write(bool _EditVertices, bool _EditColor) {
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

Line::Line() {
	_Mode = GL_LINES;
	_Points = 0;
	_PointSize = 2 * 2;
	init();
}

Line::Line(const Mat2x2& _M, const Color& color) {
	_Mode = GL_LINES;
	_Points = 2;
	_PointSize = _Points * 2;
	_Length.push_back(Vec2(_M[1] - _M[0]));
	for (int i = 0; i < 4; i++) {
		_Vertices.push_back(_M[(i & 2) >> 1][i & 1]);
	}
	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.push_back(color[i % 4]);
	}
	init();
}

Mat2x2 Line::get_position(size_t _Index) const {
	return Mat2x2(
		Vec2(_Vertices[_Index * _PointSize], _Vertices[_Index * _PointSize + 1]),
		Vec2(_Vertices[_Index * _PointSize + 2], _Vertices[_Index * _PointSize + 3])
	);
}

void Line::set_position(size_t _Index, const Mat2x2& _M, bool _Queue) {
	for (int i = 0; i < 4; i++) {
		_Vertices[_Index * _PointSize + i] = _M[(i & 2) >> 1][i & 1];
	}
	if (!_Queue) {
		write(true, false);
	}
	_Length[_Index] = Vec2(_M[1] - _M[0]);
}

void Line::push_back(const Mat2x2& _M, Color _Color, bool _Queue) {
	_Length.push_back(Vec2(_M[1] - _M[0]));
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

Vec2 Line::get_length(size_t _Index) const {
	return _Length[_Index];
}

Triangle::Triangle() {
	_Mode = GL_TRIANGLES;
	_Points = 0;
	_PointSize = 3 * 2;
	init();
}

Triangle::Triangle(const Vec2& _Position, const Vec2& _Length, const Color& _Color) {
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

void Triangle::set_position(size_t _Index, const Vec2& _Position) {
	_Vertices[_Index * _PointSize] = (_Position.x - (_Length[_Index].x / 2));
	_Vertices[_Index * _PointSize + 1] = (_Position.y + (_Length[_Index].y / 2));
	_Vertices[_Index * _PointSize + 2] = (_Position.x + (_Length[_Index].x / 2));
	_Vertices[_Index * _PointSize + 3] = (_Position.y + (_Length[_Index].y / 2));
	_Vertices[_Index * _PointSize + 4] = (_Position.x);
	_Vertices[_Index * _PointSize + 5] = (_Position.y - (_Length[_Index].y / 2));
	write(true, false);
}

Vec2 Triangle::get_position(std::size_t _Index) const {
	return _Position[_Index];
}

void Triangle::push_back(const Vec2 _Position, Vec2 _Length, Color _Color) {
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

Square::Square() {
	_Mode = GL_QUADS;
	_Points = 0;
	_PointSize = 4 * 2;
	init();
}

Square::Square(const Vec2& _Position, const Vec2& _Length, const Color& color) {
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

Square::Square(std::size_t _Reserve, const Vec2& _Position, const Vec2& _Length, const Color& color) : Square() {
	for (int i = 0; i < _Reserve; i++) {
		push_back(_Position, _Length, color, true);
	}
	break_queue();
}

std::size_t Square::amount() const {
	return _Points / 4;
}

void Square::free_to_max() {
	_Position.free_to_max();
	_Length.free_to_max();
	this->_Vertices.free_to_max();
	_Colors.free_to_max();
}

void Square::erase(std::size_t _Index) {
	for (int i = 0; i < 8; i++) {
		_Vertices.erase(_Index * _PointSize);
	}
	for (int i = 0; i < COLORSIZE * _PointSize; i++) {
		_Colors.erase(_Index * _PointSize);
	}
	write(true, true);
}

Vec2 Square::get_length(std::size_t _Index) const {
	return _Length[_Index];
}

Mat2x4 Square::get_corners(std::size_t _Index) const {
	std::size_t _Multiplier = _Index * _PointSize;
	return Mat2x4(
		Vec2(_Vertices[_Multiplier], _Vertices[_Multiplier + 1]),
		Vec2(_Vertices[_Multiplier + 2], _Vertices[_Multiplier + 3]),
		Vec2(_Vertices[_Multiplier + 4], _Vertices[_Multiplier + 5]),
		Vec2(_Vertices[_Multiplier + 6], _Vertices[_Multiplier + 7])
	);
}

Vec2 Square::get_position(std::size_t _Index) const {
	return _Position[_Index];
}

void Square::set_position(std::size_t _Index, const Vec2& _Position, bool _Queue) {
	Vec2 _Distance(_Position[0] - _Vertices[_Index * _PointSize + 0], _Position[1] - _Vertices[_Index * _PointSize + 1]);
	for (int i = 0; i < _PointSize; i++) {
		_Vertices[_Index * _PointSize + i] += _Distance[i % 2];
	}
	if (!_Queue) {
		write(true, false);
	}
	this->_Position[_Index] = _Position;
}

void Square::push_back(const Vec2& _Position, Vec2 _Length, Color color, bool _Queue) {
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

void Square::rotate(std::size_t _Index, double _Angle, bool _Queue) {
	constexpr double offset = 3 * PI / 4;
	const Vec2 position(get_position(_Index));
	const Vec2 _Radius(_Length[_Index] / 2);
	double r = Distance(get_position(_Index), get_position(_Index) + _Length[_Index] / 2);

	Mat2x4 corners(
		Vec2(r * cos(Radians(_Angle) + offset), r * sin(Radians(_Angle) + offset)),
		Vec2(r * cos(Radians(_Angle) + offset - PI / 2.f), r * sin(Radians(_Angle) + offset - PI / 2.f)),
		Vec2(r * cos(Radians(_Angle) + offset - PI), r * sin(Radians(_Angle) + offset - PI)),
		Vec2(r * cos(Radians(_Angle) + offset - PI * 3.f / 2.f), r * sin(Radians(_Angle) + offset - PI * 3.f / 2.f))
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

Circle::Circle(const Vec2& _Position, float _Radius, std::size_t _Number_Of_Points, const Color& _Color) {
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

void Circle::set_position(std::size_t _Index, const Vec2& _Position) {
	int x = 0;
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

void Circle::push_back(Vec2 _Position, float _Radius, Color _Color, bool _Queue) {
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

Sprite::Sprite(const char* path, Vec2 positio, Vec2 size, float angle, Shader shader) :
		shader(shader), angle(angle), position(position + size / 2), size(size), texture() {
	this->camera = (Camera*)glfwGetWindowUserPointer(window);
	init_image();
	load_image(path, texture);
}

void Sprite::draw() {
	shader.Use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture.texture);
	glUniform1i(glGetUniformLocation(shader.ID, "ourTexture"), 0);
	Mat4x4 view(1);
	view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
	Mat4x4 projection(1);
	projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5f, windowSize.y + windowSize.y * 0.5f, windowSize.y / 2.f, 0.1f, 1000.0f);
	GLint projLoc = glGetUniformLocation(shader.ID, "projection");
	GLint viewLoc = glGetUniformLocation(shader.ID, "view");
	GLint modelLoc = glGetUniformLocation(shader.ID, "model");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);
	Mat4x4 model(1);
	model = Translate(model, Vec2ToVec3(position));
	if (size.x || size.y) {
		model = Scale(model, Vec3(size.x, size.y, 0));
	}
	else {
		model = Scale(model, Vec3(texture.width, texture.height, 0));
	}
	if (angle) {
		model = Rotate(model, angle, Vec3(0, 0, 1));
	}
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model.vec[0].x);
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

Vec2 Sprite::get_size() const {
	return this->size;
}

Vec2 Sprite::get_position() const {
	return this->position - this->size / 2;
}

void Sprite::set_position(const Vec2& position) {
	this->position = position;
}

Timer::Timer() {}

Timer::Timer(const decltype(high_resolution_clock::now())& timer, std::size_t time) : timer(timer), time(time) {}

void Timer::start(int time) {
	this->timer = high_resolution_clock::now();
	this->time = time;
}

void Timer::restart() {
	this->timer = high_resolution_clock::now();
}

bool Timer::finished() {
	return duration_cast<milliseconds>(high_resolution_clock::now() - timer).count() >= time;
}

std::size_t Timer::passed() {
	return duration_cast<milliseconds>(high_resolution_clock::now() - timer).count();
}

Particles::Particles(std::size_t particles_amount, Vec2 particle_size, Vec2 particle_speed, float life_time, Color begin, Color end) :
	particles(particles_amount, -particle_size,
		Vec2(particle_size), begin), particle(), particleIndex(particles_amount - 1), begin(begin), end(end), life_time(life_time) {
	for (int i = 0; i < particles_amount; i++) {
		particle.push_back({ life_time, Timer(high_resolution_clock::now(), 0), 0, particle_speed * Vec2(cosf(i), sinf(i)) });
	}
}

void Particles::add(Vec2 position) {
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
			particles.set_position(i, Vec2(-particles.get_length(0)), true);
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
		particles.set_position(i, particles.get_position(i) + particle[i].particle_speed * deltaTime, true);
	}
	particles.break_queue();
	particles.draw();
}

std::vector<Vec2> LoadMap(float blockSize) {
	std::ifstream file("data");
	std::vector<float> coordinates;
	int coordinate = 0;
	while (file >> coordinate) {
		coordinates.push_back(coordinate);
	}

	std::vector<Vec2> grid;

	for (auto i : coordinates) {
		grid.push_back(Vec2((int(i) % 14) * blockSize + blockSize / 2, int(i / 14) * blockSize + blockSize / 2));
	}

	return grid;
}

Alloc<std::size_t> blocks;

Vec2 Raycast(Square& grid, const Mat2x2& direction, std::size_t gridSize) {
	Vec2 best(-1, -1);
	for (int i = 0; i < blocks.current(); i++) {
		const Mat2x4 corners = grid.get_corners(blocks[i]);
		if (direction[0].x < direction[1].x) {
			const Vec2 inter = IntersectionPoint(direction[0], direction[1], corners[0], corners[3]);
			if (inter.x != -1) {
				if (best.x != -1) {
					if (ManhattanDistance(direction[0], inter) < ManhattanDistance(direction[0], best)) {
						best = inter;
					}
				}
				else {
					best = inter;
				}
			}
		}
		else {
			const Vec2 inter = IntersectionPoint(direction[0], direction[1], corners[1], corners[2]);
			if (inter.x != -1) {
				if (best.x != -1) {
					if (ManhattanDistance(direction[0], inter) < ManhattanDistance(direction[0], best)) {
						best = inter;
					}
				}
				else {
					best = inter;
				}
			}
		}
		if (direction[1].y < direction[0].y) {
			const Vec2 inter = IntersectionPoint(direction[0], direction[1], corners[2], corners[3]);
			if (inter.x != -1) {
				if (best.x != -1) {
					if (ManhattanDistance(direction[0], inter) < ManhattanDistance(direction[0], best)) {
						best = inter;
					}
				}
				else {
					best = inter;
				}
			}
		}
		else {
			const Vec2 inter = IntersectionPoint(direction[0], direction[1], corners[0], corners[1]);
			if (inter.x != -1) {
				if (best.x != -1) {
					if (ManhattanDistance(direction[0], inter) < ManhattanDistance(direction[0], best)) {
						best = inter;
					}
				}
				else {
					best = inter;
				}
			}
		}
	}
	return best;
}