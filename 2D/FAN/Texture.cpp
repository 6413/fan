#include "Texture.hpp"
#include "FAN/Bmp.hpp"
#include <functional>

Collision collision;

void LoadImg(const char* path, Object& object, Texture& texture) {
	std::ifstream file(path);
	if (!file.good()) {
		printf("File path does not exist\n");
		return;
	}
	glGenVertexArrays(1, &object.VAO);
	glGenBuffers(1, &object.VBO);
	glBindVertexArray(object.VAO);
	glBindBuffer(GL_ARRAY_BUFFER, object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texture.vertices), texture.vertices, GL_STATIC_DRAW); //almost xd colors are durnk
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

void Texture::IntializeImage(Texture& texture) {
	const float vertices[] = {
		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
		0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
		0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
		0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f
	};
	std::copy(std::begin(vertices), std::end(vertices), std::begin(texture.vertices));
}

Sprite::Sprite(const Sprite& info) {
	this->object = info.object;
	this->texture = info.texture;
	this->position = info.position;
}

Sprite::Sprite(Camera* camera, const char* path, Vec2 size, Vec2 position, Shader shader, float angle) : shader(shader), camera(camera), angle(angle), position(0), texture(), object() {
	texture.IntializeImage(texture);
	LoadImg(path, object, texture);
	this->size = Vec2(object.width * size.x, object.height * size.y);
	this->position = Vec2(position.x - this->size.x / 2, position.y - this->size.y / 2);
}

void Sprite::SetPosition(const Vec2& position) {
	this->position = position;
}

void Sprite::Draw() {
	shader.Use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, object.texture);
	glUniform1i(glGetUniformLocation(shader.ID, "ourTexture"), 0);
	Mat4x4 view(1);
	view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
	Mat4x4 projection(1);
	projection = Ortho(windowSize.x / 2, windowSize.x  + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5, windowSize.y / 2, 0.1, 1000.0f);
	GLint projLoc = glGetUniformLocation(shader.ID, "projection");
	GLint viewLoc = glGetUniformLocation(shader.ID, "view");
	GLint modelLoc = glGetUniformLocation(shader.ID, "model");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);
	Mat4x4 model(1);
	model = Translate(model, V2ToV3(position));
	if (size.x || size.y) {
		model = Scale(model, Vec3(size.x, size.y, 0));
	}
	else {
		model = Scale(model, Vec3(object.width, object.height, 0));
	}
	if (angle) {
		model = Rotate(model, angle, Vec3(0, 0, 1));
	}
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model.vec[0].x);
	glBindVertexArray(object.VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
}

Shape::Shape(Camera* camera, const Vec2& position, const Vec2& pixelSize, const Color& color, size_t _vertSize, std::vector<float> vec,
		Shader shader) : shader(shader), position(position), size(pixelSize), color(color), camera(camera), angle(0), vertSize(_vertSize) {
	glGenVertexArrays(1, &this->object.VAO);
	glGenBuffers(1, &this->object.VBO);
	glBindVertexArray(this->object.VAO);

	for (auto i : vec) {
		vertices.push_back(i);
	}
	  
	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);


	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

Shape::~Shape() {
	glDeleteVertexArrays(1, &this->object.VAO);
	glDeleteBuffers(1, &this->object.VBO);
}

void Shape::Draw() {
	this->shader.Use();
	Mat4x4 view(1);

	view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
	
	Mat4x4 projection(1);
	projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5f, windowSize.y / 2, 0.1f, 1000.0f);
	int projLoc = glGetUniformLocation(shader.ID, "projection");
	int viewLoc = glGetUniformLocation(shader.ID, "view");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);

	//static int colorLoc = glGetUniformLocation(shader.ID, "color");
	//glUniform4f(colorLoc, this->color.r / 0xff, this->color.g / 0xff, this->color.b / 0xff, this->color.a / 0xff);

	//Mat4x4 model(1);
	//model = Translate(model, Vec3(this->position));
	//glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &view.vec[0].x);

	glBindVertexArray(this->object.VAO);
	glDrawArrays(this->type, 0, this->points);
	glBindVertexArray(0);
}

void Shape::DrawLight(Square& light) {
	this->shader.Use();

	Mat4x4 view(1);

	view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));

	Mat4x4 projection(1);
	projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5, windowSize.y / 2, 0.1, 1000.0f);
	static int projLoc = glGetUniformLocation(shader.ID, "projection");
	static int viewLoc = glGetUniformLocation(shader.ID, "view");
	static int modelLoc = glGetUniformLocation(shader.ID, "model");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);

	static int colorLoc = glGetUniformLocation(shader.ID, "color");
	glUniform4f(colorLoc, this->color.r / 0xff, this->color.g / 0xff, this->color.b / 0xff, this->color.a / 0xff);

	Mat4x4 model(1);
	model = Translate(model, Vec3(this->position));
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model.vec[0].x);

	glBindVertexArray(this->object.VAO);
	glDrawArrays(this->type, 0, this->points);
	glBindVertexArray(0);
}

void Shape::SetColor(size_t _Where, Color color) {
	for (int i = 0; i < 24; i+=6) {
		this->vertices[(i +_Where * (24)) + 2] = color.r;
		this->vertices[(i +_Where * (24)) + 3] = color.g;
		this->vertices[(i +_Where * (24)) + 4] = color.b;
		this->vertices[(i +_Where * (24)) + 5] = color.a;
	}

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shape::SetColor(size_t _Where, char _operator, int value) {
	for (int i = 0; i < 24; i += 6) {
		switch (_operator) {
		case '^': {
			this->vertices[(i + _Where * (24)) + 2] = (int)vertices[(i + _Where * (24)) + 2] ^ value;
			this->vertices[(i + _Where * (24)) + 3] = (int)vertices[(i + _Where * (24)) + 3] ^ value;
			this->vertices[(i + _Where * (24)) + 4] = (int)vertices[(i + _Where * (24)) + 4] ^ value;
			break;
		}
		}

	}
	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shape::Rotatef(float angle, Vec2 point) {
	angle = -angle;
	float c1 = this->vertices[2] - this->vertices[0];
	float c2 = this->vertices[3] - this->vertices[1];
	Vec2 middle;
	if (point != MIDDLE) {
		middle = point;
	}
	else {
		middle = Vec2(this->vertices[0] + 0.5 * c1, this->vertices[1] + 0.5 * c2);
	}

	this->vertices[0] = -(cos(Radians(angle)) * c1 - sin(Radians(angle)) * c2) * 0.5 + middle.x;
	this->vertices[1] = -(sin(Radians(angle)) * c1 + cos(Radians(angle)) * c2) * 0.5 + middle.y;
	this->vertices[2] = (cos(Radians(angle)) * c1 - sin(Radians(angle)) * c2) * 0.5 + middle.x;
	this->vertices[3] = (sin(Radians(angle)) * c1 + cos(Radians(angle)) * c2) * 0.5 + middle.y;

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Vec2 Shape::Size() const {
	return this->size;
}

Vec2 Shape::GetPosition(size_t _Where) const {
	float xstart = this->vertices[_Where * this->vertSize];
	float xend = this->vertices[_Where * this->vertSize + 2 + 4];
	float ystart = this->vertices[_Where * this->vertSize + 1];
	float yend = this->vertices[(_Where * this->vertSize + 3 + 4) * 2 - 1];
	return Vec2(xstart + (xend - xstart) / 2, ystart + (yend - ystart) / 2);
}

void Triangle::Add(const Vec2& position, Vec2 size) {
	this->vertices.push_back(position.x - (size.x / 2));
	this->vertices.push_back(position.y + (size.y / 2));
	this->vertices.push_back(position.x + (size.x / 2));
	this->vertices.push_back(position.y + (size.y / 2));
	this->vertices.push_back(position.x);
	this->vertices.push_back(position.y - (size.y / 2));

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	this->points += 3;
}

void Triangle::SetPosition(size_t _Where, const Vec2& position) {
	this->vertices[_Where * TRIANGLEVERT + 0] = position.x - (size.x / 2);
	this->vertices[_Where * TRIANGLEVERT + 1] = position.y + (size.y / 2);
	this->vertices[_Where * TRIANGLEVERT + 2] = position.x + (size.x / 2);
	this->vertices[_Where * TRIANGLEVERT + 3] = position.y + (size.y / 2);
	this->vertices[_Where * TRIANGLEVERT + 4] = position.x;
	this->vertices[_Where * TRIANGLEVERT + 5] = position.y - (size.y / 2);

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Square::Add(const Vec2& position, const Vec2& size, const Color& color) {
	const bool arr[] = { 0, 0, 1, 0, 1, 1, 0, 1 };

	for (int i = 0; i < 8; i++) {
		if (color.r != -1 && ((i % 2) == 0 && i)) {
			this->vertices.push_back(color.r);
			this->vertices.push_back(color.g);
			this->vertices.push_back(color.b);
			this->vertices.push_back(color.a);
		}
		else if ((i % 2) == 0 && i) {
			this->vertices.push_back(this->color.r);
			this->vertices.push_back(this->color.g);
			this->vertices.push_back(this->color.b);
			this->vertices.push_back(this->color.a);
		}
		this->vertices.push_back(!arr[i] ? position[i & 1] - (this->size[i & 1] / 2) : position[i & 1] + (this->size[i & 1] / 2));
	}
	for (int i = 0; i < 4; i++) {
		this->vertices.push_back(color.r != -1 ? color[i] : this->color[i]);
	}
	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	this->points += SQUAREVERT / 2;
}

void Square::SetPosition(size_t _Where, const Vec2& position) {

	if (position.x == cursorPos.x && position.y == cursorPos.y) {
		this->vertices[_Where * SQUAREVERT + 0 + COLORSIZE * 0] = camera->position.x + position.x - (this->size.x / 2);
		this->vertices[_Where * SQUAREVERT + 1 + COLORSIZE * 0] = camera->position.y + position.y - (this->size.y / 2);
		this->vertices[_Where * SQUAREVERT + 2 + COLORSIZE * 1] = camera->position.x + position.x + (this->size.x / 2);
		this->vertices[_Where * SQUAREVERT + 3 + COLORSIZE * 1] = camera->position.y + position.y - (this->size.y / 2);
		this->vertices[_Where * SQUAREVERT + 4 + COLORSIZE * 2] = camera->position.x + position.x + (this->size.x / 2);
		this->vertices[_Where * SQUAREVERT + 5 + COLORSIZE * 2] = camera->position.y + position.y + (this->size.y / 2);
		this->vertices[_Where * SQUAREVERT + 6 + COLORSIZE * 3] = camera->position.x + position.x - (this->size.x / 2);
		this->vertices[_Where * SQUAREVERT + 7 + COLORSIZE * 3] = camera->position.y + position.y + (this->size.y / 2);

	}
	else {
		this->vertices[_Where * SQUAREVERT + 0 + COLORSIZE * 0] = position.x - (this->size.x / 2);
		this->vertices[_Where * SQUAREVERT + 1 + COLORSIZE * 0] = position.y - (this->size.y / 2);
		this->vertices[_Where * SQUAREVERT + 2 + COLORSIZE * 1] = position.x + (this->size.x / 2);
		this->vertices[_Where * SQUAREVERT + 3 + COLORSIZE * 1] = position.y - (this->size.y / 2);
		this->vertices[_Where * SQUAREVERT + 4 + COLORSIZE * 2] = position.x + (this->size.x / 2);
		this->vertices[_Where * SQUAREVERT + 5 + COLORSIZE * 2] = position.y + (this->size.y / 2);
		this->vertices[_Where * SQUAREVERT + 6 + COLORSIZE * 3] = position.x - (this->size.x / 2);
		this->vertices[_Where * SQUAREVERT + 7 + COLORSIZE * 3] = position.y + (this->size.y / 2);
	}

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	this->position = position;
}

void Line::Add(const Mat2x2& begin_end, const Color& color) {
	for (int i = 0; i < 4; i++) {
		this->vertices.push_back(begin_end.vec[(i & 2) >> 1][i & 1]);
		if (color.r != -1 && ((i & 2) == 0 && i)) {
			this->vertices.push_back(color.r);
			this->vertices.push_back(color.g);
			this->vertices.push_back(color.b);
			this->vertices.push_back(color.a);
		}
		else if ((i & 2) == 0 && i) {
			this->vertices.push_back(this->color.r);
			this->vertices.push_back(this->color.g);
			this->vertices.push_back(this->color.b);
			this->vertices.push_back(this->color.a);	
		}
	}

	for (int i = 0; i < 4; i++) {
		this->vertices.push_back(color.r != -1 ? color[i] : this->color[i]);
	}


	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	this->points += 2;
}

void Line::SetPosition(size_t _Where, const Mat2x2& begin_end) {

	this->vertices[_Where + COLORSIZE + LINEVERT / 2 * LINEVERT + 0] = begin_end.vec[0][0];
	this->vertices[_Where + COLORSIZE + LINEVERT / 2 * LINEVERT + 1] = begin_end.vec[0][1];
	this->vertices[_Where + COLORSIZE + LINEVERT / 2 * LINEVERT + 4 + 2] = begin_end.vec[1][0];
	this->vertices[_Where + COLORSIZE + LINEVERT / 2 * LINEVERT + 4 + 3] = begin_end.vec[1][1];


	//for (int i = _Where * 4; i < _Where * 4 + 4; i++) {
	//	this->vertices[i] = begin_end.vec[(i & 2) >> 1][i & 1];
	//}
	 
	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Line::SetPosition(size_t _Where, const Vec2& position) {
	for (int i = 0; i < this->vertSize; i++) {
		this->vertices[_Where * this->vertSize + i] = i < 2 ? position[i & 1] - size[i & 1] / 2 : position[i & 1] + size[i & 1] / 2;
	}

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Entity::Entity(const char* path, GroupId _groupId = GroupId::NotAssigned) : velocity(0), groupId(_groupId) {
	texture.IntializeImage(texture);
	LoadImg(path, object, texture);
}

GroupId Entity::GetGroupId() {
	return this->groupId;
}

void Entity::SetGroupId(GroupId groupId) {
	this->groupId = groupId;
}

void Entity::Move() {
	static float groundPosition = floor((windowSize.y - BLOCKSIZE * BLOCKAMOUNT) - this->size.y / 2);
	bool playerOnGround = collision.IsColliding(position);

//	if (playerOnGround) {

	//}

	bool moving = false;
	if (KeyPress(GLFW_KEY_A)) {
		if (this->object.texture != this->objects[1].texture) {
			this->object = this->objects[1];
			this->texture = this->textures[1];
		}
		velocity.x -= deltaTime * this->movementSpeed;
		moving = true;
	}
	if (KeyPress(GLFW_KEY_D)) {
		if (this->object.texture != this->objects[0].texture) {
			this->object = this->objects[0];
			this->texture = this->textures[0];
		}
		velocity.x += deltaTime * this->movementSpeed;
		moving = true;
	}
	velocity.x /= (deltaTime * friction) + 1;
	velocity.y /= (deltaTime * friction) + 1;
	static float downAccel = 0;
	if (playerOnGround) {
		//position.y = groundPosition;
		velocity.y = 0;
		downAccel = 0;
	}
	else {
		downAccel += deltaTime * (this->gravity);
		//this->velocity.y += deltaTime * downAccel;
	}

	if (KeyPress(GLFW_KEY_SPACE) && playerOnGround) {
		this->velocity.y -= this->jumpForce;
	}

	Vec2 newPosition = position + velocity * deltaTime;
	//printf("%x\n", playerOnGround);
	if (collision.isCollidingCustom(this->position, newPosition) || collision.IsColliding(newPosition)) {
		if (!playerOnGround) {
			velocity = Vec2();
		}
	}
	else {
		position += (velocity * deltaTime);
	}



	//const float middle = position.x - windowSize.x / 2;
	//const float cameraMoveOffset = 200;

	//if (camera->position.x < middle - cameraMoveOffset) {
	//	camera->position.x = middle - cameraMoveOffset;
	//}
	//else if (camera->position.x > middle + cameraMoveOffset) {
	//	camera->position.x = middle + cameraMoveOffset;
	//}
}