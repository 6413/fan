#include "Texture.hpp"

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
	glBufferData(GL_ARRAY_BUFFER, sizeof(texture.vertices), texture.vertices, GL_STATIC_DRAW);
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
	unsigned char* image = SOIL_load_image(path, &object.width, &object.height, 0, SOIL_LOAD_RGBA);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, object.width, object.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
	glGenerateMipmap(GL_TEXTURE_2D);
	SOIL_free_image_data(image);
	glBindTexture(GL_TEXTURE_2D, 0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_ALPHA);
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

Sprite::Sprite(Camera* camera, const char* path, Vec2 size, Vec2 position, float angle) : camera(camera), angle(angle), position(0), texture(), object() {
	shader = Shader("GLSL/core.vs", "GLSL/core.frag");
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
	glUniform1i(glGetUniformLocation(shader.Program, "ourTexture"), 0);
	Mat4x4 view(1);
	view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
	Mat4x4 projection(1);
	projection = Ortho(windowSize.x / 2, windowSize.x  + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5, windowSize.y / 2, 0.1, 1000.0f);
	GLint projLoc = glGetUniformLocation(shader.Program, "projection");
	GLint viewLoc = glGetUniformLocation(shader.Program, "view");
	GLint modelLoc = glGetUniformLocation(shader.Program, "model");
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

Shape::Shape(Camera* camera, const Vec2& position, const Vec2& pixelSize, const Color& color, std::vector<float> vec) : 
	shader(Shader("GLSL/shapes.vs", "GLSL/shapes.frag")), position(position), size(pixelSize), color(color), camera(camera), angle(0) {
	glGenVertexArrays(1, &this->object.VAO);
	glGenBuffers(1, &this->object.VBO);
	glBindVertexArray(this->object.VAO);

	for (auto i : vec) {
		vertices.push_back(i);
	}

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

Shape::~Shape() {
	glDeleteVertexArrays(1, &this->object.VAO);
	glDeleteBuffers(1, &this->object.VBO);
}

void Shape::Draw(Entity& player) {
	this->shader.Use();

	this->shader.SetVec3("objectColor", this->GetColor(true));
	this->shader.SetVec3("lightColor", Vec3(1.0f, 1.0f, 1.0f));
	this->shader.SetVec3("lightPos", Vec2(player.GetPosition().x + 1000, player.GetPosition().y));

	Mat4x4 view(1);

	view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
	
	Mat4x4 projection(1);
	projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5, windowSize.y / 2, 0.1, 1000.0f);
	static int projLoc = glGetUniformLocation(shader.Program, "projection");
	static int viewLoc = glGetUniformLocation(shader.Program, "view");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);

	static int colorLoc = glGetUniformLocation(shader.Program, "color");
	glUniform4f(colorLoc, this->color.r / 0xff, this->color.g / 0xff, this->color.b / 0xff, this->color.a / 0xff);

	glBindVertexArray(this->object.VAO);
	glDrawArrays(this->type, 0, this->points);
	glBindVertexArray(0);
}

void Shape::SetColor(Color color) {
	this->color = color;
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Vec2 Shape::Size() const {
	return this->size;
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

void Square::Add(const Vec2& position, const Vec2& size) {
	this->vertices.push_back(position.x - (this->size.x / 2));
	this->vertices.push_back(position.y - (this->size.y / 2));
	this->vertices.push_back(position.x + (this->size.x / 2));
	this->vertices.push_back(position.y - (this->size.y / 2));
	this->vertices.push_back(position.x + (this->size.x / 2));
	this->vertices.push_back(position.y + (this->size.y / 2));
	this->vertices.push_back(position.x - (this->size.x / 2));
	this->vertices.push_back(position.y + (this->size.y / 2));

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	this->points += SQUAREVERT / 2;
}

void Square::SetPosition(size_t _Where, const Vec2& position) {
	this->vertices[_Where * SQUAREVERT + 0] = position.x - (this->size.x / 2);
	this->vertices[_Where * SQUAREVERT + 1] = position.y - (this->size.y / 2);
	this->vertices[_Where * SQUAREVERT + 2] = position.x + (this->size.x / 2);
	this->vertices[_Where * SQUAREVERT + 3] = position.y - (this->size.y / 2);
	this->vertices[_Where * SQUAREVERT + 4] = position.x + (this->size.x / 2);
	this->vertices[_Where * SQUAREVERT + 5] = position.y + (this->size.y / 2);
	this->vertices[_Where * SQUAREVERT + 6] = position.x - (this->size.x / 2);
	this->vertices[_Where * SQUAREVERT + 7] = position.y + (this->size.y / 2);

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Line::Add(const Mat2x2& begin_end) {
	for (int i = 0; i < 4; i++) {
		this->vertices.push_back(begin_end.vec[(i & 2) >> 1][i & 1]);
	}

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	this->points += 2;
}

void Line::SetPosition(size_t _Where, const Mat2x2& begin_end) {
	for (int i = _Where * 4, j = 0; i < _Where * 4 + 4; i++, j++) {
		this->vertices[i] = begin_end.vec[(i & 2) >> 1][i & 1];
	}
	 
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
	velocity.x /= (deltaTime * friction) + 1;
	velocity.y /= (deltaTime * friction) + 1;
	if (KeyPress(GLFW_KEY_A)) {
		if (this->object.texture != this->objects[1].texture) {
			this->object = this->objects[1];
			this->texture = this->textures[1];
		}
		velocity.x -= deltaTime * this->movementSpeed;
	}
	if (KeyPress(GLFW_KEY_D)) {
		if (this->object.texture != this->objects[0].texture) {
			this->object = this->objects[0];
			this->texture = this->textures[0];
		}
		velocity.x += deltaTime * this->movementSpeed;
	}
	
	static bool jump = false;
	static double jumpTime;
	const float onGrass = windowSize.y - GRASSHEIGHT - this->size.y / 2;

	if (KeyPressA(GLFW_KEY_SPACE) && position.y == onGrass) {
		jumpTime = glfwGetTime();
		jump = true;
	}

	if (jump) {

		if ((glfwGetTime() - jumpTime) > 0.2) {
			jump = false;
		}
		this->velocity.y -= deltaTime * this->jumpForce;
	}

	if (position.y > onGrass) {
		this->position.y = onGrass;
		this->velocity.y = 0;
	}
	else if (position.y < onGrass) {
		this->velocity.y += deltaTime * this->gravity;
	}

	position += (velocity * deltaTime);

	const float middle = position.x - windowSize.x / 2;
	const float cameraMoveOffset = 200;

	if (camera->position.x < middle - cameraMoveOffset) {
		camera->position.x = middle - cameraMoveOffset;
	}
	else if (camera->position.x > middle + cameraMoveOffset) {
		camera->position.x = middle + cameraMoveOffset;
	}
}