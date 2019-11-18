#include "Texture.hpp"

void LoadImg(const char* path, Object& object, Texture& texture) {
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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_REPEAT);
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
	float vertices[] = {
		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
		0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
		0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
		0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

	};
	std::copy(std::begin(vertices), std::end(vertices), std::begin(texture.vertices));
}

Sprite::Sprite(const Sprite& info) {
	this->object = info.object;
	this->texture = info.texture;
	this->position = info.position;
}

Sprite::Sprite(const char* path) : position(0), texture(), object() {
	texture.IntializeImage(texture);
	LoadImg(path, object, texture);
	position = Vec2(object.width, object.height);
}

Sprite::Sprite(Object const& _object, Texture const& _texture, Vec2 const& _position) {
	object = _object;
	texture = _texture;
	position = _position;
}

void Sprite::SetPosition(const Vec2& position) {
	this->position = position;
}

void Sprite::Draw(Main& _Main, const Vec3& rotation, float angle, const Vec2& scale) {
	_Main.shader.Use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, object.texture);
	glUniform1i(glGetUniformLocation(_Main.shader.Program, "ourTexture"), 0);
	Mat4x4 view(1);
	view = _Main.camera.GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
	Mat4x4 projection(1);
	projection = Ortho(windowSize.x / 2, windowSize.x  + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5, windowSize.y / 2, 0.1, 1000.0f);
	GLint projLoc = glGetUniformLocation(_Main.shader.Program, "projection");
	GLint viewLoc = glGetUniformLocation(_Main.shader.Program, "view");
	GLint modelLoc = glGetUniformLocation(_Main.shader.Program, "model");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);
	Mat4x4 model(1);
	model = Translate(model, V2ToV3(position));
	if (scale.x || scale.y) {
		model = Scale(model, Vec3(scale.x, scale.y, 0));
	}
	else {
		model = Scale(model, Vec3(object.width, object.height, 0));
	}
	if (angle) {
		model = Rotate(model, angle, rotation);
	}
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model.vec[0].x);
	glBindVertexArray(object.VAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
}

Shape::Shape(Camera* camera, const Vec2& position, const Vec2& pixelSize, const Color& color, std::vector<float> vec) : shader(Shader("GLSL/shapes.vs", "GLSL/shapes.frag")), position(position), size(pixelSize), color(color),
camera(camera) {
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

void Shape::Draw() {
	this->shader.Use();
	glBindVertexArray(this->object.VAO);
	static Mat4x4 view(1);

	if (view.vec[3].x != windowSize.x / 2 || view.vec[3].y != windowSize.y / 2) {
		view = Mat4x4(1);
		view = camera->GetViewMatrix(Translate(view, Vec3(windowSize.x / 2, windowSize.y / 2, -700.0f)));
	}
	
	Mat4x4 projection(1);
	projection = Ortho(windowSize.x / 2, windowSize.x + windowSize.x * 0.5, windowSize.y + windowSize.y * 0.5, windowSize.y / 2, 0.1, 1000.0f);
	static int projLoc = glGetUniformLocation(shader.Program, "projection");
	static int viewLoc = glGetUniformLocation(shader.Program, "view");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.vec[0].x);
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection.vec[0].x);
	static int modelLoc = glGetUniformLocation(shader.Program, "model");
	Mat4x4 model(1);
	model = Translate(model, Vec3(position.x, position.y, 0));
	if (size.x != 0 || size.y != 0) {
		model = Scale(model, Vec3(size.x, size.y, 0));
	}
	//model = Rotate(model, PI, Vec3(0, 0, 1));
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model.vec[0].x);
	static int colorLoc = glGetUniformLocation(shader.Program, "color");
	glUniform4f(colorLoc, this->color.r / 0xff, this->color.g / 0xff, this->color.b / 0xff, this->color.a / 0xff);

	glDrawArrays(this->type, 0, this->points);
	glBindVertexArray(0);
}

void Shape::SetPosition(const Vec2& position) {
	this->position = position;
}

void Shape::SetColor(Color color) {
	this->color = color;
}

void Triangle::Add(const Vec2& position, Vec2 size, Color color) {
	if (size.x != 0 || size.y != 0) {
		
	}
	if (color.r != 0 && color.g != 0 && color.b != 0 && color.a != 0) {
		
	}
	//this->size = size;
	//this->color = color;
	this->vertices.insert(this->vertices.end(), std::begin(triangle_vertices), std::end(triangle_vertices));

	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	this->points += 3;
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

void Line::SetPosition(size_t where, const Mat2x2& begin_end) {
	for (int i = where * 4, j = 0; i < where * 4 + 4; i++, j++) {
		this->vertices[i] = begin_end.vec[(i & 2) >> 1][i & 1];
	}
	 
	glBindBuffer(GL_ARRAY_BUFFER, this->object.VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(this->vertices[0]) * this->vertices.size(), this->vertices.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


Entity::Entity(const char* path, GroupId _groupId = GroupId::NotAssigned) : velocity(0), groupId(_groupId) {
	texture.IntializeImage(texture);
	LoadImg(path, object, texture);
}

Entity::Entity(const char* path, Vec2 pos, GroupId _groupId = GroupId::NotAssigned) : velocity(0), groupId(_groupId) {
	texture.IntializeImage(texture);
	LoadImg(path, object, texture);
	position = pos;
}

GroupId Entity::GetGroupId() {
	return this->groupId;
}

void Entity::SetGroupId(GroupId groupId) {
	this->groupId = groupId;
}

void Entity::Move(Main& _Main) {
	velocity.x /= (deltaTime * friction) + 1;
	velocity.y /= (deltaTime * friction) + 1;
	if (KeyPress(GLFW_KEY_W)) velocity.y -= deltaTime * this->movementSpeed;
	if (KeyPress(GLFW_KEY_S)) velocity.y += deltaTime * this->movementSpeed;
	if (KeyPress(GLFW_KEY_A)) velocity.x -= deltaTime * this->movementSpeed;
	if (KeyPress(GLFW_KEY_D)) velocity.x += deltaTime * this->movementSpeed;

	position.x += (velocity.x * deltaTime);
	position.y += (velocity.y * deltaTime);

	//if (moveCam) {
		_Main.camera.position = position;
	//}
}