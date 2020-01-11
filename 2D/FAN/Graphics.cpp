#include <FAN/Graphics.hpp>
#include <FAN/Bmp.hpp>
#include <functional>

//std::vector<Vec2> LoadMap(float blockSize) {
//	std::ifstream file("data");
//	std::vector<float> coordinates;
//	int coordinate = 0;
//	while (file >> coordinate) {
//		coordinates.push_back(coordinate);
//	}
//
//	std::vector<Vec2> grid;
//
//	for (auto i : coordinates) {
//		grid.push_back(Vec2((int(i) % 14) * blockSize + blockSize / 2, int(i / 14) * blockSize + blockSize / 2));
//	}
//
//	return grid;
//}

//void LoadImg(const char* path, Object& object, Texture& texture) {
//	std::ifstream file(path);
//	if (!file.good()) {
//		printf("File path does not exist\n");
//		return;
//	}
//	glGenVertexArrays(1, &object.VAO);
//	glGenBuffers(1, &object.VBO);
//	glBindVertexArray(object.VAO);
//	glBindBuffer(GL_ARRAY_BUFFER, object.VBO);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(texture.vertices), texture.vertices, GL_STATIC_DRAW); //almost xd colors are durnk
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
//	glEnableVertexAttribArray(2);
//	glBindVertexArray(0);
//	glGenTextures(1, &object.texture);
//	glBindTexture(GL_TEXTURE_2D, object.texture);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, object.width, object.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, LoadBMP(path, object));
//	glGenerateMipmap(GL_TEXTURE_2D);
//	glBindTexture(GL_TEXTURE_2D, 0);
//}
//
//void Texture::IntializeImage(Texture& texture) {
//	const float vertices[] = {
//		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
//		0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
//		0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
//		0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
//		-0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
//		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f
//	};
//	std::copy(std::begin(vertices), std::end(vertices), std::begin(texture.vertices));
//}
//
//Sprite::Sprite(const Sprite& info) {
//	this->object = info.object;
//	this->texture = info.texture;
//	this->position = info.position;
//}
//
//Sprite::Sprite(Camera* camera, const char* path, Vec2 size, Vec2 position, Shader shader, float angle) : shader(shader), camera(camera), angle(angle), position(0), texture(), object() {
//	texture.IntializeImage(texture);
//	LoadImg(path, object, texture);
//	this->size = Vec2(object.width * size.x, object.height * size.y);
//	this->position = Vec2(position.x - this->size.x / 2, position.y - this->size.y / 2);
//}

void Sprite::SetPosition(const Vec2& position) {
	this->position = position;
}

void Sprite::Draw() {
	shader.Use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture.texture);
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

//Entity::Entity(const char* path, GroupId _groupId = GroupId::NotAssigned) : velocity(0), groupId(_groupId) {
//	texture.IntializeImage(texture);
//	LoadImg(path, object, texture);
//}
//
//GroupId Entity::GetGroupId() {
//	return this->groupId;
//}
//
//void Entity::SetGroupId(GroupId groupId) {
//	this->groupId = groupId;
//}

//void Entity::Move() {
	//static float groundPosition = floor((windowSize.y - BLOCKSIZE * BLOCKAMOUNT) - this->size.y / 2);
	//bool playerOnGround = collision.IsColliding(position);

//	if (playerOnGround) {

	//}

	//bool moving = false;
	//if (KeyPress(GLFW_KEY_A)) {
	//	if (this->object.texture != this->objects[1].texture) {
	//		this->object = this->objects[1];
	//		this->texture = this->textures[1];
	//	}
	//	velocity.x -= deltaTime * this->movementSpeed;
	//	moving = true;
	//}
	//if (KeyPress(GLFW_KEY_D)) {
	//	if (this->object.texture != this->objects[0].texture) {
	//		this->object = this->objects[0];
	//		this->texture = this->textures[0];
	//	}
	//	velocity.x += deltaTime * this->movementSpeed;
	//	moving = true;
	//}
	//velocity.x /= (deltaTime * friction) + 1;
	//velocity.y /= (deltaTime * friction) + 1;
	//static float downAccel = 0;
	//if (playerOnGround) {
	//	//position.y = groundPosition;
	//	velocity.y = 0;
	//	downAccel = 0;
	//}
	//else {
	//	downAccel += deltaTime * (this->gravity);
	//	//this->velocity.y += deltaTime * downAccel;
	//}

	//if (KeyPress(GLFW_KEY_SPACE) && playerOnGround) {
	//	this->velocity.y -= this->jumpForce;
	//}

	//Vec2 newPosition = position + velocity * deltaTime;
	////printf("%x\n", playerOnGround);
	//if (collision.isCollidingCustom(this->position, newPosition) || collision.IsColliding(newPosition)) {
	//	if (!playerOnGround) {
	//		velocity = Vec2();
	//	}
	//}
	//else {
	//	position += (velocity * deltaTime);
	//}



	//const float middle = position.x - windowSize.x / 2;
	//const float cameraMoveOffset = 200;

	//if (camera->position.x < middle - cameraMoveOffset) {
	//	camera->position.x = middle - cameraMoveOffset;
	//}
	//else if (camera->position.x > middle + cameraMoveOffset) {
	//	camera->position.x = middle + cameraMoveOffset;
	//}
//}