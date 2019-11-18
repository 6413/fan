#pragma once
#include "Math.hpp"

class Camera
{
public:
	Vec3 position;

	Camera(Vec3 position = Vec3(0.0f, 0.0f, 0.0f), Vec3 up = Vec3(0.0f, 1.0f, 0.0f), float yaw = -90.0f, float pitch = 0.0f) : front(Vec3(0.0f, 0.0f, -1.0f))
	{
		this->position = position;
		this->worldUp = up;
		this->yaw = yaw;
		this->pitch = pitch;
		this->updateCameraVectors();
	}

	Mat4x4 GetViewMatrix(Mat4x4 m) {
		return m * LookAt(this->position, (this->position + Round(this->front)), Round(this->up));
	}

	void ProcessKeyboard(Vec2 direction) {
		this->position.x += direction.x;
		this->position.y += direction.y;
	}

private:
	Vec3 front;
	Vec3 up;
	Vec3 right;
	Vec3 worldUp;

	GLfloat yaw;
	GLfloat pitch;
	void updateCameraVectors() {
		Vec3 front;
		front.x = cos(Radians(this->yaw)) * cos(Radians(this->pitch));
		front.y = sin(Radians(this->pitch));
		front.z = sin(Radians(this->yaw)) * cos(Radians(this->pitch));
		this->front = Normalize(front);
		this->right = Normalize(Cross(this->front, this->worldUp));
		this->up = Normalize(Cross(this->right, this->front));
	}
};