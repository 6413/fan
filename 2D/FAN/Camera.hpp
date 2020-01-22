#pragma once
#include "Math.hpp"

const float SENSITIVITY = 0.1f;

enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

class Camera
{
public:
	Vec3 position;
	float MouseSensitivity;

	Camera(Vec3 position = Vec3(0.0f, 0.0f, 0.0f), Vec3 up = Vec3(0.0f, 1.0f, 0.0f), float yaw = -90.0f, float pitch = 0.0f) : front(Vec3(0.0f, 0.0f, -1.0f)), MouseSensitivity(SENSITIVITY)
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

	void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
	{
		xoffset *= MouseSensitivity;
		yoffset *= MouseSensitivity;

		yaw += xoffset;
		pitch += yoffset;

		// Make sure that when pitch is out of bounds, screen doesn't get flipped
		if (constrainPitch)
		{
			if (pitch > 89.0f)
				pitch = 89.0f;
			if (pitch < -89.0f)
				pitch = -89.0f;
		}

		// Update Front, Right and Up Vectors using the updated Euler angles
		updateCameraVectors();
	}

	Vec3 GetPosition() const {
		return this->position;
	}

	Vec3 GetFront() const {
		return this->front;
	}

	GLfloat yaw;
	GLfloat pitch;
private:
	Vec3 front;
	Vec3 up;
	Vec3 right;
	Vec3 worldUp;


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