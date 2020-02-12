#include <iostream>
#include <FAN/Graphics.hpp>
#include <FAN/DBT.hpp>
#include <vector>
#include <chrono>

struct Particle {
	float life_time;
	decltype(std::chrono::high_resolution_clock::now()) time;
	bool display;
	Vec2 particle_speed;
};

class Particles {
public:
	Particles(std::size_t particles_amount, Vec2 particle_size, Vec2 particle_speed, float life_time) :
		particles(particles_amount, -particle_size, 
		Vec2(particle_size), Color(/*105.f / 255.f, 105.f / 255.f, 105.f / 255.f, 1)*/1, 0, 0)), particle(), particleIndex(particles_amount) {
		for (int i = 0; i < particles_amount; i++) {
			//float theta = 2.0f * 3.1415926f * float(i) / float(100);
			float x = particle_speed.x * cosf(i);
			float y = particle_speed.y * sinf(i);
			particle.push_back({ life_time, decltype(std::chrono::high_resolution_clock::now())(), 0, Vec2(x, y) });
			particles.rotate(i, rand(), true);
		}
		particles.break_queue();
	}

	void add(Vec2 position) {
		if (!particle[particleIndex].time.time_since_epoch().count()) {
			particles.set_position(particleIndex, position);
			particle[particleIndex].time = std::chrono::high_resolution_clock::now();
			particle[particleIndex].display = true;
			particleIndex = (particleIndex - 1) % particles.amount();
		}
	}

	void draw() {
		using namespace std::chrono;
		for (int i = 0; i < particles.amount(); i++) {
			if (!particle[i].display) {
				continue;
			}
			if (duration_cast<milliseconds>(high_resolution_clock::now() - particle[i].time).count() >= particle[i].life_time) {
				particles.set_position(i, Vec2(-particles.get_length(0)), true);
				particle[i].display = false;
				particle[i].time = decltype(high_resolution_clock::now())();
				continue;
			}
			Color color = particles.get_color(i);
			particles.set_color(i, Color(color.r, color.g, (float)duration_cast<milliseconds>((high_resolution_clock::now() - particle[i].time)).count() / 1000.f, 
				(particle[i].life_time - (float)duration_cast<milliseconds>((high_resolution_clock::now() - particle[i].time)).count() / 1.f) / particle[i].life_time), true);
			particles.set_position(i, particles.get_position(i) + Vec2(particle[i].particle_speed.x * deltaTime, particle[i].particle_speed.y * deltaTime), true);
		}

		particles.break_queue();
		particles.draw();
	}

private:
	std::size_t particleIndex;
	Square particles;
	Alloc<Particle> particle;
};

class Timer {
public:
	void start(int time) {
		this->timer = std::chrono::high_resolution_clock::now();
		this->time = time;
	}
	void restart() {
		this->timer = std::chrono::high_resolution_clock::now();
	}
	bool finished() {
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timer).count() >= time;
	}
private:
	decltype(std::chrono::high_resolution_clock::now()) timer;
	int time;
};

int main() {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		printf("GLFW ded\n");
		return 0;
	}
	WindowInit();
	Main _Main;
	_Main.shader.Use();
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharacterCallback);
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetWindowUserPointer(window, &_Main.camera);
	glfwSetCursorPosCallback(window, CursorPositionCallback);
	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetFramebufferSizeCallback(window, FrameSizeCallback);

	using namespace std::chrono;

	Particles particles(10000, Vec2(20), Vec2(100), 2000);

	Timer timer;
	timer.start(10);

	auto time = high_resolution_clock::now();

	Square mycar(Vec2(900 / 2), Vec2(64), Color(1, 0, 0, 1));

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		if (KeyPress(GLFW_MOUSE_BUTTON_LEFT)) {
			particles.add(cursorPos);
		}

		particles.draw();

		if (KeyPress(GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(window, true);
		}

		GetFps();
		glfwSwapBuffers(window);
		KeysReset();
	}
}