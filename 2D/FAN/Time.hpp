#pragma once

#include <chrono>

using namespace std::chrono;

class Timer {
public:
	Timer() : time(0) {}

	Timer(const decltype(high_resolution_clock::now())& timer, std::size_t time) : timer(timer), time(time) {}

	void start(int time) {
		this->timer = high_resolution_clock::now();
		this->time = time;
	}

	static decltype(high_resolution_clock::now()) start() {
		return high_resolution_clock::now();
	}

	void restart() {
		this->timer = high_resolution_clock::now();
	}

	bool finished() {
		return duration_cast<milliseconds>(high_resolution_clock::now() - timer).count() >= time;
	}

	std::size_t passed() {
		return duration_cast<milliseconds>(high_resolution_clock::now() - timer).count();
	}

private:
	decltype(high_resolution_clock::now()) timer;
	std::size_t time;
};