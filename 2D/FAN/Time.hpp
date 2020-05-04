#pragma once

#include <chrono>

using namespace std::chrono;

class Timer {
public:
	Timer() : time(0) {}

	enum class mode {
		WAIT_FINISH,
		EVERY_OTHER
	};

	Timer(
		const decltype(high_resolution_clock::now())& timer, 
		std::size_t time,
		mode mode = mode::WAIT_FINISH,
		const std::function<void()>& function = 
		std::function<void()>()
	) : timer(timer), time(time) {
		if (function) {
			functions.push_back(function);
		}
	}

	void add_function(mode mode, const std::function<void()>& function) {
		functions.push_back(function);
		modes.push_back(mode);
	}

	void run_functions() {
		for (auto i : modes) {
			switch (i) {
			case mode::WAIT_FINISH: {
				if (finished()) {
					for (auto i : functions) {
						i();
					}
					restart();
				}
				break;
			}
			case mode::EVERY_OTHER: {
				static bool exec = true;
				if (finished()) {
					for (auto i : functions) {
						i();
					}
					exec = !exec;
					restart();
				}
				if (exec) {
					for (auto i : functions) {
						i();
					}
				}
				break;
			}
			}
		}
	}

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
	std::vector<std::function<void()>> functions;
	std::vector<mode> modes;
	decltype(high_resolution_clock::now()) timer;
	std::size_t time;
};