#pragma once

#include <chrono>
#include <functional>

typedef std::chrono::high_resolution_clock chrono_t;

class Timer {
public:
	Timer() : time(0) {}

	enum class mode {
		WAIT_FINISH,
		EVERY_OTHER
	};

	Timer(
		const decltype(chrono_t::now())& timer, 
		uint64_t time,
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
		this->timer = chrono_t::now();
		this->time = time;
	}

	static decltype(chrono_t::now()) start() {
		return chrono_t::now();
	}

	void restart() {
		this->timer = chrono_t::now();
	}

	bool finished() {
		return duration_cast<std::chrono::milliseconds>(chrono_t::now() - timer).count() >= time;
	}

	uint64_t elapsed() {
		return duration_cast<std::chrono::milliseconds>(chrono_t::now() - timer).count();
	}

private:
	std::vector<std::function<void()>> functions;
	std::vector<mode> modes;
	decltype(chrono_t::now()) timer;
	uint64_t time;
};