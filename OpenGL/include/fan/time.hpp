#pragma once

#include <chrono>
#include <functional>

namespace fan {
	using namespace std::chrono;

	typedef high_resolution_clock chrono_t;

	template <typename T = milliseconds>
	class _timer {
	public:

		using value_type = chrono_t::duration::rep;

		_timer() : time(0) {}


		_timer(
			const decltype(chrono_t::now())& timer, 
			uint64_t time = 0
		) : timer(timer), time(time) { }

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
			return duration_cast<T>(chrono_t::now() - timer).count() >= time;
		}

		value_type elapsed() {
			return duration_cast<T>(chrono_t::now() - timer).count();
		}

	private:

		decltype(chrono_t::now()) timer;
		value_type time;
	};
	
	using timer = _timer<>;

}