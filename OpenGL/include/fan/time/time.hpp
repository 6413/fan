#pragma once

#include <chrono>
#include <functional>
#include <thread>

namespace fan {
	using namespace std::chrono;

	typedef high_resolution_clock chrono_t;

	template <typename T = milliseconds>
	class timer {
	public:

		using value_type = chrono_t::duration::rep;
		using chrono_t_r_type = decltype(chrono_t::now());

		timer() : m_time(0) {}


		timer(
			const decltype(chrono_t::now())& timer, 
			uint64_t time = 0
		) : m_timer(timer), m_time(time) { }

		void start(int time) {
			this->m_timer = chrono_t::now();
			this->m_time = time;
		}

		static chrono_t_r_type start() {
			return chrono_t::now();
		}

		auto get_reset_time() {
			return m_time;
		}

		static auto get_time() {
			return time_point_cast<T>(system_clock::now()).time_since_epoch().count();
		}

		void restart() {
			this->m_timer = chrono_t::now();
		}

		bool finished() {
			return duration_cast<T>(chrono_t::now() - m_timer).count() >= m_time;
		}

		value_type elapsed() {
			return duration_cast<T>(chrono_t::now() - m_timer).count();
		}

	private:

		chrono_t_r_type m_timer;
		value_type m_time;
	};

	template <typename time_format>
	void delay(time_format time) {
		std::this_thread::sleep_for(time);
	}

}