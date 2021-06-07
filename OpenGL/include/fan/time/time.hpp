#pragma once

#include <chrono>
#include <functional>
#include <thread>

#ifdef FAN_PLATFORM_WINDOWS

	#define WIN32_LEAN_AND_MEAN

	#include <Windows.h>

	typedef long(*NtDelayExecution_t)(int Alertable, PLARGE_INTEGER DelayInterval);
	typedef long(* ZwSetTimerResolution_t)(IN ULONG RequestedResolution, IN BOOLEAN Set, OUT PULONG ActualResolution);

	static NtDelayExecution_t NtDelayExecution = (long(__stdcall*)(BOOL, PLARGE_INTEGER)) GetProcAddress(GetModuleHandle("ntdll.dll"), "NtDelayExecution");
	static ZwSetTimerResolution_t ZwSetTimerResolution = (long(__stdcall*)(ULONG, BOOLEAN, PULONG)) GetProcAddress(GetModuleHandle("ntdll.dll"), "ZwSetTimerResolution");


	static void delay_w(float us)
	{
		static bool once = true;
		if (once) {
			ULONG actualResolution;
			ZwSetTimerResolution(1, true, &actualResolution);
			once = false;
		}

		LARGE_INTEGER interval;
		interval.QuadPart = -10 * us;
		NtDelayExecution(false, &interval);
	}

#endif


namespace fan {

	using std::chrono::nanoseconds;
	using std::chrono::microseconds;
	using std::chrono::milliseconds;
	using std::chrono::seconds;
	using std::chrono::minutes;
	using std::chrono::hours;
	
	using std::chrono::duration_cast;
	using std::chrono::system_clock;
	using std::chrono::time_point_cast;

	typedef std::chrono::high_resolution_clock chrono_t;

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

		auto get_reset_time() const {
			return m_time;
		}

		static auto get_time() {
			return time_point_cast<T>(system_clock::now()).time_since_epoch().count();
		}

		auto time_left() const {
			return get_reset_time() - elapsed();
		}

		void restart() {
			this->m_timer = chrono_t::now();
		}

		bool finished() const {
			return duration_cast<T>(chrono_t::now() - m_timer).count() >= m_time;
		}

		value_type elapsed() const {
			return duration_cast<T>(chrono_t::now() - m_timer).count();
		}

	private:

		chrono_t_r_type m_timer;
		value_type m_time;
	};

	template <typename time_format>
	void delay(time_format time) {
		#ifdef FAN_PLATFORM_WINDOWS
			delay_w(std::chrono::duration_cast<fan::microseconds>(time).count());
		#else
			std::this_thread::sleep_for(time);
		#endif
	}

}