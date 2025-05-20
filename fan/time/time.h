#pragma once

#include <fan/types/types.h>

#include <cmath>

#ifdef fan_platform_windows

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

typedef long(*NtDelayExecution_t)(int Alertable, PLARGE_INTEGER DelayInterval);
typedef long(* ZwSetTimerResolution_t)(IN ULONG RequestedResolution, IN BOOLEAN Set, OUT PULONG ActualResolution);

static NtDelayExecution_t NtDelayExecution = (NtDelayExecution_t)(long(__stdcall*)(BOOL, PLARGE_INTEGER)) GetProcAddress(GetModuleHandle("ntdll.dll"), "NtDelayExecution");
static ZwSetTimerResolution_t ZwSetTimerResolution = (ZwSetTimerResolution_t)(long(__stdcall*)(ULONG, BOOLEAN, PULONG)) GetProcAddress(GetModuleHandle("ntdll.dll"), "ZwSetTimerResolution");


static void delay_w(float us)
{
	static bool once = true;
	if (once) {
		ULONG actualResolution;
		ZwSetTimerResolution(1, true, &actualResolution);
		once = false;
	}

	LARGE_INTEGER interval;
	interval.QuadPart = (LONGLONG)(-10.f * us);
	NtDelayExecution(false, &interval);
}

#elif defined(fan_platform_unix)
#include <time.h>

#endif


namespace fan {

	namespace time {
		enum class time_unit {
			nanoseconds,
			microseconds,
			milliseconds,
			seconds,
			minutes,
			hours
		};

		constexpr bool operator<(const time_unit a, const time_unit b) {
			return (int)a < (int)b;
		}

		constexpr bool operator>(const time_unit a, const time_unit b) {
			return (int)a > (int)b;
		}

		template <time_unit>
		struct base_time_unit;

		using nanoseconds = base_time_unit<time_unit::nanoseconds>;
		using microseconds = base_time_unit<time_unit::microseconds>;
		using milliseconds = base_time_unit<time_unit::milliseconds>;
		using seconds = base_time_unit<time_unit::seconds>;
		using minutes = base_time_unit<time_unit::minutes>;
		using hours = base_time_unit<time_unit::hours>;

		// default e_time_unit::__first
		template <time_unit ratio>
		struct base_time_unit {

			constexpr base_time_unit() : time_unit_value(ratio) { }

			time_unit time_unit_value = ratio;

			constexpr uint64_t count() const {
				return m_time;
			}

			constexpr base_time_unit(uint64_t time) {
				switch (time_unit_value) {
					case time_unit::nanoseconds: {

						m_time = time;
						break;
					}
					case time_unit::microseconds: {

						m_time = time * 1000;
						break;
					}
					case time_unit::milliseconds: {

						m_time = time * 1000000;
						break;
					}
					case time_unit::seconds: {

						m_time = time * 1000000000;
						break;
					}
					case time_unit::minutes: {

						m_time = time * 60000000000;
						break;
					}
					case time_unit::hours: {

						m_time = time * 3600000000000;
						break;
					}
					default: {
						fan::throw_error_impl();
					}
				}

				time_unit_value = time_unit::nanoseconds;
			}

			uint64_t m_time = 0;
		};

		struct clock {

			clock() = default;
      explicit clock(uint64_t time, bool) {
        start(time);
      }

			template <time_unit unit>
			clock(const base_time_unit<unit>& time_unit_) {

				m_time = time_unit_.m_time;

				time_unit_value = time_unit_.time_unit_value;
			}

			constexpr uint64_t count() const {
				return m_time;
			}

			// returns time in nanoseconds
			static uint64_t now() {

#if defined(fan_platform_windows)

				LARGE_INTEGER freq;
				QueryPerformanceFrequency(&freq);
				double nanoseconds_per_count = 1.0e9 / static_cast<double>(freq.QuadPart);

				LARGE_INTEGER time;
				QueryPerformanceCounter(&time);

				return (uint64_t)((double)time.QuadPart * nanoseconds_per_count);

#elif defined(fan_platform_unix)

				struct timespec t;
				clock_gettime(CLOCK_MONOTONIC, &t);

				return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
#endif

			}

			void start() {
				m_timer = this->now();
			}

			template <time_unit unit>
			void start(const base_time_unit<unit>& time_unit_) {
				m_time = time_unit_.m_time;

				time_unit_value = time_unit_.time_unit_value;

				this->start();
			}

      void start(uint64_t time) {
        auto time_unit_ = fan::time::nanoseconds(time);
        m_time = time_unit_.m_time;

        time_unit_value = time_unit_.time_unit_value;

        this->start();
      }


			void restart() {
				start();
			}

			bool finished() const {
				auto elapsed = this->elapsed(m_timer);
				return elapsed >= m_time;
			}

			bool started() const {
				return m_time;
			}

			// returns time in nanoseconds
			uint64_t elapsed() const {
				return this->elapsed(m_timer);
			}

			// returns time in nanoseconds
			static uint64_t elapsed(uint64_t time) {
				return fan::time::clock::now() - time;
			}

			uint64_t m_timer;
			uint64_t m_time;

			time_unit time_unit_value;

		};

		template <typename new_t, typename old_t>
		constexpr uint64_t convert(old_t old, new_t x = new_t()) {
			int goal = (int)x.time_unit_value - (int)old.time_unit_value;
			f32_t divisionFactor = (f32_t)std::pow(10, 3 * goal);
			if (goal >= 4) divisionFactor *= 60;
			if (goal >= 5) divisionFactor *= 60;
			if (goal < 3) divisionFactor = (f32_t)std::pow(10, 3 * goal);
			if (goal > 0) return old.m_time / divisionFactor;
			else if (goal < 0) return old.m_time * (uint64_t)divisionFactor;
			else return old.m_time;
		}
	}

	static void delay(fan::time::nanoseconds time) {
#ifdef fan_platform_windows

		delay_w((float)(time.m_time / 1000));

#elif defined(fan_platform_unix)

		struct timespec t;

		t.tv_sec = time.m_time / 1000000000;
		t.tv_nsec = time.m_time % 1000000000;

		nanosleep(&t, 0);

#endif
	}

}

//struct event_timer_t
//{
//  template <class callable, class... arguments>
//  event_timer_t(uint64_t ns, bool async, callable&& f, arguments&&... args)
//  {
//    std::function<typename std::result_of<callable(arguments...)>::type()> task(std::bind(std::forward<callable>(f), std::forward<arguments>(args)...));
//
//    if (async)
//    {
//      std::thread([after, task]() {
//				fan::delay(ns);
//          task();
//      }).detach();
//    }
//    else
//    {
//      std::this_thread::sleep_for(std::chrono::milliseconds(after));
//      task();
//    }
//  }
//
//};
