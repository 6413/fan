module;

#include <fan/utility.h>
#include <cmath>
#include <functional>

#include <cstdio>

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

export module fan.time;

import fan.types;

export namespace fan {
	namespace time {
    // returns time in nanoseconds
    uint64_t now() {
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
    f64_t seconds() {
      return fan::time::now() / 1e9;
    }

		struct timer {
			timer() = default;
      explicit timer(uint64_t time, bool start_timer) {
        if (start_timer) {
          start(time);
        }
      }
      explicit timer(f64_t time, bool start_timer) {
        if (start_timer) {
          start(time);
        }
      }
      explicit timer(bool start_timer) {
        if (start_timer) {
          start();
        }
      }
      static timer seconds_timer(f64_t s) {
        timer t;
        t.start_seconds(s);
        return t;
      }

      static timer millis_timer(f64_t ms) {
        timer t;
        t.start_millis(ms);
        return t;
      }

      static timer micros_timer(f64_t us) {
        timer t;
        t.start_micros(us);
        return t;
      }

			constexpr uint64_t count() const {
				return m_time;
			}
      // Returns the total length of the timer
      constexpr uint64_t duration() const {
        return count();
      }
      constexpr f64_t duration_seconds() const {
        return duration() / 1e9;
      }
			void start() {
				m_timer = fan::time::now();
        m_time = -2;
			}
      void start(uint64_t time) {
        this->start();
        m_time = time;
      }
      void start_seconds(f64_t s) {
        start();
        m_time = (uint64_t)(s * 1e9);
      }
      void start_millis(f64_t ms) {
        start();
        m_time = (uint64_t)(ms * 1e6);
      }
      void start_micros(f64_t us) {
        start();
        m_time = (uint64_t)(us * 1e3);
      }

      void set_time(uint64_t time) {
        m_time = time;
      }
			void restart() {
        auto prev = m_time;
				start();
        m_time = prev;
			}
			bool finished() const {
				auto elapsed = this->elapsed();
				return elapsed >= m_time;
			}
      explicit operator bool const() {
        return finished();
      }
			bool started() const {
				return m_time != (uint64_t)-1;
			}
			// returns elapsed time since start in nanoseconds
			uint64_t elapsed() const {
				return m_timer == 0 ? 0 : fan::time::now() - m_timer;
			}
      // returns elapsed time since start in seconds
      double seconds() const {
        return elapsed() / 1e9;
      }
      double millis() const {
        return elapsed() / 1e6;
      }

      uint64_t start_time() const {
        return m_timer;
      }
      uint64_t start_time_seconds() const {
        return m_timer / 1e9;
      }

			uint64_t m_timer = 0;
			uint64_t m_time = (uint64_t)-1;
		};
    void delay(uint64_t time) {
#ifdef fan_platform_windows

      delay_w((float)(time / 1000));

#elif defined(fan_platform_unix)

      struct timespec t;

      t.tv_sec = time / 1000000000;
      t.tv_nsec = time % 1000000000;

      nanosleep(&t, 0);
#endif
    }

    struct scope_timer {
      fan::time::timer t;
      std::function<void(const fan::time::timer&)> cb;
      scope_timer(std::function<void(const fan::time::timer&)> cb_) :
        cb(std::move(cb_)) 
      {
        t.start();
      }
      ~scope_timer() {
        cb(t);
      }
    };
    struct scope_timer_print {
      scope_timer_print() :
        sc([](auto timer){
        printf("elapsed: %.2fms", timer.millis());
      }) {}
      scope_timer_print(const scope_timer_print&) = delete;
      scope_timer_print(scope_timer_print&&) = delete;
      scope_timer sc;
    };
	}
}