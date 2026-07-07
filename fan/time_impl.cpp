module;

#include <fan/utility.h>

#ifdef fan_platform_windows
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
  typedef long(*NtDelayExecution_t)(int Alertable, PLARGE_INTEGER DelayInterval);
  typedef long(* ZwSetTimerResolution_t)(IN ULONG RequestedResolution, IN BOOLEAN Set, OUT PULONG ActualResolution);
  static NtDelayExecution_t NtDelayExecution = (NtDelayExecution_t)(long(__stdcall*)(BOOL, PLARGE_INTEGER)) GetProcAddress(GetModuleHandle("ntdll.dll"), "NtDelayExecution");
  static ZwSetTimerResolution_t ZwSetTimerResolution = (ZwSetTimerResolution_t)(long(__stdcall*)(ULONG, BOOLEAN, PULONG)) GetProcAddress(GetModuleHandle("ntdll.dll"), "ZwSetTimerResolution");
  static void delay_w(float us) {
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

module fan.time;

import fan.print;

namespace fan {
  namespace time {
    std::uint64_t now() {
    #if defined(fan_platform_windows)
      LARGE_INTEGER freq;
      QueryPerformanceFrequency(&freq);
      double nanoseconds_per_count = 1.0e9 / static_cast<double>(freq.QuadPart);

      LARGE_INTEGER time;
      QueryPerformanceCounter(&time);
      return (std::uint64_t)((double)time.QuadPart * nanoseconds_per_count);
    #elif defined(fan_platform_unix)
      struct timespec t;
      clock_gettime(CLOCK_MONOTONIC, &t);

      return (std::uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
    #endif
    }

    f64_t seconds() {
      return fan::time::now() / 1e9;
    }

    void delay(std::uint64_t time) {
    #ifdef fan_platform_windows
      delay_w((float)(time / 1000));
    #elif defined(fan_platform_unix)
      struct timespec t;
      t.tv_sec = time / 1000000000;
      t.tv_nsec = time % 1000000000;
      nanosleep(&t, 0);
    #endif
    }

    scope_timer_print::scope_timer_print() {
      t.start();
    }

    scope_timer_print::~scope_timer_print() {
      std::printf("elapsed: %.2fms\n", t.millis());
    }
  }
  namespace time {
    bool& is_measuring() {
      static bool measure = false;
      return measure;
    }
    void set_measuring(bool state) {
      is_measuring() = state;
    }
    void print_measure(const std::string_view msg) {
      if (is_measuring()) {
        fan::print_dbg(msg);
      }
    }
    void print_measure(const std::string_view msg, f64_t time, const std::string_view unit) {
      if (is_measuring()) {
        fan::print_dbg(msg, time, unit);
      }
    }
    void measure(fan::time::timer& timer, const std::string_view msg) {
      if (is_measuring()) {
        fan::print_dbg(msg, "took:", timer.millis(), "ms");
      }
      timer.restart();
    }
  }
}